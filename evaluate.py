import numpy as np
import torch
import pdb

import gym
#import d4rl

from denoising_diffusion_pytorch.test_model import BC, RNNGMM
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.datasets.rlbench_data import RLBenchDataset, SeqDataset
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.mixer import MixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet, CondTemporalUnet
from denoising_diffusion_pytorch import utils
import sys
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel
from utils import RLBenchEnv
from PIL import Image
import time
from rlbench.backend.utils import task_file_to_task_class

import argparse

import random

max_lens_task = {
    'open_box': 250,
    'wipe_desk': 450,
    'close_fridge': 200,
    'place_cups': 450,
    'close_box': 400
}



class Vis_Clip:
    def __init__(self):
        self.img_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def parse_state(self, obs, env, get_features=False):
        pos_step =  np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        cur_state, prev_action = env.get_rgb_pos_action(obs)
        clip_image_features = None
        if get_features:
            img_step = cur_state['rgb']
            clip_image_features = []
            for img in img_step:
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                inputs = self.img_processor(images=img, return_tensors="pt")
                outputs = self.img_encoder(**inputs)
                # last_hidden_state = outputs.last_hidden_state # [1,50,768]
                pooled_output = outputs.pooler_output  # pooled CLS states, [1,768]
                clip_image_features.append(pooled_output)
            clip_image_features = torch.cat(clip_image_features, dim=1) # [1, 2304]
        return pos_step, clip_image_features
        
class Vis_R3M:
    def __init__(self, device):
        from r3m import load_r3m
        self.vis_model = load_r3m('resnet50')
        self.vis_model.eval()
        self.vis_model.to(device)
        self.device = device
    
    def parse_state(self, obs, env, get_features=False):
        pos_step =  np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        cur_state, prev_action = env.get_rgb_pos_action(obs)
        image_features = None
        if get_features:
            img_step = cur_state['rgb']
            image_features = []
            for img in img_step:
                import omegaconf
                import hydra
                import torchvision.transforms as T

                ## DEFINE PREPROCESSING
                transforms = T.Compose([T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor()]) # ToTensor() divides by 255
                    
                preprocessed_image = transforms(Image.fromarray(img.astype(np.uint8))).reshape(-1, 3, 224, 224)
                preprocessed_image.to(self.device) 
                with torch.no_grad():
                    embedding = self.vis_model(preprocessed_image * 255.0).detach().cpu() ## R3M expects image input to be [0-255]
                    image_features.append(embedding)
            image_features = torch.cat(image_features, dim=1)
        return pos_step, image_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--env', default='open_box')
    #chioce: open_box
    parser.add_argument('--dsteps', type=int, default=400)
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--recorder', default=False, action='store_true')
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--num_eval', default=10, type=int)
    parser.add_argument('--small_interval', default=10, type=int)
    parser.add_argument('--model', default='diffusion')
    parser.add_argument('--act', default='IK')
    parser.add_argument('--vlb', default=False, action='store_true')
    parser.add_argument('--replan', default=0, type=int)
    parser.add_argument('--threshold', default=0.004, type=float)
    parser.add_argument('--small_threshold', default=0.001, type=float)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    
    vis_mode = 'clip'
    
    device = torch.device('cuda:1')
    vis_model = Vis_Clip()    
    H = args.H
    if args.model == 'diffusion':
        dataset = RLBenchDataset(H, args.env)
    else:
        dataset = SeqDataset(H, args.env)

    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim
    
    model = TemporalUnet(
        horizon = H,
        cond_dim = obs_dim,
        transition_dim = act_dim,
        dim = 128,
        dim_mults = (1, 2, 4, 8),
    ).to(device)

    print('dim:', obs_dim, act_dim)
    if args.model == 'rnn':
        diffusion = RNNGMM(act_dim, obs_dim, horizon=args.H).to(device)
    elif H > 1:
        diffusion = GaussianDiffusion(
            model,
            channels = 1,
            image_size = (H, act_dim),        
            timesteps = args.dsteps,   # number of steps
            loss_type = 'l1'    # L1 or L2
        ).to(device)
    else:
        diffusion = BC(
            act_dim,
            obs_dim,
        ).to(device)
    
    
    act_folder = 'actlog'
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size = 64,
        train_lr = 2e-5,
        train_num_steps = 300000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        fp16 = False,                     # turn on mixed precision training with apex
        results_folder = f'./{act_folder}/RLBench_{args.env}_{H}_{args.dsteps}',
        device=device
    )
    diffusion_epoch = args.epoch
    trainer.load(diffusion_epoch)
    
    trainer.ema_model.eval()
    trainer.model.eval()
    
    
    def make_env():
        return RLBenchEnv(
            data_path="",
            apply_rgb=True,
            apply_pc=False,
            apply_depth=False,
            apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
            static_positions=False,
            headless=(args.vis==False),
            act=args.act,
        )
    env = make_env()
    try:
        task = args.env
        print('task', task)
        task_type = task_file_to_task_class(task)
        
        task_env_list = []
        task_env = env.env.get_task(task_type)
        task_env.set_variation(0)
        
        from save_videos import Recorder
        recorder = Recorder()
        
        
        total_rewards = []
        total_lengths = []
        total_diffusion_steps = 0
        state_error_mean = np.zeros((1000, 50))
        vlb_mean = np.zeros((1000, 50))
        fail_idx = []
        rnn_states = None
        Replan_flag = False
        for idx in range(args.num_eval):
            instrs, obs = task_env.reset()
            
            pos_step, clip_image_features = vis_model.parse_state(obs, env, get_features=False)
            
            
            
            max_len = max_lens_task[args.env]
            
            episode_step = 0
            episode_rewards = 0
            
            interval = args.interval
            last_update = -3*H
            small_interval = args.small_interval
            last_small_update = last_update
            
            fail_action = 0
            vlb_value = 0
            state_error = 0
            
            act_size = (1, H, act_dim)
            
            
            trainer.model.eval()
            last_features = None
            print('begin:', idx)

            while episode_step < max_len:
                #print(vlb_value)
                if H > 50:
                    if args.replan < 2:
                        Replan_flag = (args.vlb and vlb_value > args.threshold)
                        Fast_flag = (args.vlb and vlb_value > args.small_threshold and episode_step - last_small_update >= args.small_interval)
                    else:
                        Replan_flag = (args.vlb and state_error > args.threshold)
                        Fast_flag = (args.vlb and state_error > args.small_threshold and episode_step - last_small_update >= args.small_interval)

                    if episode_step - last_update >= interval or Replan_flag:
                        pos_step, clip_image_features = vis_model.parse_state(obs, env, get_features=True)
                        features = torch.cat([torch.tensor(pos_step)[None], clip_image_features],1 ).float()
                        features = normalize(dataset, features).to(device)
                        last_features = features.clone()
                        conditions = normalize(dataset, pos_step, 'states')[None, None, :]
                        conditions = torch.tensor(conditions).to(device)
                        vlb_conditions = conditions.clone()
                        samples_nor = trainer.ema_model.conditional_sample(features, act_size, conditions)
                        last_update = episode_step-1
                        last_small_update = last_update
                        total_diffusion_steps += args.dsteps
                        samples = dataset.unnormalize(samples_nor[0], 'states')
                    elif Fast_flag:
                        pos_step, clip_image_features = vis_model.parse_state(obs, env, get_features=True)
                        features = torch.cat([torch.tensor(pos_step)[None], clip_image_features],1 ).float()
                        features = normalize(dataset, features).to(device)
                        conditions = normalize(dataset, pos_step, 'states')[None, None, :]
                        conditions = torch.tensor(conditions, device=device)
                        if args.replan == 0:
                            vlb_conditions = conditions.clone()
                        if args.replan == 0:
                            samples_nor = torch.cat([conditions, samples_nor[:, episode_step - last_small_update:, :], samples_nor[:, -1:, :].repeat(1, episode_step-last_small_update-1, 1)], 1)
                            samples_nor = trainer.ema_model.conditional_sample(features, act_size, conditions, samples=samples_nor, diffusion_step=100)
                        else:
                            samples_nor = trainer.ema_model.conditional_sample(last_features, act_size, vlb_conditions, samples=samples_nor, diffusion_step=100)
                        total_diffusion_steps += 100
                        last_small_update = episode_step - 1
                        samples = dataset.unnormalize(samples_nor[0], 'states')
                        
                        
                        
                    if args.act == 'IK':
                        if args.replan ==0:
                            action = samples[episode_step - last_small_update].data.cpu().numpy()
                        else:
                            action = samples[episode_step - last_update].data.cpu().numpy()
                    elif args.act == 'joint_v':
                        temp_conditions = normalize(dataset, pos_step, 'states')[None, :]
                        temp_conditions = torch.tensor(temp_conditions).to(device).float()
                        state_comb = torch.cat([temp_conditions, samples_nor[0, episode_step - last_small_update][None, :].float()], -1)
                        action = trainer.model.inv_model(state_comb)[0]
                        action = dataset.unnormalize(action, 'actions').data.cpu().numpy()
                        
                    next_state = samples[episode_step - last_small_update].data.cpu().numpy()
                elif H > 5:
                    pos_step, clip_image_features = vis_model.parse_state(obs, env, get_features=True)
                    features = torch.cat([torch.tensor(pos_step)[None], clip_image_features],1 ).float()
                    features = normalize(dataset, features).to(device)[:, None, :]
                    if episode_step % H == 0:
                        rnn_states = None
                    action, rnn_states = trainer.model.get_action(features, rnn_states)
                    action = action[0, ]
                    action = dataset.unnormalize(action, 'actions').cpu().numpy()
                else:
                    pos_step, clip_image_features = vis_model.parse_state(obs, env, get_features=True)
                    features = torch.cat([torch.tensor(pos_step)[None], clip_image_features],1 ).float()
                    features = normalize(dataset, features).to(device)
                    conditions = normalize(dataset, pos_step, 'states')[None, None, :]
                    conditions = torch.tensor(conditions).to(device)
                    action = trainer.model.get_action(features)[0, 0].data
                    action = dataset.unnormalize(action, 'states').cpu().numpy()
                
                
                if args.act == 'IK':
                    action[3:7] = normalise_quat(action[3:7])
                    
                if action[-1] < 0.5:
                    action[-1] = 0
                else:
                    action[-1] = 1
                    
                try:                
                    obs, reward, terminate = task_env.step(action)
                except Exception as e:
                    print(e)
                    if H > 100:
                        print(episode_step, pos_step, action)
                    fail_action += 1
                    if fail_action <= 3:
                        vlb_value = 1
                        continue
                    episode_step = max_len
                    
                    break
                
                if args.recorder:
                    recorder.add()
                
                episode_rewards += reward
                
                episode_step += 1

                if terminate:
                    if episode_rewards == 0:
                        episode_step = max_len
                    break
            
                
                pos_step, clip_image_features = vis_model.parse_state(obs, env, get_features=True)
                features = torch.cat([torch.tensor(pos_step)[None], clip_image_features], 1).float()
                features = normalize(dataset, features).to(device)
                
                nor_pos_step = pos_step.copy()
                if H > 100:
                    state_error = ((np.array(nor_pos_step[:3]) - next_state[:3])**2).sum()**0.5
                    state_error_mean[episode_step-1][idx] = state_error
                    
                    temp_conditions = normalize(dataset, pos_step, 'states')[None, None, :]
                    temp_conditions = torch.tensor(temp_conditions).to(device)
                    if vlb_conditions is not None:
                        vlb_conditions = torch.cat([vlb_conditions, temp_conditions], 1)
                    else:
                        vlb_conditions = temp_conditions.clone()
                    if args.replan == 0:
                        vlb_samples_nor = torch.cat([temp_conditions, samples_nor[:, episode_step - last_small_update:, :], samples_nor[:, -1:, :].repeat(1, episode_step-last_small_update-1, 1)], 1)
                        vlb_value = []
                        for t in [5, 10, 15]:
                            vlb_value.append(trainer.ema_model.calc_vlb(vlb_samples_nor.float(), features.float(), temp_conditions.float(), t=t).data.cpu().numpy())
                    else:
                        vlb_value = []
                        for t in [5, 10, 15]:
                            vlb_value.append(vlb_value = trainer.ema_model.calc_vlb(samples_nor.float(), last_features.float(), vlb_conditions.float(), t=15).data.cpu().numpy())
                    vlb_value = np.array(vlb_value)
                    vlb_mean[episode_step-1][idx] = vlb_value = vlb_value.mean()
                
            print(episode_rewards, episode_step)
            total_rewards.append(episode_rewards)
            total_lengths.append(episode_step)
            if args.recorder:
                recorder.save(f'./videos/{args.env}/{idx}_{H}_{interval}_{args.act}_{args.vlb}.mp4')
        print('mean reward:', np.mean(total_rewards))
        print('mean length:', np.mean(total_lengths))
        print('diffusion step:', total_diffusion_steps / args.num_eval)
        print(fail_idx)
        
        
    finally:
        try:
            env.env.shutdown()
        except:
            print('close error')
def normalise_quat(x):
    return x / (((x**2).sum(-1)**0.5)[..., None])
    
    
def parse_state(obs, env, vis_model, device, get_features=False):
    pos_step =  np.concatenate([obs.gripper_pose, [obs.gripper_open]])
    cur_state, prev_action = env.get_rgb_pos_action(obs)
    image_features = None
    if get_features:
        img_step = cur_state['rgb']
        image_features = []
        for img in img_step:
            import omegaconf
            import hydra
            import torchvision.transforms as T

            ## DEFINE PREPROCESSING
            transforms = T.Compose([T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor()]) # ToTensor() divides by 255
                
            preprocessed_image = transforms(Image.fromarray(img.astype(np.uint8))).reshape(-1, 3, 224, 224)
            preprocessed_image.to(device) 
            with torch.no_grad():
                embedding = vis_model(preprocessed_image * 255.0).detach().cpu() ## R3M expects image input to be [0-255]
                image_features.append(embedding)
        image_features = torch.cat(image_features, dim=1)
    return pos_step, image_features


def normalize(dataset, x, key = 'observations'):
    mins = dataset.mins[key]
    maxs = dataset.maxs[key]

    x = (x - mins) / (maxs - mins + 1e-5)
    x = x * 2 - 1
    return x
    
if __name__ == '__main__':
    main()




