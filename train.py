import numpy as np
import torch
import pdb

import gym
import d4rl

from denoising_diffusion_pytorch.test_model import BC
from denoising_diffusion_pytorch.datasets.rlbench_data import RLBenchDataset
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.mixer import MixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet, CondTemporalUnet
import sys

import wandb
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--env', default='wipe_desk')
    parser.add_argument('--dsteps', type=int, default=400)
    args = parser.parse_args()
    H = args.H
    dataset = RLBenchDataset(H, args.env)

    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim
    
    
    model = TemporalUnet(
        horizon = H,
        cond_dim = obs_dim,
        transition_dim = act_dim,
        dim = 128,
        dim_mults = (1, 2, 4, 8),
    ).cuda()

    print('dim:', obs_dim, act_dim)
    diffusion = GaussianDiffusion(
        model,
        channels = 1,
        image_size = (H, act_dim),        
        timesteps = args.dsteps,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).cuda()

    
    wandb.init(
        project="RLBench",        
        name = f'{args.env}_{H}',
    )
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size = 64,
        train_lr = 2e-5,
        train_num_steps = 600000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        fp16 = False,                     # turn on mixed precision training with apex
        results_folder = f'./logs/RLBench_{args.env}_{H}_{args.dsteps}',
    )

    trainer.train()


    
if __name__ == '__main__':
    main()