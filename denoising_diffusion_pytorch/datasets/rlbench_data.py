import collections
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset
from glob import glob

import pickle

import random


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)

    
class RLBenchDataset(Dataset):


    def __init__(self, H, env, max_path_length=1000, max_n_episodes=600):
        mode = 'clip'
        if env != 'wipe_desk':
            path = f'/home/zfchen/zsy/dataset/RLBench/{mode}/{env}+0/{mode}/{env}+0.pkl'
        else:
            import pdb
            pdb.set_trace()
            print(f'loading data from {env}')
            path = f'/home/zfchen/zsy/RLBench_data/clip/{env}+0.pkl'
        #path = f'/home/zfchen/zsy/dataset/RLBench/clip/{env}+0.pkl'
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        if mode == 'r3m':
            img_dim = 6144
        elif mode == 'clip':
            img_dim = 2304
        self.obs_dim = obs_dim = 8+img_dim
        self.act_dim = act_dim = 8
        #self.dataset = dataset
        self.env = env
        max_n_episodes = len(dataset)
        self.fields = {
            'observations': np.zeros((max_n_episodes, max_path_length, obs_dim)),
            'actions': np.zeros((max_n_episodes, max_path_length, act_dim)),
            'states': np.zeros((max_n_episodes, max_path_length, act_dim))
        }
        path_lengths = np.zeros(max_n_episodes, dtype=np.int32)
        for i, d in enumerate(dataset):
            path_length = len(d['actions'])
            self.fields['observations'][i, :path_length] = np.concatenate((d['poses'], d['imgs']), 1) 
            #self.fields['actions'][i, :path_length] = np.concatenate((d['poses'], d['actions']), 1)
            
            self.fields['states'][i, :path_length] = d['poses']
            self.fields['states'][i, path_length:path_length+H] = d['poses'][-1:]
            
            self.fields['actions'][i, 0:path_length] = np.concatenate((d['actions'], d['poses'][:, -1:]), -1)
            self.fields['actions'][i, path_length-1:path_length+H] = np.concatenate((np.zeros_like(d['actions'][-1:]), d['poses'][-1:, -1:]), -1)
            
            path_lengths[i] = path_length
            
        self.normalize()
        print('len info:', path_lengths[:len(dataset)].max(), path_lengths[:len(dataset)].mean(), path_lengths[:len(dataset)].std())
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))
        indices = np.array(indices)
        self.indices = indices
        
    def normalize(self, keys=['observations', 'actions', 'states']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        self.mins = {}
        self.maxs = {}
        for key in keys:
            mins = self.mins[key] = self.fields[key].min()
            maxs = self.maxs[key] = self.fields[key].max()
            
            ## [ 0, 1 ]
            self.fields[key] = (self.fields[key] - mins) / (maxs - mins + 1e-5)
            ## [ -1, 1 ]
            self.fields[key] = self.fields[key] * 2 - 1
            
    def unnormalize(self, x, key):
        mins = self.mins[key]
        maxs = self.maxs[key]
        x = (x + 1) / 2
        #assert x.max() <= 1.1 and x.min() >= -0.1, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins
    
    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]
        observations = self.fields['observations'][path_ind, start]
        states = self.fields['states'][path_ind, start:end]
        actions = self.fields['actions'][path_ind, start:end]
        #conditions = self.get_conditions(observations)
        observations = to_tensor(observations)
        actions = to_tensor(actions)
        return observations, actions, states
        
        
class SeqDataset(RLBenchDataset):
    def __init__(self, H, env, max_path_length=2000, max_n_episodes=600, type='all'):
        super().__init__(H, env, max_path_length, max_n_episodes=600)
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]
        observations = self.fields['observations'][path_ind, start]
        actions = self.fields['actions'][path_ind, start+1:end+1]
        #conditions = self.get_conditions(observations)
        observations = to_tensor(observations)
        actions = to_tensor(actions)
        return observations, actions
    
