import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.distributions as D

class BC(nn.Module):
    def __init__(self, transition_dim, cond_dim):
        dim = 128
        super().__init__()
        #cond_dim = 8
        self.base = nn.Sequential(
            nn.Linear(cond_dim, dim * 8),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(dim * 8, dim * 4),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(dim * 4, dim),            
        )
        self.final = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, 8),
        )
        self.image_size = (1, 8)
        
    def get_action(self, obs):
        pred = self.final(self.base(obs))[:, None, :]
        return pred
        
    def forward(self, act, obs):
        pred = self.final(self.base(obs))[:, None, :]
        #act = act[:, :, :]
        #return (pred - act).abs().mean()
        
        
        losses = {}
        losses["position"] = F.mse_loss(pred[:, :, :3], act[:, :, :3]) * 30
        losses['rotation'] = F.mse_loss(pred[:, :, 3:7], act[:, :, 3:7]) * 3
        # losses['rotation'] = self.compute_rotation_loss(act_pred[:, :, 3:7], action_target[:, :, 3:7])
        losses["gripper"] = F.mse_loss(pred[:, :, 7:8], act[:, :, 7:8])            
        # print('rot test', act_pred[4,3,3:7].square().sum(), 'target_test', action_target[2,8, 3:7].square().sum())
        # print('gripper', act_pred[4,3,7:8].square().sum(), 'target_test', action_target[2,8, 7:8].square().sum())
        # input()
        losses["total_loss"] = sum(list(losses.values())) / 30
        
        return losses
        
class RNNGMM(nn.Module):
    def __init__(self, transition_dim, cond_dim, num_modes=5, horizon=20):
        dim = 400
        super().__init__()
        #cond_dim = 8
        self.base = nn.GRU(
            input_size = cond_dim,
            hidden_size=dim,
            num_layers=2,
            batch_first=True,            
        )
        
        self.mean_mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Mish(),
            nn.Linear(128, 8 * num_modes),
        )
        self.std_mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Mish(),
            nn.Linear(128, 8 * num_modes),
        )
        self.logits_mlp = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Mish(),
            nn.Linear(128, num_modes),
        )
        self.image_size = (1, 8)
        self.num_modes = num_modes
        self.min_std = 0.0001
        
    def forward(self, act, obs, mask):
        assert len(obs.shape) == 3
        B, T, _ = obs.shape
        
        feats = self.base(obs)[0]
        mean = self.mean_mlp(feats).view(-1, self.num_modes, 8)
        mean = mean.clamp_(-1, 1)
        std = self.std_mlp(feats).view(-1, self.num_modes, 8)
        logits = self.logits_mlp(feats).view(-1, self.num_modes)
        
        std = F.softplus(std) + self.min_std
        
        #import pdb
        #pdb.set_trace()
        
        g_dist = D.Normal(loc=mean, scale=std)
        g_dist = D.Independent(g_dist, 1)
        
        mixture_dist = D.Categorical(logits=logits)
        
        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=g_dist
        )
        
        act = act.view(-1, 8)
        log_probs = dists.log_prob(act)
        
        losses = {}
        #losses["position"] = -log_probs[:, 0:3].mean() * 30
        #losses['rotation'] = -log_probs[:, 3:7].mean() * 3
        #losses["gripper"] = -log_probs[:, 7:8].mean()            
        #losses["total_loss"] = sum(list(losses.values())) / 30
        losses["total_loss"] = -log_probs.mean()
        return losses
        
    def get_action(self, obs, rnn_states=None):
        if len(obs.shape) == 2:
            obs = obs[:, None]
        if rnn_states is not None:
            feats, rnn_states = self.base(obs, rnn_states)
        else:
            feats, rnn_states = self.base(obs)        
        mean = self.mean_mlp(feats).view(-1, self.num_modes, 8)
        std = self.std_mlp(feats).view(-1, self.num_modes, 8)
        logits = self.logits_mlp(feats).view(-1, self.num_modes)
        
        std = F.softplus(std) + self.min_std
        
        g_dist = D.Normal(loc=mean, scale=std)
        g_dist = D.Independent(g_dist, 1)
        
        mixture_dist = D.Categorical(logits=logits)
        
        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=g_dist
        )
        
        return dists.sample(), rnn_states
