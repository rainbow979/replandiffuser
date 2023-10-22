import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import pdb

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import imageio
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange
from imageio import get_writer

import wandb

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, (4, 3), (2, 1), (1, 1))

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, (2, 1), 1)

    def forward(self, x):
        return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, mask, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []
        x = torch.cat([x, mask], dim=1)

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

    
    
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=[0,2])
    
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        self.inv_model = nn.Sequential(
            nn.Linear(16, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 8)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
        
    def model_predict(self, x, mask, t, cond, clip_denoised, wt=3.0):
        n = len(cond)
        cond = torch.Tensor(cond)
        uncond_eps = self.denoise_fn(x, mask, t, cond[:1], force_dropout=True)
        cond_eps = self.denoise_fn(x.expand(n, -1, -1), mask.expand(n, -1), t.expand(n), cond, use_dropout=False)

        cond_eps_pos = cond_eps[:1]
        cond_eps_neg = cond_eps[1:2]

        eps = uncond_eps + wt * (cond_eps_pos - uncond_eps)

        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        
        return x_recon

    def p_mean_variance(self, x, obs, t, clip_denoised: bool):
        
        eps = self.denoise_fn(x, obs, t)
        
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, obs, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, obs=obs, t=t, clip_denoised=clip_denoised)
        noise = 0.1 * noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, x, obs, conditions, samples=None, diffusion_step=None):
        if diffusion_step is None:
            diffusion_step = self.num_timesteps
        device = self.betas.device

        shape = x.shape
        b = shape[0]
        if samples is None:
            img = torch.randn(shape, device=device)
        else:            
            noise = torch.randn(shape, device=device)
            t = torch.full((b,), diffusion_step // 3, device=device, dtype=torch.long)
            img = self.q_sample(samples, t, noise).float()
            
        if conditions is not None:
            clen = conditions.size()[1]
            img[:, :clen, :8] = conditions[:, :, :8]

        for i in tqdm(reversed(range(0, diffusion_step)), desc='sampling loop time step', total=diffusion_step, leave=False):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), obs)
            if conditions is not None:
                img[:, :clen, :8] = conditions[:, :, :8]
        return img
    
    
        
    

    @torch.no_grad()
    def conditional_sample(self, obs, act_size, conditions, samples=None, diffusion_step=None):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = obs.size()[0]
        
        x = torch.zeros(act_size, device=device)
        
        return self.p_sample_loop(x, obs, conditions, samples, diffusion_step)


    def q_sample(self, x_start, t, noise=None):
        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
        return sample

    def p_losses(self, obs, state, t, act):
        x_start = state
        b, h, w = x_start.shape
        device = x_start.device
        noise = torch.randn(*x_start.size(), device=device)
        noise[:, 0, :8] = 0
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy[:, 0, :8] = x_start[:, 0, :8]
        x_recon = self.denoise_fn(x_noisy, obs, t)

        assert noise.shape == x_recon.shape
        losses = {}
        pos_dim = 0
        
        
        losses["position"] = F.mse_loss(noise[:, :, pos_dim:pos_dim+3], x_recon[:, :, pos_dim:pos_dim+3]) * 30
        losses['rotation'] = F.mse_loss(noise[:, :, pos_dim+3:pos_dim+7], x_recon[:, :, pos_dim+3:pos_dim+7]) * 3
        losses["gripper"] = F.mse_loss(noise[:, :, pos_dim+7:pos_dim+8], x_recon[:, :, pos_dim+7:pos_dim+8])            
        
        x = state[:, :-1]
        x_n = state[:, 1:]
        a_t = act[:, :-1]
        assert x.shape[1] == a_t.shape[1]
        x_c = torch.cat([x, x_n], -1).reshape(-1, 16)
        a_t = a_t.reshape(-1, 8)
        
        pred_a = self.inv_model(x_c)
        inv_loss = F.mse_loss(pred_a, a_t)
        
        losses['actions'] = inv_loss * 30
        
        
        losses["total_loss"] = sum(list(losses.values())) / 30
        
        return losses
        
    def calc_vlb(self, x_start, obs, cond_val, t=15):
        device = x_start.device
        noise = torch.randn_like(x_start).to(device)
        clen = cond_val.size()[1]
        noise[:, :clen, :] = 0
        t = torch.full((x_start.size()[0],), t, device=x_start.device, dtype=torch.long)        
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start, x_t, t)
        mean, _, log_variance_clipped = self.p_mean_variance(x=x_t, obs=obs, t=t, clip_denoised=True)
        
        kl = normal_kl(true_mean, true_log_variance_clipped, mean, log_variance_clipped)
        kl = mean_flat(kl) / np.log(2.0)
        return kl
    
    
    def forward(self, state, obs, act):
        batch = state.size()[0]
        device = state.device
        t = torch.randint(0, self.num_timesteps, (batch,), device=device).long()
        return self.p_losses(obs, state, t, act)

        
        
def conditional_noise(mask, x):
    mask = mask[:, :x.size(1)]
    noise = torch.randn_like(x)
    noise[mask.bool()] = 0
    return noise


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        device='cuda',
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if type(folder) == str:
            self.ds = Dataset(folder, image_size)
        else:
            self.ds = folder
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        #self.renderer = renderer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()
        
        self.device=device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        milestone = (milestone // 50) * 50
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        from collections import deque
        #queue = deque(maxlen=50)
        LQ = {}
        for l in tqdm(range(self.train_num_steps)):
        #while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                obs, act, state = next(self.dl)
                obs = obs.float().to(device)
                act = act.float().to(device)
                state = state.float().to(device)
                losses = self.model(state, obs, act)
                loss = losses["total_loss"]
                for key in losses.keys():
                    if key not in LQ:
                        LQ[key] = deque(maxlen=50)
                    try:
                        LQ[key].append(losses[key].cpu().item())
                    except:
                        import pdb
                        pdb.set_trace()
                #print('loss:', loss)
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if (self.step+1) % (self.save_and_sample_every // 5) == 0:
                for key in LQ.keys():
                    wandb.log({key: np.array(LQ[key]).mean()}, step=self.step)
                print(f'step {self.step}, loss {loss.cpu().item()}')
                
            if (self.step+1) % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
                


            self.step += 1

        print('training completed')

def to_np(x):
    return x.detach().cpu().numpy()

def plot_samples(savepath, samples_l, renderer):
    '''
        samples : [ B x 1 x H x obs_dim ]
    '''
    render_kwargs = {
        'trackbodyid': 2,
        'distance': 10,
        'lookat': [10, 2, 0.5],
        'elevation': 0
    }
    images = []
    # for samples in samples_l:
    #     ## [ H x obs_dim ]
    #     samples = samples.squeeze(0)
    samples = samples_l
    samples = samples.squeeze(0)
    imgs = renderer.composite(None, to_np(samples), dim=(1024, 256), qvel=True, render_kwargs=render_kwargs)

    savepath = savepath.replace('.png', '.mp4')
    writer = get_writer(savepath)

    for img in imgs:
        writer.append_data(img)

    writer.close()
