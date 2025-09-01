"""diffusion.py

Module for diffusion processes in TimeGrad, handling forward and reverse diffusion for time series.
Fits project by providing the core DDPM mechanism for synthetic data generation.
Enhanced with GaussianDiffusion class from the reference implementation.
"""

import math
from typing import Tuple
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from pydantic import BaseModel, ValidationError, field_validator

# Helper functions from the reference implementation
def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class DiffusionConfig(BaseModel):
    """Configuration for diffusion process."""
    num_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.1

    @field_validator('num_steps')
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_steps must be positive.")
        return v


class Diffusion:
    """Diffusion probabilistic model for denoising."""
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at time t.
        
        Args:
            x0: Clean data, shape (batch_size, seq_len, input_dim)
            t: Time steps, shape (batch_size,)
            
        Returns:
            xt: Noisy data, same shape as x0
            noise: Added noise, same shape as x0
        """
        try:
            # Generate noise with same shape as input
            noise = torch.randn_like(x0)
            
            # Get alpha values for each batch item
            # t shape: (batch_size,) -> need to broadcast to x0 shape
            sqrt_alphas_cumprod = torch.sqrt(self.alpha_cumprod[t])
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alpha_cumprod[t])
            
            # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1)
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)
            
            # Apply forward diffusion
            xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
            
            return xt, noise
            
        except RuntimeError as e:
            print(f"Forward process debug info:")
            print(f"  x0 shape: {x0.shape}")
            print(f"  t shape: {t.shape}")
            print(f"  noise shape: {noise.shape}")
            raise ValueError(f"Forward process failed: {str(e)}")

    def reverse_process(self, xt: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step using Langevin sampling approximation.
        
        Args:
            xt: Noisy input, shape (batch_size, seq_len, input_dim)
            t: Time steps, shape (batch_size,)
            model_output: Predicted noise from model, same shape as xt
            variance: Variance for sampling, shape (batch_size,)
            
        Returns:
            x_{t-1}: Less noisy sample, same shape as xt
        """
        try:
            # Get coefficients for this timestep
            beta_t = self.betas[t].view(-1, 1, 1)
            alpha_t = self.alphas[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
            
            # Compute mean of reverse process
            mean = (xt - beta_t * model_output / sqrt_one_minus_alpha_cumprod_t) / torch.sqrt(alpha_t)
            
            # Add noise (only if not final step)
            if torch.any(t > 0):
                noise = torch.randn_like(xt)
                # Only add noise where t > 0
                noise_mask = (t > 0).float().view(-1, 1, 1)
                variance_term = torch.sqrt(variance.view(-1, 1, 1)) * noise * noise_mask
            else:
                variance_term = 0
                
            return mean + variance_term
            
        except RuntimeError as e:
            print(f"Reverse process debug info:")
            print(f"  xt shape: {xt.shape}")
            print(f"  t shape: {t.shape}")
            print(f"  model_output shape: {model_output.shape}")
            print(f"  variance shape: {variance.shape}")
            raise ValueError(f"Reverse process failed: {str(e)}")


class GaussianDiffusion(nn.Module):
    """
    Enhanced GaussianDiffusion class from reference implementation.
    Provides more comprehensive diffusion functionality with multiple beta schedules
    and advanced sampling methods.
    """
    def __init__(
        self,
        denoise_fn,
        input_size,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.__scale = None

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape
        x_hat = self.p_sample_loop(shape, cond)  # TODO reshape x_hat to (B,T,-1)

        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def log_prob(self, x, cond, *args, **kwargs):
        if self.scale is not None:
            x /= self.scale

        B, T, _ = x.shape

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs
        )

        return loss


if __name__ == "__main__":
    try:
        # Test original Diffusion class
        config = DiffusionConfig()
        diffusion = Diffusion(config)
        
        # Test with proper 3D tensors
        batch_size = 2
        seq_len = 10
        input_dim = 1
        
        x0 = torch.randn(batch_size, seq_len, input_dim)
        t = torch.tensor([50, 75])
        
        print(f"Original Diffusion test:")
        print(f"Input shapes:")
        print(f"  x0: {x0.shape}")
        print(f"  t: {t.shape}")
        
        xt, noise = diffusion.forward_process(x0, t)
        print(f"Output shapes:")
        print(f"  xt: {xt.shape}")
        print(f"  noise: {noise.shape}")
        print("Original Diffusion test passed!")
        
        # Test GaussianDiffusion class
        print(f"\nGaussianDiffusion test:")
        
        # Create a dummy denoising function for testing
        class DummyDenoiser(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.linear = nn.Linear(input_size, input_size)
            
            def forward(self, x, t, cond=None):
                # Simple dummy implementation
                return self.linear(x)
        
        dummy_denoiser = DummyDenoiser(input_dim)
        gaussian_diffusion = GaussianDiffusion(
            denoise_fn=dummy_denoiser,
            input_size=input_dim,
            diff_steps=100,
            beta_schedule="cosine"
        )
        
        print(f"GaussianDiffusion initialized with {gaussian_diffusion.num_timesteps} timesteps")
        
        # Test forward sampling (q_sample)
        t_gauss = torch.randint(0, 100, (batch_size,))
        x_noisy = gaussian_diffusion.q_sample(x0.squeeze(-1), t_gauss)  # Remove last dim for compatibility
        print(f"Forward sample shape: {x_noisy.shape}")
        
        print("GaussianDiffusion test passed!")
        
    except ValidationError as ve:
        print(f"Config error: {ve}")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()