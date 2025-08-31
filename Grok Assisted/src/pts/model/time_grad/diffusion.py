"""diffusion.py

Module for diffusion processes in TimeGrad, handling forward and reverse diffusion for time series.
Fits project by providing the core DDPM mechanism for synthetic data generation.
"""

import math
from typing import Tuple

import torch
from pydantic import BaseModel, ValidationError, field_validator

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

if __name__ == "__main__":
    try:
        config = DiffusionConfig()
        diffusion = Diffusion(config)
        
        # Test with proper 3D tensors
        batch_size = 2
        seq_len = 10
        input_dim = 1
        
        x0 = torch.randn(batch_size, seq_len, input_dim)
        t = torch.tensor([50, 75])
        
        print(f"Input shapes:")
        print(f"  x0: {x0.shape}")
        print(f"  t: {t.shape}")
        
        xt, noise = diffusion.forward_process(x0, t)
        print(f"Output shapes:")
        print(f"  xt: {xt.shape}")
        print(f"  noise: {noise.shape}")
        print("Diffusion test passed!")
        
    except ValidationError as ve:
        print(f"Config error: {ve}")
    except Exception as e:
        print(f"Test error: {e}")