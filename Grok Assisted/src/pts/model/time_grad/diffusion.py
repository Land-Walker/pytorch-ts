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
        """Forward diffusion: add noise at time t."""
        try:
            noise = torch.randn_like(x0)
            sqrt_alphas_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
            xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
            return xt, noise
        except RuntimeError as e:
            raise ValueError(f"Forward process failed: {str(e)}")

    def reverse_process(self, xt: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step using Langevin sampling approximation."""
        try:
            beta_t = self.betas[t].view(-1, 1, 1)
            alpha_t = self.alphas[t].view(-1, 1, 1)
            mean = (xt - beta_t * model_output / math.sqrt(1 - self.alpha_cumprod[t].view(-1, 1, 1))) / math.sqrt(alpha_t)
            noise = torch.randn_like(xt) if t > 0 else 0
            return mean + torch.sqrt(variance) * noise
        except RuntimeError as e:
            raise ValueError(f"Reverse process failed: {str(e)}")

if __name__ == "__main__":
    try:
        config = DiffusionConfig()
        diffusion = Diffusion(config)
        x0 = torch.randn(2, 10)  # Batch of time series
        t = torch.tensor([50, 50])
        xt, noise = diffusion.forward_process(x0, t)
        print("Forward output shape:", xt.shape)
    except ValidationError as ve:
        print(f"Config error: {ve}")