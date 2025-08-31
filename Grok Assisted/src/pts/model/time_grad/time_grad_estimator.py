"""time_grad_estimator.py

Estimator for training TimeGrad model.
Handles loss computation and optimization for DDPM training in your project.
"""

from typing import Dict

import torch
from pydantic import BaseModel
from .time_grad_network import TimeGradNetwork
from .diffusion import Diffusion

class EstimatorConfig(BaseModel):
    """Config for TimeGrad estimator."""
    learning_rate: float = 1e-3
    batch_size: int = 64

class TimeGradEstimator:
    """Estimator for training TimeGrad."""
    def __init__(self, config: EstimatorConfig, network: TimeGradNetwork, diffusion: Diffusion):
        self.config = config
        self.network = network
        self.diffusion = diffusion
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        try:
            x0 = batch['target']  # Shape: (batch_size, seq_len, input_dim)
            hidden = batch['hidden']  # Tuple of tensors
            
            batch_size = x0.size(0)
            
            # Sample random timesteps for each item in batch
            t = torch.randint(0, self.diffusion.config.num_steps, (batch_size,))
            
            # Forward diffusion process
            xt, noise = self.diffusion.forward_process(x0, t)
            
            # Predict noise using network
            predicted_noise = self.network(xt, hidden, t)
            
            # Compute MSE loss
            loss = torch.mean((predicted_noise - noise)**2)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
            
        except RuntimeError as e:
            print(f"Train step debug info:")
            print(f"  x0 shape: {x0.shape}")
            print(f"  xt shape: {xt.shape}")
            print(f"  noise shape: {noise.shape}")
            print(f"  t shape: {t.shape}")
            if 'predicted_noise' in locals():
                print(f"  predicted_noise shape: {predicted_noise.shape}")
            raise ValueError(f"Train step failed: {str(e)}")

if __name__ == "__main__":
    try:
        from time_grad_network import NetworkConfig
        from diffusion import DiffusionConfig
    except ImportError:
        from .time_grad_network import NetworkConfig
        from .diffusion import DiffusionConfig
    
    net_config = NetworkConfig(input_dim=1)
    net = TimeGradNetwork(net_config)
    diff_config = DiffusionConfig()
    diffusion = Diffusion(diff_config)
    config = EstimatorConfig()
    estimator = TimeGradEstimator(config, net, diffusion)
    
    # Test batch with proper shapes
    batch = {
        'target': torch.randn(2, 10, 1),  # (batch_size, seq_len, input_dim)
        'hidden': (
            torch.randn(2, 2, 40),  # (num_layers, batch_size, hidden_dim)
            torch.randn(2, 2, 40)   # (num_layers, batch_size, hidden_dim)
        )
    }
    loss = estimator.train_step(batch)
    print("Loss:", loss)