"""time_grad_estimator.py

Estimator for training TimeGrad model.
Handles loss computation and optimization for DDPM training in your project.
"""

from typing import Dict

import torch
from pydantic import BaseModel

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
            x0 = batch['target']
            t = torch.randint(0, self.diffusion.config.num_steps, (x0.size(0),))
            xt, noise = self.diffusion.forward_process(x0, t)
            predicted_noise = self.network(xt, batch['hidden'], t)
            loss = torch.mean((predicted_noise - noise)**2)  # MSE loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        except RuntimeError as e:
            raise ValueError(f"Train step failed: {str(e)}")

if __name__ == "__main__":
    net_config = NetworkConfig(input_dim=1)
    net = TimeGradNetwork(net_config)
    diff_config = DiffusionConfig()
    diffusion = Diffusion(diff_config)
    config = EstimatorConfig()
    estimator = TimeGradEstimator(config, net, diffusion)
    batch = {'target': torch.randn(64, 10, 1), 'hidden': (torch.randn(2, 64, 40), torch.randn(2, 64, 40))}
    loss = estimator.train_step(batch)
    print("Loss:", loss)