"""time_grad_network.py

Network for TimeGrad, predicting noise conditioned on RNN hidden state.
Core for multivariate time series modeling in your DDPM adaptation.
"""

from typing import Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel

class NetworkConfig(BaseModel):
    """Config for TimeGrad network."""
    input_dim: int
    hidden_dim: int = 40
    num_layers: int = 2

class TimeGradNetwork(nn.Module):
    """TimeGrad noise prediction network."""
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.rnn = nn.LSTM(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)
        self.noise_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + 32, config.hidden_dim),  # + position embedding
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim)  # Predict noise
        )
        self.pos_embedding = nn.Parameter(torch.randn(32))  # Fourier positional

    def forward(self, xt: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy input xt, hidden h, time t."""
        try:
            _, (h, _) = self.rnn(xt, h)  # Update hidden
            pos = self.pos_embedding.repeat(xt.size(0), 1)  # Dummy pos
            input = torch.cat([h.squeeze(0), pos], dim=1)
            return self.noise_predictor(input)
        except RuntimeError as e:
            raise ValueError(f"Network forward failed: {str(e)}")

if __name__ == "__main__":
    config = NetworkConfig(input_dim=1)
    net = TimeGradNetwork(config)
    xt = torch.randn(2, 10, 1)
    h = (torch.randn(2, 2, 40), torch.randn(2, 2, 40))
    t = torch.tensor([50])
    output = net(xt, h, t)
    print("Output shape:", output.shape)