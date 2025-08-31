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
        self.config = config
        self.rnn = nn.LSTM(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)
        
        # Time embedding for diffusion timestep
        self.time_embed_dim = 32
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.ReLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim + self.time_embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def forward(self, xt: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy input xt, hidden h, time t.
        
        Args:
            xt: Noisy input tensor, shape (batch_size, seq_len, input_dim)
            h: Hidden state tuple (h, c), each shape (num_layers, batch_size, hidden_dim)  
            t: Time step tensor, shape (batch_size,)
        
        Returns:
            Predicted noise, shape (batch_size, seq_len, input_dim)
        """
        try:
            batch_size = xt.size(0)
            seq_len = xt.size(1)
            
            # Process through RNN
            rnn_out, (h_new, c_new) = self.rnn(xt, h)
            
            # Take the last hidden state for each sequence
            # rnn_out shape: (batch_size, seq_len, hidden_dim)
            last_hidden = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
            
            # Process time embedding
            # t should be shape (batch_size,) -> (batch_size, 1) for MLP
            t_reshaped = t.float().unsqueeze(-1)  # Shape: (batch_size, 1)
            time_embed = self.time_mlp(t_reshaped)  # Shape: (batch_size, time_embed_dim)
            
            # Concatenate hidden state and time embedding
            combined = torch.cat([last_hidden, time_embed], dim=-1)  # Shape: (batch_size, hidden_dim + time_embed_dim)
            
            # Predict noise
            noise_pred = self.noise_predictor(combined)  # Shape: (batch_size, input_dim)
            
            # Expand to match input sequence length
            noise_pred = noise_pred.unsqueeze(1).expand(-1, seq_len, -1)  # Shape: (batch_size, seq_len, input_dim)
            
            return noise_pred
            
        except RuntimeError as e:
            print(f"Network forward debug info:")
            print(f"  xt shape: {xt.shape}")
            print(f"  h[0] shape: {h[0].shape}, h[1] shape: {h[1].shape}")
            print(f"  t shape: {t.shape}")
            raise ValueError(f"Network forward failed: {str(e)}")

if __name__ == "__main__":
    config = NetworkConfig(input_dim=1)
    net = TimeGradNetwork(config)
    
    # Test with proper dimensions
    batch_size = 2
    seq_len = 10
    
    xt = torch.randn(batch_size, seq_len, 1)  # (batch_size, seq_len, input_dim)
    h = (
        torch.randn(2, batch_size, 40),  # (num_layers, batch_size, hidden_dim)
        torch.randn(2, batch_size, 40)   # (num_layers, batch_size, hidden_dim)
    )
    t = torch.tensor([50, 75])  # (batch_size,)
    
    print(f"Input shapes:")
    print(f"  xt: {xt.shape}")
    print(f"  h[0]: {h[0].shape}, h[1]: {h[1].shape}")
    print(f"  t: {t.shape}")
    
    output = net(xt, h, t)
    print(f"Output shape: {output.shape}")
    print("Network test passed!")