"""time_grad_predictor.py

Predictor for generating samples from trained TimeGrad.
Used for synthetic data inference in your RL pipeline.
"""

from typing import Tuple

import torch
from .time_grad_network import TimeGradNetwork, NetworkConfig
from .diffusion import Diffusion, DiffusionConfig

class TimeGradPredictor:
    """Predictor for generating samples from trained TimeGrad."""
    def __init__(self, network: TimeGradNetwork, diffusion: Diffusion):
        self.network = network
        self.diffusion = diffusion

    def predict(self, context_length: int, num_samples: int = 100, device: str = 'cpu') -> torch.Tensor:
        """Generate samples using reverse process.
        
        Args:
            context_length: Length of the sequence to generate
            num_samples: Number of samples to generate
            device: Device to run on
            
        Returns:
            Generated samples, shape (num_samples, context_length)
        """
        try:
            self.network.eval()
            
            with torch.no_grad():
                # Initialize with pure noise
                # Shape: (num_samples, context_length, input_dim)
                xt = torch.randn(num_samples, context_length, 1, device=device)
                
                # Create hidden states for the full batch
                # Shape: (num_layers, num_samples, hidden_dim)
                h_init = torch.zeros(
                    self.network.config.num_layers, 
                    num_samples, 
                    self.network.config.hidden_dim,
                    device=device
                )
                c_init = torch.zeros(
                    self.network.config.num_layers, 
                    num_samples, 
                    self.network.config.hidden_dim,
                    device=device
                )
                hidden = (h_init, c_init)
                
                # Reverse diffusion process
                for step in range(self.diffusion.config.num_steps - 1, -1, -1):
                    # Create time tensor for full batch
                    t = torch.full((num_samples,), step, device=device)
                    
                    # Predict noise
                    predicted_noise = self.network(xt, hidden, t)
                    
                    # Compute variance (simplified)
                    if step > 0:
                        variance = self.diffusion.betas[step].expand(num_samples)
                    else:
                        variance = torch.zeros(num_samples, device=device)
                    
                    # Reverse step
                    xt = self.diffusion.reverse_process(xt, t, predicted_noise, variance)
                
                # Return squeezed samples (remove the last dimension)
                # Shape: (num_samples, context_length)
                return xt.squeeze(-1)
                
        except RuntimeError as e:
            print(f"Prediction debug info:")
            print(f"  num_samples: {num_samples}")
            print(f"  context_length: {context_length}")
            print(f"  xt shape: {xt.shape}")
            if 'hidden' in locals():
                print(f"  hidden[0] shape: {hidden[0].shape}")
                print(f"  hidden[1] shape: {hidden[1].shape}")
            if 't' in locals():
                print(f"  t shape: {t.shape}")
            raise ValueError(f"Predict failed: {str(e)}")

if __name__ == "__main__":
    # Test the predictor
    net = TimeGradNetwork(NetworkConfig(input_dim=1))
    diffusion = Diffusion(DiffusionConfig())
    predictor = TimeGradPredictor(net, diffusion)
    
    # Test prediction
    samples = predictor.predict(context_length=24, num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
    print("Predictor test passed!")