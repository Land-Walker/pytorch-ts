"""time_grad_predictor.py

Predictor for generating samples from trained TimeGrad.
Used for synthetic data inference in your RL pipeline.
"""

import torch

class TimeGradPredictor:
    """Predictor for generating samples from trained TimeGrad."""
    def __init__(self, network: TimeGradNetwork, diffusion: Diffusion):
        self.network = network
        self.diffusion = diffusion

    def predict(self, h: Tuple[torch.Tensor, torch.Tensor], num_samples: int = 100) -> torch.Tensor:
        """Generate samples using reverse process."""
        try:
            xt = torch.randn(num_samples, h[0].size(-1), device=h[0].device)  # Start from noise
            for t in range(self.diffusion.config.num_steps - 1, -1, -1):
                t_tensor = torch.full((num_samples,), t, device=xt.device)
                model_output = self.network(xt.unsqueeze(-1), h, t_tensor)  # Adjust for dim
                variance = self.diffusion.betas[t_tensor]  # Simplified
                xt = self.diffusion.reverse_process(xt.unsqueeze(-1), t_tensor, model_output, variance).squeeze(-1)
            return xt
        except RuntimeError as e:
            raise ValueError(f"Predict failed: {str(e)}")

if __name__ == "__main__":
    net = TimeGradNetwork(NetworkConfig(input_dim=1))
    diffusion = Diffusion(DiffusionConfig())
    predictor = TimeGradPredictor(net, diffusion)
    h = (torch.randn(2, 1, 40), torch.randn(2, 1, 40))
    samples = predictor.predict(h)
    print("Samples shape:", samples.shape)