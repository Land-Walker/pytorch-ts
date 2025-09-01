"""epsilon_theta.py

Advanced denoising network for TimeGrad using residual blocks and dilated convolutions.
Provides sophisticated noise prediction with diffusion step embeddings and conditional upsampling.
Alternative to the simple LSTM-based TimeGradNetwork for enhanced performance.
"""

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from pydantic import BaseModel, field_validator


class EpsilonThetaConfig(BaseModel):
    """Configuration for EpsilonTheta network."""
    target_dim: int
    cond_length: int = 1
    time_emb_dim: int = 16
    residual_layers: int = 8
    residual_channels: int = 32
    dilation_cycle_length: int = 2
    residual_hidden: int = 64
    
    @field_validator('residual_layers')
    @classmethod
    def validate_residual_layers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("residual_layers must be positive.")
        return v
    
    @field_validator('residual_channels')
    @classmethod
    def validate_residual_channels(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("residual_channels must be positive.")
        return v


class DiffusionEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""
    
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        """
        Args:
            diffusion_step: Tensor of shape (batch_size,) with timestep indices
        Returns:
            Embedded timesteps of shape (batch_size, proj_dim)
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        """Build sinusoidal position embeddings."""
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for temporal modeling."""
    
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        """
        Args:
            x: Input tensor (batch_size, residual_channels, sequence_length)
            conditioner: Conditioning tensor (batch_size, 1, sequence_length)
            diffusion_step: Diffusion embedding (batch_size, hidden_size)
        Returns:
            Tuple of (residual_output, skip_connection)
        """
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    """Upsampler for conditioning information."""
    
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        """
        Args:
            x: Conditioning tensor (batch_size, cond_length)
        Returns:
            Upsampled conditioning (batch_size, target_dim)
        """
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    """
    Advanced denoising network using residual blocks with dilated convolutions.
    Designed to predict noise in diffusion models with conditioning support.
    """
    
    def __init__(self, config: EpsilonThetaConfig):
        super().__init__()
        self.config = config
        
        self.input_projection = nn.Conv1d(
            1, config.residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = DiffusionEmbedding(
            config.time_emb_dim, proj_dim=config.residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=config.target_dim, cond_length=config.cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=config.residual_channels,
                    dilation=2 ** (i % config.dilation_cycle_length),
                    hidden_size=config.residual_hidden,
                )
                for i in range(config.residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(config.residual_channels, config.residual_channels, 3)
        self.output_projection = nn.Conv1d(config.residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond=None):
        """
        Forward pass predicting noise.
        
        Args:
            inputs: Noisy input tensor (batch_size, sequence_length, 1) or (batch_size, 1, sequence_length)
            time: Diffusion timesteps (batch_size,)
            cond: Optional conditioning tensor (batch_size, cond_length)
        
        Returns:
            Predicted noise (batch_size, sequence_length, 1) or (batch_size, 1, sequence_length)
        """
        try:
            # Handle input shape - convert to (batch_size, 1, sequence_length) if needed
            original_shape = inputs.shape
            if len(inputs.shape) == 3 and inputs.shape[-1] == 1:
                # Convert (batch_size, sequence_length, 1) -> (batch_size, 1, sequence_length)
                inputs = inputs.transpose(1, 2)
                need_transpose_back = True
            else:
                need_transpose_back = False
            
            # Project input
            x = self.input_projection(inputs)
            x = F.leaky_relu(x, 0.4)

            # Get diffusion step embedding
            diffusion_step = self.diffusion_embedding(time)
            
            # Handle conditioning
            if cond is None:
                # Create dummy conditioning if none provided
                batch_size = inputs.size(0)
                cond = torch.zeros(batch_size, self.config.cond_length, device=inputs.device)
            
            cond_up = self.cond_upsampler(cond)
            # Reshape conditioning for conv1d: (batch_size, cond_length) -> (batch_size, 1, cond_length)
            cond_up = cond_up.unsqueeze(1)
            
            # Process through residual layers
            skip = []
            for layer in self.residual_layers:
                x, skip_connection = layer(x, cond_up, diffusion_step)
                skip.append(skip_connection)

            # Combine skip connections
            x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
            x = self.skip_projection(x)
            x = F.leaky_relu(x, 0.4)
            x = self.output_projection(x)
            
            # Convert back to original shape if needed
            if need_transpose_back:
                x = x.transpose(1, 2)
            
            return x
            
        except RuntimeError as e:
            print(f"EpsilonTheta forward debug info:")
            print(f"  inputs shape: {inputs.shape}")
            print(f"  time shape: {time.shape}")
            if cond is not None:
                print(f"  cond shape: {cond.shape}")
            raise ValueError(f"EpsilonTheta forward failed: {str(e)}")


# Convenience function to create EpsilonTheta with TimeGrad-compatible interface
def create_timegrad_epsilon_theta(
    sequence_length: int,
    context_features: int = 1,
    residual_layers: int = 8,
    residual_channels: int = 32
) -> EpsilonTheta:
    """
    Create EpsilonTheta network compatible with TimeGrad pipeline.
    
    Args:
        sequence_length: Length of time series sequences
        context_features: Number of conditioning features
        residual_layers: Number of residual blocks
        residual_channels: Number of channels in residual blocks
    
    Returns:
        Configured EpsilonTheta network
    """
    config = EpsilonThetaConfig(
        target_dim=sequence_length,
        cond_length=context_features,
        residual_layers=residual_layers,
        residual_channels=residual_channels
    )
    return EpsilonTheta(config)


if __name__ == "__main__":
    try:
        # Test the EpsilonTheta network
        config = EpsilonThetaConfig(
            target_dim=24,  # sequence length
            cond_length=1,  # conditioning features
            residual_layers=4,
            residual_channels=16
        )
        
        network = EpsilonTheta(config)
        
        # Test with TimeGrad-compatible shapes
        batch_size = 2
        sequence_length = 24
        
        # Test with 3D input (batch_size, sequence_length, 1)
        inputs = torch.randn(batch_size, sequence_length, 1)
        time = torch.randint(0, 100, (batch_size,))
        cond = torch.randn(batch_size, 1)  # Optional conditioning
        
        print(f"Input shapes:")
        print(f"  inputs: {inputs.shape}")
        print(f"  time: {time.shape}")
        print(f"  cond: {cond.shape}")
        
        output = network(inputs, time, cond)
        print(f"Output shape: {output.shape}")
        
        # Test without conditioning
        output_no_cond = network(inputs, time)
        print(f"Output shape (no cond): {output_no_cond.shape}")
        
        # Test convenience function
        print(f"\nTesting convenience function:")
        network2 = create_timegrad_epsilon_theta(
            sequence_length=24,
            context_features=2,
            residual_layers=6
        )
        
        cond2 = torch.randn(batch_size, 2)
        output2 = network2(inputs, time, cond2)
        print(f"Output shape (convenience): {output2.shape}")
        
        print("EpsilonTheta network test passed!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()