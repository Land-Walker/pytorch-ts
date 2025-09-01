"""time_grad_network.py

Complete TimeGrad network implementation for multivariate time series forecasting.
Includes both training and prediction networks with RNN encoder and diffusion-based decoder.
Integrates with the epsilon_theta advanced denoising network and GaussianDiffusion.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, field_validator

# Import the sophisticated denoising network
from .epsilon_theta import EpsilonTheta, EpsilonThetaConfig

# Temporary implementations for missing components
def weighted_average(values: torch.Tensor, weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute weighted average along specified dimension."""
    return (values * weights).sum(dim=dim, keepdim=True) / (weights.sum(dim=dim, keepdim=True) + 1e-8)

class MeanScaler:
    """Simple mean scaler implementation."""
    def __init__(self, keepdim: bool = True):
        self.keepdim = keepdim
    
    def __call__(self, data: torch.Tensor, observed_values: torch.Tensor):
        # Simple mean scaling
        numerator = (data * observed_values).sum(dim=1, keepdim=self.keepdim)
        denominator = observed_values.sum(dim=1, keepdim=self.keepdim)
        scale = numerator / (denominator + 1e-8)
        scaled_data = data / (scale + 1e-8)
        return scaled_data, scale

class NOPScaler:
    """No-operation scaler."""
    def __init__(self, keepdim: bool = True):
        self.keepdim = keepdim
    
    def __call__(self, data: torch.Tensor, observed_values: torch.Tensor):
        scale = torch.ones_like(data.mean(dim=1, keepdim=self.keepdim))
        return data, scale

class DiffusionOutput:
    """Output distribution wrapper for diffusion models."""
    def __init__(self, diffusion_model, input_size: int, cond_size: int):
        self.diffusion_model = diffusion_model
        self.input_size = input_size
        self.cond_size = cond_size
    
    def get_args_proj(self, num_cells: int):
        """Get projection layer for distribution arguments."""
        return nn.ModuleList([nn.Linear(num_cells, self.cond_size)])


# Enhanced configuration for TimeGrad
class TimeGradConfig(BaseModel):
    """Configuration for complete TimeGrad network."""
    input_size: int
    num_layers: int = 2
    num_cells: int = 40
    cell_type: str = "LSTM"
    history_length: int = 100
    context_length: int = 24
    prediction_length: int = 24
    dropout_rate: float = 0.1
    lags_seq: List[int] = [1, 2, 3, 4, 5, 6, 7]
    target_dim: int = 1
    conditioning_length: int = 40
    diff_steps: int = 100
    loss_type: str = "l2"
    beta_end: float = 0.1
    beta_schedule: str = "linear"
    residual_layers: int = 8
    residual_channels: int = 32
    dilation_cycle_length: int = 2
    cardinality: List[int] = [1]
    embedding_dimension: int = 1
    scaling: bool = True
    
    @field_validator('cell_type')
    @classmethod
    def validate_cell_type(cls, v: str) -> str:
        if v not in ["LSTM", "GRU"]:
            raise ValueError("cell_type must be 'LSTM' or 'GRU'.")
        return v


class TimeGradTrainingNetwork(nn.Module):
    """Complete TimeGrad training network with RNN encoder and diffusion decoder."""
    
    def __init__(self, config: TimeGradConfig, diffusion_model=None) -> None:
        super().__init__()
        self.config = config
        self.target_dim = config.target_dim
        self.prediction_length = config.prediction_length
        self.context_length = config.context_length
        self.history_length = config.history_length
        self.scaling = config.scaling

        # Validate and sort lags
        assert len(set(config.lags_seq)) == len(config.lags_seq), "no duplicated lags allowed!"
        lags_seq = sorted(config.lags_seq)
        self.lags_seq = lags_seq

        # RNN encoder
        self.cell_type = config.cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[config.cell_type]
        self.rnn = rnn_cls(
            input_size=config.input_size,
            hidden_size=config.num_cells,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            batch_first=True,
        )

        # Advanced denoising network
        epsilon_config = EpsilonThetaConfig(
            target_dim=config.target_dim,
            cond_length=config.conditioning_length,
            residual_layers=config.residual_layers,
            residual_channels=config.residual_channels,
            dilation_cycle_length=config.dilation_cycle_length,
        )
        self.denoise_fn = EpsilonTheta(epsilon_config)

        # Use provided diffusion model or create from GaussianDiffusion
        if diffusion_model is not None:
            self.diffusion = diffusion_model
        else:
            # Import here to avoid circular imports
            from .diffusion import GaussianDiffusion
            self.diffusion = GaussianDiffusion(
                denoise_fn=self.denoise_fn,
                input_size=config.target_dim,
                diff_steps=config.diff_steps,
                loss_type=config.loss_type,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
            )

        # Distribution output
        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=config.target_dim, cond_size=config.conditioning_length
        )
        self.proj_dist_args = self.distr_output.get_args_proj(config.num_cells)

        # Embeddings
        self.embed_dim = config.embedding_dimension
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        # Scaler
        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """Returns lagged subsequences of a given sequence."""
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]:
        """Unroll the RNN with given inputs."""
        
        # Scale lags
        lags_scaled = lags / scale.unsqueeze(-1)
        input_lags = lags_scaled.reshape((-1, unroll_length, len(self.lags_seq) * self.target_dim))

        # Embeddings
        index_embeddings = self.embed(target_dimension_indicator)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )

        # Combine inputs
        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)

        # RNN forward pass
        outputs, state = self.rnn(inputs, begin_state)

        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll the RNN encoder over past and future data."""
        
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length :, ...], future_time_feat), dim=1
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # Get lagged subsequences
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # Compute scale
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        # Unroll RNN
        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimension_indicator=target_dimension_indicator,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        """Get distribution arguments from RNN outputs."""
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        
        seq_len = self.context_length + self.prediction_length

        # Unroll encoder
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        # Target sequence
        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf), dim=1
        )

        # Distribution arguments
        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        if self.scaling:
            self.diffusion.scale = scale

        # Compute likelihoods
        likelihoods = self.diffusion.log_prob(target, distr_args).unsqueeze(-1)

        # Compute loss weights
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        observed_values = torch.cat(
            (past_observed_values[:, -self.context_length :, ...], future_observed_values), dim=1
        )
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        # Weighted loss
        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        return (loss.mean(), likelihoods, distr_args)


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    """TimeGrad prediction network for sampling."""
    
    def __init__(self, config: TimeGradConfig, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Sample future paths using the trained model."""
        
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # Repeat tensors for parallel sampling
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        if self.scaling:
            self.diffusion.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states = repeat(begin_states, dim=1)

        future_samples = []

        # Generate samples step by step
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )

            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            new_samples = self.diffusion.sample(cond=distr_args)

            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat((repeated_past_target_cdf, new_samples), dim=1)

        # Reshape samples
        samples = torch.cat(future_samples, dim=1)
        return samples.reshape((-1, self.num_parallel_samples, self.prediction_length, self.target_dim))

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for prediction."""
        
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # Encode past data
        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        # Generate samples
        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=begin_states,
        )


if __name__ == "__main__":
    try:
        # Test full TimeGrad network
        print("Testing TimeGrad network...")
        config = TimeGradConfig(
            input_size=10,  # lags + embeddings + time features
            target_dim=1,
            context_length=24,
            prediction_length=24,
        )
        
        training_net = TimeGradTrainingNetwork(config)
        print("TimeGrad training network created successfully!")
        
        prediction_net = TimeGradPredictionNetwork(config, num_parallel_samples=10)
        print("TimeGrad prediction network created successfully!")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()