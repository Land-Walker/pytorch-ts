"""time_grad_predictor.py

Complete TimeGrad predictor for generating samples from trained TimeGrad.
Full implementation with sophisticated forecasting capabilities.
"""

from typing import Tuple, Optional, Dict, Any
import torch
from pydantic import BaseModel

from .time_grad_network import TimeGradPredictionNetwork, TimeGradConfig
from .time_grad_estimator import TimeGradEstimator, TimeGradEstimatorConfig


class PredictorConfig(BaseModel):
    """Configuration for TimeGrad predictor."""
    context_length: int = 24
    num_samples: int = 100
    device: str = 'cpu'


class TimeGradPredictor:
    """
    Advanced TimeGrad predictor with sophisticated forecasting capabilities.
    Supports both simple generation and full time series forecasting.
    """
    
    def __init__(
        self, 
        prediction_network: TimeGradPredictionNetwork,
        diffusion=None
    ):
        self.prediction_network = prediction_network
        self.diffusion = diffusion or getattr(prediction_network, 'diffusion', None)

    def predict_from_data(
        self,
        past_target: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_values: Optional[torch.Tensor] = None,
        target_dimension_indicator: Optional[torch.Tensor] = None,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Generate predictions from historical data using full TimeGrad pipeline.
        
        Args:
            past_target: Historical target values (batch_size, history_length, target_dim)
            past_time_features: Historical time features (batch_size, history_length, num_time_features)
            future_time_features: Future time features (batch_size, prediction_length, num_time_features)
            past_observed_values: Mask for observed values (batch_size, history_length, target_dim)
            target_dimension_indicator: Target dimension indices (batch_size, target_dim)
            num_samples: Number of forecast samples to generate
            
        Returns:
            Generated forecasts (batch_size, num_samples, prediction_length, target_dim)
        """
        try:
            self.prediction_network.eval()
            
            with torch.no_grad():
                batch_size, history_length, target_dim = past_target.shape
                
                # Create default masks and indicators if not provided
                if past_observed_values is None:
                    past_observed_values = torch.ones_like(past_target)
                
                if target_dimension_indicator is None:
                    target_dimension_indicator = torch.arange(target_dim).expand(batch_size, -1)
                
                # Create padding indicator (assume no padding for simplicity)
                past_is_pad = torch.zeros(batch_size, history_length)
                
                # Update network's num_parallel_samples for this prediction
                original_samples = self.prediction_network.num_parallel_samples
                self.prediction_network.num_parallel_samples = num_samples
                
                try:
                    # Generate predictions using the full network
                    predictions = self.prediction_network(
                        target_dimension_indicator=target_dimension_indicator,
                        past_time_feat=past_time_features,
                        past_target_cdf=past_target,
                        past_observed_values=past_observed_values,
                        past_is_pad=past_is_pad,
                        future_time_feat=future_time_features,
                    )
                    
                    return predictions
                    
                finally:
                    # Restore original sample count
                    self.prediction_network.num_parallel_samples = original_samples
                    
        except Exception as e:
            print(f"Advanced prediction debug info:")
            print(f"  past_target shape: {past_target.shape}")
            print(f"  past_time_features shape: {past_time_features.shape}")
            print(f"  future_time_features shape: {future_time_features.shape}")
            if past_observed_values is not None:
                print(f"  past_observed_values shape: {past_observed_values.shape}")
            raise ValueError(f"Advanced predict failed: {str(e)}")

    def predict_simple(
        self, 
        context_length: int, 
        num_samples: int = 100, 
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Simple prediction interface for compatibility.
        Generates samples from noise without historical context.
        
        Args:
            context_length: Length of sequences to generate
            num_samples: Number of samples to generate
            device: Device to run on
            
        Returns:
            Generated samples (num_samples, context_length, target_dim)
        """
        try:
            self.prediction_network.eval()
            
            with torch.no_grad():
                # Create dummy historical data
                batch_size = 1
                history_length = self.prediction_network.config.history_length
                target_dim = self.prediction_network.config.target_dim
                
                # Initialize with random historical data
                past_target = torch.randn(batch_size, history_length, target_dim, device=device)
                past_observed_values = torch.ones_like(past_target)
                past_is_pad = torch.zeros(batch_size, history_length, device=device)
                
                # Create dummy time features (all zeros for simplicity)
                num_time_features = 1  # Minimal time features
                future_time_features = torch.zeros(
                    batch_size, context_length, num_time_features, device=device
                )
                past_time_features = torch.zeros(
                    batch_size, history_length, num_time_features, device=device
                )
                
                # Target dimension indicator
                target_dimension_indicator = torch.arange(target_dim, device=device).expand(batch_size, -1)
                
                # Update sample count
                original_samples = self.prediction_network.num_parallel_samples
                self.prediction_network.num_parallel_samples = num_samples
                
                try:
                    # Generate predictions
                    predictions = self.prediction_network(
                        target_dimension_indicator=target_dimension_indicator,
                        past_time_feat=past_time_features,
                        past_target_cdf=past_target,
                        past_observed_values=past_observed_values,
                        past_is_pad=past_is_pad,
                        future_time_feat=future_time_features,
                    )
                    
                    # Return reshaped predictions: (num_samples, context_length, target_dim)
                    return predictions.squeeze(0)  # Remove batch dimension
                    
                finally:
                    self.prediction_network.num_parallel_samples = original_samples
                    
        except Exception as e:
            print(f"Simple prediction debug info:")
            print(f"  context_length: {context_length}")
            print(f"  num_samples: {num_samples}")
            raise ValueError(f"Simple predict failed: {str(e)}")


class TimeGradPredictorFactory:
    """Factory class for creating TimeGrad predictors."""
    
    @staticmethod
    def create_predictor(
        network: TimeGradPredictionNetwork,
        diffusion=None,
        config: Optional[PredictorConfig] = None
    ) -> TimeGradPredictor:
        """Create TimeGrad predictor from network."""
        return TimeGradPredictor(network, diffusion)

    @staticmethod
    def create_from_estimator(
        estimator: TimeGradEstimator,
        trained_network: TimeGradTrainingNetwork,
        device: torch.device = torch.device('cpu')
    ) -> TimeGradPredictor:
        """Create predictor from trained estimator."""
        
        # Create prediction network with same config as training
        network_config = TimeGradConfig(
            input_size=estimator.config.input_size,
            num_layers=estimator.config.num_layers,
            num_cells=estimator.config.num_cells,
            cell_type=estimator.config.cell_type,
            history_length=estimator.history_length,
            context_length=estimator.context_length,
            prediction_length=estimator.config.prediction_length,
            dropout_rate=estimator.config.dropout_rate,
            lags_seq=estimator.lags_seq,
            target_dim=estimator.config.target_dim,
            conditioning_length=estimator.config.conditioning_length,
            diff_steps=estimator.config.diff_steps,
            loss_type=estimator.config.loss_type,
            beta_end=estimator.config.beta_end,
            beta_schedule=estimator.config.beta_schedule,
            residual_layers=estimator.config.residual_layers,
            residual_channels=estimator.config.residual_channels,
            dilation_cycle_length=estimator.config.dilation_cycle_length,
            cardinality=estimator.config.cardinality,
            embedding_dimension=estimator.config.embedding_dimension,
            scaling=estimator.config.scaling,
        )

        prediction_network = TimeGradPredictionNetwork(
            config=network_config,
            num_parallel_samples=estimator.config.num_parallel_samples,
        ).to(device)
        
        # Copy trained weights
        prediction_network.load_state_dict(trained_network.state_dict())
        
        return TimeGradPredictor(prediction_network)


if __name__ == "__main__":
    try:
        print("Testing TimeGrad predictor...")
        
        # Create network configuration
        config = TimeGradConfig(
            input_size=10,
            target_dim=1,
            context_length=24,
            prediction_length=12,
            history_length=48
        )
        
        prediction_network = TimeGradPredictionNetwork(
            config=config,
            num_parallel_samples=5
        )
        
        predictor = TimeGradPredictor(prediction_network)
        
        # Test simple interface
        samples = predictor.predict_simple(
            context_length=12, num_samples=3
        )
        print(f"Predictor samples shape: {samples.shape}")
        
        # Test factory creation
        factory_predictor = TimeGradPredictorFactory.create_predictor(prediction_network)
        factory_samples = factory_predictor.predict_simple(context_length=12, num_samples=3)
        print(f"Factory predictor samples shape: {factory_samples.shape}")
        
        print("All predictor tests passed!")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()