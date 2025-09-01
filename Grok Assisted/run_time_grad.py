"""run_time_grad.py

Updated script to run TimeGrad on S&P500 data with advanced implementation.
Uses the sophisticated TimeGrad networks, estimators, and predictors.
"""
import sys
import os
from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import from the updated modules
from src.pts.model.time_grad.time_grad_network import TimeGradTrainingNetwork, TimeGradConfig
from src.pts.model.time_grad.time_grad_estimator import TimeGradEstimator, TimeGradEstimatorConfig
from src.pts.model.time_grad.time_grad_predictor import TimeGradPredictor, TimeGradPredictorFactory
from src.pts.model.time_grad.diffusion import GaussianDiffusion
from data_fetch import fetch_sp500_data, prepare_gluonts_dataset, DataConfig, normalize_data, denormalize_data


class SimpleTrainer:
    """Simple trainer for TimeGrad without full GluonTS dependency."""
    
    def __init__(self, epochs: int = 10, learning_rate: float = 1e-3):
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def train(self, network: TimeGradTrainingNetwork, dataloader, device: torch.device):
        """Train the TimeGrad network."""
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        
        losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch in dataloader:
                try:
                    # Extract batch data
                    target_dim_indicator = batch['target_dimension_indicator']
                    past_time_feat = batch['past_time_feat']
                    past_target_cdf = batch['past_target_cdf']
                    past_observed_values = batch['past_observed_values']
                    past_is_pad = batch['past_is_pad']
                    future_time_feat = batch['future_time_feat']
                    future_target_cdf = batch['future_target_cdf']
                    future_observed_values = batch['future_observed_values']
                    
                    # Move to device
                    target_dim_indicator = target_dim_indicator.to(device)
                    past_time_feat = past_time_feat.to(device)
                    past_target_cdf = past_target_cdf.to(device)
                    past_observed_values = past_observed_values.to(device)
                    past_is_pad = past_is_pad.to(device)
                    future_time_feat = future_time_feat.to(device)
                    future_target_cdf = future_target_cdf.to(device)
                    future_observed_values = future_observed_values.to(device)
                    
                    # Forward pass
                    loss, likelihoods, distr_args = network(
                        target_dimension_indicator=target_dim_indicator,
                        past_time_feat=past_time_feat,
                        past_target_cdf=past_target_cdf,
                        past_observed_values=past_observed_values,
                        past_is_pad=past_is_pad,
                        future_time_feat=future_time_feat,
                        future_target_cdf=future_target_cdf,
                        future_observed_values=future_observed_values,
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Batch failed: {e}")
                    continue
            
            avg_loss = epoch_loss / max(batch_count, 1)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return losses


class TimeGradDataLoader:
    """Simple dataloader for TimeGrad training."""
    
    def __init__(self, data, config: TimeGradEstimatorConfig, batch_size: int = 8):
        self.data = data
        self.config = config
        self.batch_size = batch_size
        self.target_dim = config.target_dim
        
    def __iter__(self):
        """Generate batches for training."""
        for item in self.data:
            target_data = item['target']
            
            # Create sliding windows
            total_length = len(target_data)
            context_length = self.config.context_length or self.config.prediction_length
            history_length = context_length + max([1, 2, 3, 4, 5, 6, 7])  # Default lags
            
            if total_length < history_length + self.config.prediction_length:
                continue
            
            # Create multiple samples from this series
            samples = []
            max_start = total_length - history_length - self.config.prediction_length
            
            for start_idx in range(0, max_start, max(1, max_start // 10)):  # Take 10 samples max per series
                # Extract past data
                past_end = start_idx + history_length
                past_target = target_data[start_idx:past_end]
                
                # Extract future data
                future_start = past_end
                future_end = future_start + self.config.prediction_length
                future_target = target_data[future_start:future_end]
                
                # Create batch item
                batch_item = {
                    'target_dimension_indicator': torch.arange(self.target_dim).unsqueeze(0),
                    'past_time_feat': torch.zeros(1, history_length, 1),  # Dummy time features
                    'past_target_cdf': torch.tensor(past_target.reshape(1, -1, 1), dtype=torch.float32),
                    'past_observed_values': torch.ones(1, history_length, 1),
                    'past_is_pad': torch.zeros(1, history_length),
                    'future_time_feat': torch.zeros(1, self.config.prediction_length, 1),
                    'future_target_cdf': torch.tensor(future_target.reshape(1, -1, 1), dtype=torch.float32),
                    'future_observed_values': torch.ones(1, self.config.prediction_length, 1),
                }
                
                samples.append(batch_item)
                
                if len(samples) >= self.batch_size:
                    yield self._collate_batch(samples)
                    samples = []
            
            # Yield remaining samples
            if samples:
                yield self._collate_batch(samples)
    
    def _collate_batch(self, samples):
        """Collate samples into a batch."""
        batch = {}
        for key in samples[0].keys():
            batch[key] = torch.cat([sample[key] for sample in samples], dim=0)
        return batch
    
    def __len__(self):
        return 1  # Simplified length calculation


def main():
    """Run TimeGrad training and inference on S&P500 data."""
    try:
        print("=== TimeGrad S&P500 Experiment ===")
        
        # Configuration
        data_config = DataConfig(
            start_date='2020-01-01', 
            interval='1d',
            context_length=30, 
            prediction_length=10
        )
        
        estimator_config = TimeGradEstimatorConfig(
            input_size=8,  # lags + embeddings + time features
            freq='D',
            prediction_length=data_config.prediction_length,
            target_dim=1,
            context_length=data_config.context_length,
            num_layers=2,
            num_cells=64,
            conditioning_length=64,
            diff_steps=50,  # Reduced for faster training
            residual_layers=4,  # Reduced for faster training
            residual_channels=16,  # Reduced for faster training
            batch_size=8,
            epochs=5,  # Reduced for faster training
            learning_rate=1e-3,
        )

        print("Fetching S&P500 data...")
        # Fetch and prepare data
        df = fetch_sp500_data(data_config)
        print(f"Fetched {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
        # Normalize data
        df_normalized, data_mean, data_std = normalize_data(df)
        print(f"Data normalized - Mean: {data_mean:.2f}, Std: {data_std:.2f}")
        
        # Create dataset
        dataset = prepare_gluonts_dataset(df_normalized, data_config)
        train_data = list(dataset)
        print(f"Dataset created with {len(train_data)} series")

        # Create dataloader
        dataloader = TimeGradDataLoader(train_data, estimator_config)
        print("DataLoader created")

        # Initialize model components
        print("Initializing TimeGrad components...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create estimator and training network
        estimator = TimeGradEstimator(estimator_config)
        training_network = estimator.create_training_network(device)
        print("Training network created")

        # Create trainer
        trainer = SimpleTrainer(epochs=estimator_config.epochs, learning_rate=estimator_config.learning_rate)
        
        print("Starting training...")
        losses = trainer.train(training_network, dataloader, device)
        print(f"Training completed. Final loss: {losses[-1]:.4f}")
        
        # Plot training losses
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        print("Creating predictor...")
        # Create predictor
        from src.pts.model.time_grad.time_grad_network import TimeGradPredictionNetwork
        
        prediction_network = TimeGradPredictionNetwork(
            config=TimeGradConfig(
                input_size=estimator_config.input_size,
                num_layers=estimator_config.num_layers,
                num_cells=estimator_config.num_cells,
                cell_type=estimator_config.cell_type,
                history_length=estimator.history_length,
                context_length=estimator.context_length,
                prediction_length=estimator_config.prediction_length,
                dropout_rate=estimator_config.dropout_rate,
                lags_seq=estimator.lags_seq,
                target_dim=estimator_config.target_dim,
                conditioning_length=estimator_config.conditioning_length,
                diff_steps=estimator_config.diff_steps,
                loss_type=estimator_config.loss_type,
                beta_end=estimator_config.beta_end,
                beta_schedule=estimator_config.beta_schedule,
                residual_layers=estimator_config.residual_layers,
                residual_channels=estimator_config.residual_channels,
                dilation_cycle_length=estimator_config.dilation_cycle_length,
                cardinality=estimator_config.cardinality,
                embedding_dimension=estimator_config.embedding_dimension,
                scaling=estimator_config.scaling,
            ),
            num_parallel_samples=20
        ).to(device)
        
        # Copy weights from training network
        prediction_network.load_state_dict(training_network.state_dict())
        
        predictor = TimeGradPredictor(prediction_network)
        
        print("Generating predictions...")
        # Generate some predictions using simple interface
        num_samples = 10
        samples = predictor.predict_simple(
            context_length=data_config.prediction_length,
            num_samples=num_samples,
            device=str(device)
        )
        
        print(f"Generated {samples.shape[0]} forecast samples of length {samples.shape[1]}")
        
        # Denormalize samples
        samples_denorm = denormalize_data(samples, data_mean, data_std)
        
        # Create visualization
        print("Creating visualization...")
        
        plt.subplot(1, 2, 2)
        
        # Plot recent historical data
        recent_data = df['close'][-50:]
        plt.plot(range(len(recent_data)), recent_data.values, 'b-', label='Historical Data', linewidth=2)
        
        # Plot predictions
        forecast_start = len(recent_data)
        forecast_range = range(forecast_start, forecast_start + data_config.prediction_length)
        
        for i in range(min(5, num_samples)):
            alpha = 0.6 if i == 0 else 0.3
            label = 'Forecasts' if i == 0 else '_nolegend_'
            plt.plot(forecast_range, samples_denorm[i].detach().cpu().numpy(), 
                    'r-', alpha=alpha, label=label)
        
        # Plot ground truth for comparison if available
        if len(df) > len(recent_data):
            ground_truth = df['close'][-len(recent_data):].values
            if len(ground_truth) >= forecast_start + data_config.prediction_length:
                true_forecast = ground_truth[forecast_start:forecast_start + data_config.prediction_length]
                plt.plot(forecast_range, true_forecast, 'g-', linewidth=2, label='Ground Truth')
        
        plt.title('S&P 500: Historical Data and TimeGrad Forecasts')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

        # Print statistics
        print("\n=== Results Summary ===")
        print(f"Historical data range: {df.index[0]} to {df.index[-1]}")
        print(f"Number of data points: {len(df)}")
        print(f"Prediction length: {data_config.prediction_length}")
        print(f"Number of forecast samples: {num_samples}")
        
        recent_prices = df['close'][-10:].values
        forecast_mean = samples_denorm.mean(dim=0).detach().cpu().numpy()
        forecast_std = samples_denorm.std(dim=0).detach().cpu().numpy()
        
        print(f"Recent actual prices (last 10): {recent_prices}")
        print(f"Forecast mean: {forecast_mean}")
        print(f"Forecast std: {forecast_std}")
        
        print("\nTimeGrad experiment completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()