"""run_timegrad.py

Script to run TimeGrad on S&P500 data, adapted from notebook.
Trains and tests for MVP validation in your project.
"""
import sys
import os
from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from gluonts.dataset.split import split

# Import from the modules directly
from src.pts.model.time_grad.diffusion import Diffusion, DiffusionConfig
from src.pts.model.time_grad.time_grad_network import TimeGradNetwork, NetworkConfig
from src.pts.model.time_grad.time_grad_estimator import TimeGradEstimator, EstimatorConfig
from src.pts.model.time_grad.time_grad_predictor import TimeGradPredictor
from src.pts.trainer import Trainer
from data_fetch import fetch_sp500_data, prepare_gluonts_dataset, DataConfig

def main():
    """Run TimeGrad training and inference on S&P500 data."""
    try:
        # Configs
        data_config = DataConfig(start_date='2024-01-01', context_length=24, prediction_length=24)
        net_config = NetworkConfig(input_dim=1, hidden_dim=40, num_layers=2)
        diff_config = DiffusionConfig(num_steps=100, beta_start=1e-4, beta_end=0.1)
        est_config = EstimatorConfig(learning_rate=1e-3, batch_size=64)

        print("Fetching S&P500 data...")
        # Fetch data
        df = fetch_sp500_data(data_config)
        print(f"Fetched {len(df)} data points")
        
        # Simple normalization (optional - you can skip this for now)
        data_mean = df['close'].mean()
        data_std = df['close'].std()
        df_normalized = df.copy()
        df_normalized['close'] = (df['close'] - data_mean) / data_std
        print(f"Data normalized - Mean: {data_mean:.2f}, Std: {data_std:.2f}")
        
        # Create dataset
        dataset = prepare_gluonts_dataset(df_normalized, data_config)
        print("Dataset prepared successfully")

        # Split data
        train_dataset, test_dataset = split(dataset, offset=-data_config.prediction_length)
        train_data = list(train_dataset)
        
        print(f"Training data: {len(train_data)} series")

        # Simple dataloader
        class SimpleLoader:
            def __init__(self, ds, context_length, num_layers, hidden_dim):
                self.ds = ds
                self.context_length = context_length
                self.num_layers = num_layers
                self.hidden_dim = hidden_dim
                
            def __iter__(self):
                for item in self.ds:
                    target_data = item['target']
                    
                    if len(target_data) >= self.context_length:
                        context_data = target_data[-self.context_length:]
                        target_tensor = torch.tensor(
                            context_data.reshape(1, -1, 1), 
                            dtype=torch.float32
                        )
                        
                        hidden = (
                            torch.zeros(self.num_layers, 1, self.hidden_dim),
                            torch.zeros(self.num_layers, 1, self.hidden_dim)
                        )
                        
                        yield {
                            'target': target_tensor,
                            'hidden': hidden
                        }
                        
            def __len__(self):
                return len(self.ds)

        train_loader = SimpleLoader(
            train_data, 
            data_config.context_length, 
            net_config.num_layers, 
            net_config.hidden_dim
        )

        print("Initializing models...")
        # Initialize models
        diffusion = Diffusion(diff_config)
        network = TimeGradNetwork(net_config)
        estimator = TimeGradEstimator(est_config, network, diffusion)
        trainer = Trainer(estimator, epochs=10)
        
        print("Starting training...")
        losses = trainer.train(train_loader)
        print(f"Training completed. Final loss: {losses[-1]:.4f}")

        print("Generating synthetic samples...")
        # Generate samples
        predictor = TimeGradPredictor(network, diffusion)
        samples = predictor.predict(
            context_length=data_config.prediction_length, 
            num_samples=100
        )
        print(f"Generated samples shape: {samples.shape}")

        # Simple plotting without normalization issues
        print("Creating visualization...")
        plt.figure(figsize=(12, 6))
        
        # Get original data for comparison
        original_data = df['close'][-data_config.prediction_length:].values
        
        # Denormalize synthetic data
        synthetic_sample = samples[0].detach().numpy()
        synthetic_denorm = synthetic_sample * data_std + data_mean
        
        # Plot comparison
        plt.subplot(1, 2, 1)
        plt.plot(original_data, label='Real S&P500', color='blue', linewidth=2)
        plt.plot(synthetic_denorm, label='Synthetic', color='red', linewidth=2)
        plt.legend()
        plt.title('Real vs Synthetic Time Series')
        plt.ylabel('Price')
        plt.xlabel('Time Steps')
        
        # Plot multiple samples
        plt.subplot(1, 2, 2)
        for i in range(min(3, samples.shape[0])):
            sample_denorm = (samples[i].detach().numpy() * data_std + data_mean)
            plt.plot(sample_denorm, alpha=0.7, label=f'Sample {i+1}')
        
        plt.plot(original_data, color='black', linewidth=3, label='Real Data')
        plt.title('Multiple Synthetic Samples')
        plt.ylabel('Price')
        plt.xlabel('Time Steps')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n=== Results Summary ===")
        print(f"Real data - Mean: {np.mean(original_data):.2f}, Std: {np.std(original_data):.2f}")
        print(f"Synthetic data - Mean: {np.mean(synthetic_denorm):.2f}, Std: {np.std(synthetic_denorm):.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()