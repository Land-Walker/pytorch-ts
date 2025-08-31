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
import pandas as pd

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
        # Use daily data ('1d') to fetch the full history from 2010.
        # yfinance limits hourly ('1h') data to the last ~2 years.
        data_config = DataConfig(
            start_date='2010-01-01', 
            interval='1d',
            context_length=24, 
            prediction_length=24
        )
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
        # We only need the training part from the split for this script.
        train_dataset, _ = split(dataset, offset=-data_config.prediction_length)
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

                    # Create sliding windows over the entire training series
                    # to ensure the model sees all the data.
                    for i in range(len(target_data) - self.context_length + 1):
                        context_data = target_data[i : i + self.context_length]
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
                # The length is the total number of windows we can create.
                total_windows = 0
                for item in self.ds:
                    total_windows += max(0, len(item['target']) - self.context_length + 1)
                return total_windows

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

        # --- Visualization ---
        print("Creating visualization...")

        # 1. Define the forecast period
        forecast_start_date = df.index[-data_config.prediction_length]
        forecast_index = pd.date_range(
            start=forecast_start_date, 
            periods=data_config.prediction_length, 
            freq=df.index.freq
        )

        # 2. Get historical data to plot (from 2020 onwards, as requested)
        historical_data_to_plot = df['close']['2020':]

        # 3. Get the actual data for the forecast period for comparison
        ground_truth_forecast = df['close'][forecast_start_date:]

        # 4. Create the plot
        # Adjust width based on the number of historical points to display
        dynamic_width = max(15, len(historical_data_to_plot) / 50) 
        plt.figure(figsize=(dynamic_width, 7))

        # Plot historical data
        plt.plot(
            historical_data_to_plot.index, 
            historical_data_to_plot.values, 
            label='Historical Observations (from 2020)', 
            color='blue'
        )

        # Plot multiple synthetic samples for the forecast period
        for i in range(min(5, samples.shape[0])):
            synthetic_sample = samples[i].detach().numpy()
            synthetic_denorm = synthetic_sample * data_std + data_mean
            plt.plot(
                forecast_index, 
                synthetic_denorm, 
                color='red', 
                alpha=0.3, 
                label='_nolegend_' if i > 0 else 'Synthetic Forecasts'
            )

        # Plot the ground truth for the forecast period
        plt.plot(
            ground_truth_forecast.index, 
            ground_truth_forecast.values, 
            label='Ground Truth (Forecast Period)', 
            color='black', 
            linewidth=2
        )

        plt.title('S&P 500 Price: Historical Data and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print statistics for the forecast period
        original_data = df['close'][-data_config.prediction_length:].values
        # Use the first sample for statistics comparison
        synthetic_denorm_for_stats = (samples[0].detach().numpy() * data_std + data_mean)
        print("\n=== Results Summary ===")
        print(f"Real data (forecast period) - Mean: {np.mean(original_data):.2f}, Std: {np.std(original_data):.2f}")
        print(f"Synthetic data (forecast period) - Mean: {np.mean(synthetic_denorm_for_stats):.2f}, Std: {np.std(synthetic_denorm_for_stats):.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()