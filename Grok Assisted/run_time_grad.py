"""run_timegrad.py

Script to run TimeGrad on S&P500 data, adapted from notebook.
Trains and tests for MVP validation in your project.
"""

from typing import List
import torch
import matplotlib.pyplot as plt
from gluonts.dataset.split import split
from src.pts.model.time_grad import Diffusion, DiffusionConfig, TimeGradNetwork, NetworkConfig, TimeGradEstimator, EstimatorConfig, TimeGradPredictor
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

        # Fetch and prepare data
        df = fetch_sp500_data(data_config)
        dataset = prepare_gluonts_dataset(df, data_config)

        # Split: Use GluonTS split (train/test)
        train_dataset, test_dataset = split(dataset, offset=-data_config.prediction_length)
        train_data = train_dataset.dataset  # ListDataset

        # Dummy dataloader (adapt for full; here uses simple iterable for demo)
        class SimpleLoader:
            def __init__(self, ds):
                self.ds = ds
            def __iter__(self):
                for item in self.ds:
                    yield {'target': torch.tensor(item['target'][-data_config.context_length:].reshape(1, -1, 1).astype(float)),
                           'hidden': (torch.zeros(net_config.num_layers, 1, net_config.hidden_dim),
                                      torch.zeros(net_config.num_layers, 1, net_config.hidden_dim))}
            def __len__(self):
                return len(self.ds)

        train_loader = SimpleLoader(train_data)

        # Initialize and train
        diffusion = Diffusion(diff_config)
        network = TimeGradNetwork(net_config)
        estimator = TimeGradEstimator(est_config, network, diffusion)
        trainer = Trainer(estimator, epochs=10)
        losses = trainer.train(train_loader)
        print(f"Training losses: {losses}")

        # Test: Generate samples (use dummy hidden for simplicity)
        predictor = TimeGradPredictor(network, diffusion)
        h = (torch.zeros(net_config.num_layers, 1, net_config.hidden_dim),
             torch.zeros(net_config.num_layers, 1, net_config.hidden_dim))
        samples = predictor.predict(h, num_samples=100)
        print(f"Generated samples shape: {samples.shape}")

        # Plot example: Real vs Synthetic (first sample)
        real = df['close'][-data_config.prediction_length:]
        synth = samples[0].detach().numpy()[:len(real)]
        plt.plot(real.values, label='Real S&P500 Close')
        plt.plot(synth, label='Synthetic')
        plt.legend()
        plt.title('Real vs Synthetic Time Series')
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()