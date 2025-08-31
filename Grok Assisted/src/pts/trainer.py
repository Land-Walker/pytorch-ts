"""trainer.py

Trainer for TimeGrad model.
Manages epochs and logging for efficient training in your MVP.
"""

from typing import List

import torch
from tqdm import tqdm

class Trainer:
    """Trainer for TimeGrad model."""
    def __init__(self, estimator: TimeGradEstimator, epochs: int = 10):
        self.estimator = estimator
        self.epochs = epochs

    def train(self, dataloader: torch.utils.data.DataLoader) -> List[float]:
        """Train the model over epochs."""
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(dataloader):
                loss = self.estimator.train_step(batch)
                epoch_loss += loss
            losses.append(epoch_loss / len(dataloader))
            print(f"Epoch {epoch} loss: {losses[-1]}")
        return losses

if __name__ == "__main__":
    class DummyLoader:
        def __iter__(self):
            yield {'target': torch.randn(64, 10, 1), 'hidden': (torch.randn(2, 64, 40), torch.randn(2, 64, 40))}
        def __len__(self):
            return 1

    estimator = TimeGradEstimator(EstimatorConfig(), TimeGradNetwork(NetworkConfig(input_dim=1)), Diffusion(DiffusionConfig()))
    trainer = Trainer(estimator, epochs=2)
    losses = trainer.train(DummyLoader())
    print("Losses:", losses)