"""trainer.py

Advanced trainer for TimeGrad model with sophisticated training loop.
Manages epochs, logging, and optimization for the advanced TimeGrad implementation.
"""

from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm

# Import advanced TimeGrad components
from .model.time_grad.time_grad_estimator import TimeGradEstimator, TimeGradEstimatorConfig
from .model.time_grad.time_grad_network import TimeGradTrainingNetwork, TimeGradConfig
from .model.time_grad.diffusion import GaussianDiffusion


class AdvancedTrainer:
    """Advanced trainer for TimeGrad model with comprehensive training features."""
    
    def __init__(
        self, 
        network: TimeGradTrainingNetwork,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        device: torch.device = None,
        gradient_clip_norm: float = 1.0,
        scheduler_patience: int = 5,
        early_stopping_patience: int = 10
    ):
        self.network = network
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_clip_norm = gradient_clip_norm
        
        # Move network to device
        self.network = self.network.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=scheduler_patience,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Training history
        self.train_losses = []
        self.learning_rates = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with proper error handling."""
        self.network.train()
        
        try:
            # Move batch to device
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            loss, likelihoods, distr_args = self.network(
                target_dimension_indicator=batch_device['target_dimension_indicator'],
                past_time_feat=batch_device['past_time_feat'],
                past_target_cdf=batch_device['past_target_cdf'],
                past_observed_values=batch_device['past_observed_values'],
                past_is_pad=batch_device['past_is_pad'],
                future_time_feat=batch_device['future_time_feat'],
                future_target_cdf=batch_device['future_target_cdf'],
                future_observed_values=batch_device['future_observed_values'],
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Training step failed: {e}")
            return float('inf')

    def train(self, dataloader, validation_dataloader: Optional = None) -> List[float]:
        """Train the model with comprehensive monitoring."""
        print(f"Starting training on {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            epoch_loss = 0
            batch_count = 0
            
            progress_bar = tqdm(
                dataloader, 
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False
            )
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                
                if loss != float('inf'):
                    epoch_loss += loss
                    batch_count += 1
                    progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Calculate average loss
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
            else:
                avg_loss = float('inf')
                print("Warning: No successful batches in this epoch")
            
            self.train_losses.append(avg_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Validation phase (if available)
            val_loss = None
            if validation_dataloader is not None:
                val_loss = self.validate(validation_dataloader)
                print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                scheduler_loss = val_loss
            else:
                print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}")
                scheduler_loss = avg_loss
            
            # Learning rate scheduling
            self.scheduler.step(scheduler_loss)
            
            # Early stopping and best model tracking
            if scheduler_loss < self.best_loss:
                self.best_loss = scheduler_loss
                self.patience_counter = 0
                best_model_state = self.network.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            self.network.load_state_dict(best_model_state)
            print(f"Loaded best model with loss: {self.best_loss:.4f}")
        
        return self.train_losses

    def validate(self, validation_dataloader) -> float:
        """Validation phase."""
        self.network.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in validation_dataloader:
                try:
                    # Move batch to device
                    batch_device = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    loss, _, _ = self.network(
                        target_dimension_indicator=batch_device['target_dimension_indicator'],
                        past_time_feat=batch_device['past_time_feat'],
                        past_target_cdf=batch_device['past_target_cdf'],
                        past_observed_values=batch_device['past_observed_values'],
                        past_is_pad=batch_device['past_is_pad'],
                        future_time_feat=batch_device['future_time_feat'],
                        future_target_cdf=batch_device['future_target_cdf'],
                        future_observed_values=batch_device['future_observed_values'],
                    )
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Validation batch failed: {e}")
                    continue
        
        return total_loss / max(batch_count, 1)

    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Dict[str, Any] = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
            'best_loss': self.best_loss,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint


# Backward compatibility trainer (simplified version)
class Trainer:
    """Simplified trainer for backward compatibility."""
    
    def __init__(self, network: TimeGradTrainingNetwork, epochs: int = 10):
        self.advanced_trainer = AdvancedTrainer(network, epochs=epochs)

    def train(self, dataloader) -> List[float]:
        """Simple training interface."""
        return self.advanced_trainer.train(dataloader)


# Factory function for creating trainers
def create_trainer(
    estimator_or_network,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    use_advanced_features: bool = True,
    **kwargs
) -> AdvancedTrainer:
    """Create appropriate trainer based on input."""
    
    # Extract network from estimator if needed
    if hasattr(estimator_or_network, 'create_training_network'):
        # It's an estimator
        device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        network = estimator_or_network.create_training_network(device)
    else:
        # It's already a network
        network = estimator_or_network
    
    if use_advanced_features:
        return AdvancedTrainer(
            network=network,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs
        )
    else:
        return Trainer(network, epochs)


if __name__ == "__main__":
    try:
        # Test with advanced TimeGrad
        from .model.time_grad.time_grad_estimator import TimeGradEstimatorConfig
        
        config = TimeGradEstimatorConfig(
            input_size=10,
            freq='D',
            prediction_length=10,
            target_dim=1,
            epochs=2,
            batch_size=4
        )
        
        estimator = TimeGradEstimator(config)
        device = torch.device('cpu')
        
        # Create trainer from estimator
        trainer = create_trainer(estimator, epochs=2, device=device)
        print("Advanced trainer created successfully!")
        
        # Test with dummy data
        class DummyDataLoader:
            def __iter__(self):
                for _ in range(2):  # 2 batches
                    yield {
                        'target_dimension_indicator': torch.arange(1).unsqueeze(0).repeat(2, 1),
                        'past_time_feat': torch.zeros(2, 40, 1),
                        'past_target_cdf': torch.randn(2, 40, 1),
                        'past_observed_values': torch.ones(2, 40, 1),
                        'past_is_pad': torch.zeros(2, 40),
                        'future_time_feat': torch.zeros(2, 10, 1),
                        'future_target_cdf': torch.randn(2, 10, 1),
                        'future_observed_values': torch.ones(2, 10, 1),
                    }
            
            def __len__(self):
                return 2
        
        dummy_loader = DummyDataLoader()
        losses = trainer.train(dummy_loader)
        print(f"Training completed. Losses: {losses}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()