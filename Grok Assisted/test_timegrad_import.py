"""test_timegrad_import.py

Test module to verify and demonstrate importing TimeGradTrainingNetwork.
Handles potential dependency issues and provides example usage for financial time series.
"""

from typing import Dict, Any
import logging
import sys
import torch
from gluonts.dataset.common import ListDataset
import numpy as np
import pandas as pd
from pydantic import ValidationError
from pydantic import BaseModel

# Attempt corrected import (remove 'src.' if using installed package)
try:
    from pts.model.time_grad.time_grad_network import TimeGradTrainingNetwork, TimeGradConfig
except ImportError as e:
    # Fallback: Add path manually if not installed properly
    sys.path.append('/workspaces/pytorch-ts/Grok Assisted/src')  # Adjust to your workspace root
    from pts.model.time_grad.time_grad_network import TimeGradTrainingNetwork, TimeGradConfig
    logging.warning("Manual path added to sys.path. Consider reinstalling with 'pip install -e .'")

from pts import trainer  # Assuming installed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TimeGradTestConfig(BaseModel):
    """Simplified config for testing TimeGrad import and setup."""
    input_size: int = 10  # e.g., features including log returns, volume
    target_dim: int = 1  # e.g., BTC price log returns
    context_length: int = 24  # 1 day of hourly data
    prediction_length: int = 12  # Forecast 12 hours ahead

def create_sample_dataset() -> ListDataset:
    """Create a sample GluonTS dataset from synthetic financial data.
    
    Returns:
        ListDataset: Dataset for TimeGrad training/testing.
    
    Raises:
        ValueError: If data generation fails.
    """
    try:
        # Synthetic BTC/USD hourly log returns (replace with Upbit data in production)
        dates = pd.date_range(start="2025-01-01", periods=100, freq="H")
        log_returns = np.random.normal(0, 0.01, size=100)  # Mean 0, std 1% volatility
        df = pd.DataFrame({"log_return": log_returns}, index=dates)
        
        dataset = ListDataset(
            [{"start": df.index[0], "target": df["log_return"].values}],
            freq="1H"
        )
        logger.info(f"Created sample dataset with {len(df)} points.")
        return dataset
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise ValueError(f"Invalid data: {e}")

def test_timegrad_network(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Test instantiation and basic forward pass of TimeGradTrainingNetwork.
    
    Args:
        config_dict: Dictionary for TimeGradConfig.
    
    Returns:
        Dict with test results (e.g., {'status': 'success', 'output_shape': ...}).
    
    Raises:
        ValidationError: If config is invalid.
        RuntimeError: If network forward pass fails.
    """
    try:
        config = TimeGradConfig(**config_dict)
        network = TimeGradTrainingNetwork(config)
        logger.info("TimeGradTrainingNetwork instantiated successfully.")
        
        # Sample inputs (batch_size=2, seq_len=context+pred, feat_dim=...)
        batch_size = 2
        target_dim = config.target_dim
        past_time_feat = torch.randn(batch_size, config.history_length, config.input_size)
        past_target_cdf = torch.randn(batch_size, config.history_length, target_dim)
        past_observed_values = torch.ones_like(past_target_cdf)
        past_is_pad = torch.zeros(batch_size, config.history_length)
        future_time_feat = torch.randn(batch_size, config.prediction_length, config.input_size)
        future_target_cdf = torch.randn(batch_size, config.prediction_length, target_dim)
        future_observed_values = torch.ones_like(future_target_cdf)
        target_dimension_indicator = torch.tensor([[0] for _ in range(batch_size)])  # Dummy
        
        # Forward pass
        loss, likelihoods, distr_args = network(
            target_dimension_indicator,
            past_time_feat,
            past_target_cdf,
            past_observed_values,
            past_is_pad,
            future_time_feat,
            future_target_cdf,
            future_observed_values,
        )
        
        logger.info(f"Forward pass successful. Loss shape: {loss.shape}")
        return {
            "status": "success",
            "loss_shape": str(loss.shape),
            "likelihoods_shape": str(likelihoods.shape),
        }
    except ValidationError as ve:
        logger.error(f"Config validation failed: {ve}")
        raise
    except RuntimeError as re:
        logger.error(f"Network runtime error: {re}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise RuntimeError(f"Test failed: {e}")

if __name__ == "__main__":
    try:
        # Install missing deps if needed (run manually if script fails)
        # pip install pydantic gluonts torch
        
        config_dict = TimeGradTestConfig().dict()
        result = test_timegrad_network(config_dict)
        logger.info(f"Test results: {result}")
        
        # Example integration with dataset
        dataset = create_sample_dataset()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(epochs=1, device=device)  # Minimal for test
        # Note: For full training, use TimeGradEstimator from the repo
        logger.info("Ready for full TimeGrad training/integration.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")