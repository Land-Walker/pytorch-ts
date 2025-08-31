"""data_fetch.py

Module to fetch and preprocess S&P500 (^GSPC) time-series data for TimeGrad.
Handles yfinance fetching, missing data, normalization; fits data pipeline for synthetic generation.
"""

from typing import Tuple, Optional
import datetime
import pandas as pd
import torch
import yfinance as yf
from pydantic import BaseModel, ValidationError, field_validator
from gluonts.dataset.common import ListDataset

class DataConfig(BaseModel):
    """Configuration for data fetching and preprocessing."""
    symbol: str = '^GSPC'
    start_date: str
    end_date: Optional[str] = None
    interval: str = '1h'
    context_length: int = 24
    prediction_length: int = 24

    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in ['1h', '1d']:
            raise ValueError("Interval must be '1h' or '1d'.")
        return v

def fetch_sp500_data(config: DataConfig) -> pd.DataFrame:
    """Fetch S&P500 data from yfinance.

    Args:
        config: DataConfig with symbol, dates, and interval.

    Returns:
        pd.DataFrame: OHLCV data with datetime index.

    Raises:
        ValueError: If fetch fails or data is empty.
    """
    try:
        # Fix the yfinance warning by explicitly setting auto_adjust
        df = yf.download(
            config.symbol, 
            start=config.start_date, 
            end=config.end_date or datetime.date.today().isoformat(),
            interval=config.interval,
            auto_adjust=False  # Explicitly set to avoid warning
        )
        
        if df.empty:
            raise ValueError("Fetched data is empty.")
        
        # Handle missing data
        df = df.ffill().bfill()
        
        # Return only the Close column and ensure it's a Series (1D)
        result = df[['Close']].copy()
        result.columns = ['close']  # Rename for consistency
        
        return result
        
    except Exception as e:
        raise ValueError(f"Data fetch failed: {str(e)}")

def prepare_gluonts_dataset(df: pd.DataFrame, config: DataConfig) -> ListDataset:
    """Convert DataFrame to GluonTS dataset for TimeGrad.

    Args:
        df: DataFrame with time-series data.
        config: DataConfig with context and prediction lengths.

    Returns:
        ListDataset: GluonTS-compatible dataset.
    """
    try:
        # Extract the close prices as a 1D numpy array
        target_values = df['close'].values  # This is 1D
        
        # Ensure we have enough data
        min_length = config.context_length + config.prediction_length
        if len(target_values) < min_length:
            raise ValueError(f"Not enough data points. Need at least {min_length}, got {len(target_values)}")
        
        # Create the dataset entry
        data = [{
            'target': target_values,  # 1D array as expected by GluonTS
            'start': df.index[0]
        }]
        
        # Use lowercase 'h' instead of 'H' to fix the deprecation warning
        freq = 'h' if config.interval == '1h' else 'D'
        
        return ListDataset(
            data,
            freq=freq,
            one_dim_target=True
        )
        
    except Exception as e:
        raise ValueError(f"Dataset preparation failed: {str(e)}")

def normalize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Normalize the data to help with training stability.
    
    Args:
        df: DataFrame with close prices
        
    Returns:
        Tuple of (normalized_df, mean, std) for denormalization later
    """
    mean = df['close'].mean()
    std = df['close'].std()
    
    normalized_df = df.copy()
    normalized_df['close'] = (df['close'] - mean) / std
    
    return normalized_df, mean, std

def denormalize_data(normalized_values: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Denormalize the generated samples back to original scale."""
    return normalized_values * std + mean

if __name__ == "__main__":
    try:
        config = DataConfig(start_date='2024-01-01')
        df = fetch_sp500_data(config)
        print(f"Fetched data shape: {df.shape}")
        print(f"Data head:\n{df.head()}")
        
        dataset = prepare_gluonts_dataset(df, config)
        print(f"Dataset created successfully with {len(dataset)} series")
        
        # Test normalization
        norm_df, mean, std = normalize_data(df)
        print(f"Normalization - Mean: {mean:.2f}, Std: {std:.2f}")
        
    except ValidationError as ve:
        print(f"Config error: {ve}")
    except ValueError as ve:
        print(f"Processing error: {ve}")