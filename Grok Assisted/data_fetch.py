"""data_fetch.py

Module to fetch and preprocess S&P500 (^GSPC) time-series data for TimeGrad.
Handles yfinance fetching, missing data, normalization; fits data pipeline for synthetic generation.
Updated to work with advanced TimeGrad implementation.
"""

from typing import Tuple, Optional
import datetime
import pandas as pd
import torch
import yfinance as yf
from pydantic import BaseModel, ValidationError, field_validator

# Try to import GluonTS, provide fallback if not available
try:
    from gluonts.dataset.common import ListDataset
    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False
    print("Warning: GluonTS not available. Using fallback dataset implementation.")
    
    class ListDataset:
        """Fallback implementation when GluonTS is not available."""
        def __init__(self, data, freq, one_dim_target=True):
            self.data = data
            self.freq = freq
            self.one_dim_target = one_dim_target
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)


class DataConfig(BaseModel):
    """Configuration for data fetching and preprocessing."""
    symbol: str = ''
    start_date: str
    end_date: Optional[str] = None
    interval: str = '1h'
    context_length: int = 24
    prediction_length: int = 24

    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in ['1h', '1d', '5m', '15m', '30m', '1wk', '1mo']:
            raise ValueError("Interval must be valid yfinance interval.")
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
        print(f"Fetching {config.symbol} data from {config.start_date} to {config.end_date or 'present'}")
        print(f"Interval: {config.interval}")
        
        # Download data with explicit parameters
        df = yf.download(
            config.symbol, 
            start=config.start_date, 
            end=config.end_date or datetime.date.today().isoformat(),
            interval=config.interval,
            auto_adjust=False,
            prepost=True,
            threads=True,
            progress=True
        )
        
        if df.empty:
            raise ValueError("Fetched data is empty. Check symbol and date range.")
        
        print(f"Raw data shape: {df.shape}")
        
        # Handle missing data
        initial_missing = df.isnull().sum().sum()
        if initial_missing > 0:
            print(f"Found {initial_missing} missing values, applying forward/backward fill")
            df = df.ffill().bfill()
        
        # Clean up column names if they're multi-level (happens with single symbol)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the expected columns
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_cols = [col for col in expected_cols if col in df.columns]
        
        if 'Close' not in available_cols:
            raise ValueError("Close price not found in downloaded data")
        
        # Return only the Close column, renamed for consistency
        result = df[['Close']].copy()
        result.columns = ['close']
        
        # Remove any remaining NaN values
        result = result.dropna()
        
        if len(result) == 0:
            raise ValueError("No valid data after cleaning")
        
        print(f"Clean data shape: {result.shape}")
        print(f"Date range: {result.index[0]} to {result.index[-1]}")
        
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
        target_values = df['close'].values
        
        # Ensure we have enough data
        min_length = config.context_length + config.prediction_length
        if len(target_values) < min_length:
            raise ValueError(
                f"Not enough data points. Need at least {min_length}, got {len(target_values)}"
            )
        
        print(f"Creating dataset with {len(target_values)} data points")
        
        # Create the dataset entry
        data = [{
            'target': target_values,
            'start': df.index[0]
        }]
        
        # Determine frequency based on interval
        freq_mapping = {
            '1m': 'T',    # Minute
            '5m': '5T',   # 5 Minutes  
            '15m': '15T', # 15 Minutes
            '30m': '30T', # 30 Minutes
            '1h': 'H',    # Hour
            '1d': 'D',    # Day
            '1wk': 'W',   # Week
            '1mo': 'M'    # Month
        }
        
        freq = freq_mapping.get(config.interval, 'D')
        
        print(f"Using frequency: {freq}")
        
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
    try:
        mean = df['close'].mean()
        std = df['close'].std()
        
        if std == 0:
            print("Warning: Standard deviation is 0, using std=1 for normalization")
            std = 1.0
        
        normalized_df = df.copy()
        normalized_df['close'] = (df['close'] - mean) / std
        
        print(f"Normalization stats - Mean: {mean:.4f}, Std: {std:.4f}")
        
        return normalized_df, mean, std
        
    except Exception as e:
        raise ValueError(f"Data normalization failed: {str(e)}")


def denormalize_data(normalized_values: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Denormalize the generated samples back to original scale."""
    return normalized_values * std + mean


def create_sliding_windows(data: pd.DataFrame, window_size: int, prediction_length: int) -> list:
    """Create sliding windows for training data.
    
    Args:
        data: Time series dataframe
        window_size: Size of input window
        prediction_length: Length of prediction horizon
        
    Returns:
        List of windowed samples
    """
    samples = []
    total_length = len(data)
    
    for i in range(total_length - window_size - prediction_length + 1):
        # Input window
        input_window = data.iloc[i:i + window_size]['close'].values
        
        # Target window (what we want to predict)
        target_window = data.iloc[i + window_size:i + window_size + prediction_length]['close'].values
        
        sample = {
            'input': input_window,
            'target': target_window,
            'start_date': data.index[i],
            'end_date': data.index[i + window_size + prediction_length - 1]
        }
        
        samples.append(sample)
    
    return samples


def validate_data_quality(df: pd.DataFrame, config: DataConfig) -> dict:
    """Validate data quality and return statistics.
    
    Args:
        df: DataFrame to validate
        config: Configuration used for data fetching
        
    Returns:
        Dictionary with validation results
    """
    stats = {
        'total_points': len(df),
        'missing_values': df.isnull().sum().sum(),
        'date_range': (df.index[0], df.index[-1]),
        'price_range': (df['close'].min(), df['close'].max()),
        'mean_price': df['close'].mean(),
        'volatility': df['close'].std(),
        'sufficient_data': len(df) >= (config.context_length + config.prediction_length),
        'data_gaps': [],
        'outliers': []
    }
    
    # Check for data gaps (missing trading days for daily data)
    if config.interval == '1d':
        date_diff = df.index.to_series().diff()
        large_gaps = date_diff > pd.Timedelta(days=5)  # Weekend + holidays
        if large_gaps.any():
            stats['data_gaps'] = date_diff[large_gaps].tolist()
    
    # Check for outliers (price changes > 10% in one period)
    price_changes = df['close'].pct_change().abs()
    outliers = price_changes > 0.1
    if outliers.any():
        stats['outliers'] = df[outliers].index.tolist()
    
    return stats


if __name__ == "__main__":
    try:
        # Test the updated data fetching
        print("Testing updated data fetching...")
        
        config = DataConfig(
            start_date='2023-01-01',
            end_date='2024-01-01',
            interval='1d',
            context_length=30,
            prediction_length=10
        )
        
        # Fetch data
        df = fetch_sp500_data(config)
        print(f"✓ Fetched data shape: {df.shape}")
        
        # Validate data quality
        quality_stats = validate_data_quality(df, config)
        print(f"✓ Data validation completed")
        print(f"  - Total points: {quality_stats['total_points']}")
        print(f"  - Date range: {quality_stats['date_range'][0]} to {quality_stats['date_range'][1]}")
        print(f"  - Price range: ${quality_stats['price_range'][0]:.2f} - ${quality_stats['price_range'][1]:.2f}")
        print(f"  - Sufficient data: {quality_stats['sufficient_data']}")
        
        # Test dataset creation
        dataset = prepare_gluonts_dataset(df, config)
        print(f"✓ Dataset created with {len(dataset)} series")
        
        # Test normalization
        norm_df, mean, std = normalize_data(df)
        print(f"✓ Normalization completed - Mean: {mean:.2f}, Std: {std:.2f}")
        
        # Test sliding windows
        windows = create_sliding_windows(df, config.context_length, config.prediction_length)
        print(f"✓ Created {len(windows)} sliding windows")
        
        print("\n✅ All data processing tests passed!")
        
    except ValidationError as ve:
        print(f"❌ Config error: {ve}")
    except ValueError as ve:
        print(f"❌ Processing error: {ve}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()