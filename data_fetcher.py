import pandas as pd
from datetime import datetime
from pathlib import Path
from ib_insync import IB, Stock, util
from config import DataFetcherConfig, DataConfig
import logging
from typing import Dict
import time
from dateutil.relativedelta import relativedelta

class DataFetcher:
    """Fetches historical OHLCVA data from Interactive Brokers
    
    Manages the IB connection lifecycle, retrieves and caches historical 
    bars, and performs data cleaning and validation
    """

    TIMEFRAME_MAP = {
        '1m': '1 min',
        '5m': '5 mins',
        '15m': '15 mins',
        '30m': '30 mins',
        '1h': '1 hour',
        '1d': '1 day',
        '1w': '1 week',
        '1M': '1 month'
    }

    INTRADAY_TIMEFRAMES = {'1m','5m','15m','30m','1h'}

    def __init__(self, config: DataFetcherConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self.ib = IB()
        self.connected = False
        self.cache_dir = Path(data_config.data_path)
        self.logger = logging.getLogger(__name__)
        self.request_count = 0

    def connect(self) -> bool:
        """Establish a readonly connection to IB TWS."""
        # Check if already connected
        if self.connected:
            self.logger.info("Already connected to IB")
            return True
        
        # Try to connect
        try:
            self.ib.connect(
                host=self.config.tws_host,
                port=self.config.tws_port,
                clientId=self.config.client_id,
                readonly=True # ONLY WHILE BACKTESTING
            )
            
            # Verify connection
            if self.ib.isConnected():
                self.connected = True
                self.logger.info(f"Connected to IB at {self.config.tws_host}:{self.config.tws_port}")
                return True
            else:
                self.logger.error("Connection failed")
                self.connected = False
                return False
        
        # Handle errors
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            self.connected = False
            return False
    
    def fetch_historical_data(self,symbol:str, timeframe:str) -> pd.DataFrame | None:
        """Fetch, clean, and validate historical bars for a symbol and timeframe."""
        
        # Check connection
        if not self.connected:
            self.logger.error("Not connected to IB. Call connect() first")
            return None
        if not symbol or not isinstance(symbol,str):
            self.logger.error(f"Symbol must be non-empty string, got: {symbol}")
            return None
        
        # Try cache first
        if self.data_config.cache_data:
            cached_df = self._load_from_cache(symbol,timeframe)
            if cached_df is not None:
                return cached_df
        self.logger.info(f"Fetching {symbol}_{timeframe} from IB...")

        # Create Contract
        contract = Stock(symbol, 'SMART', 'USD')

        bar_size = self.TIMEFRAME_MAP.get(timeframe,'1 day')

        # Fetch bars - route based on timeframe
        if timeframe in self.INTRADAY_TIMEFRAMES:
            all_bars = self._fetch_bars_chunked(contract, bar_size,symbol,timeframe)
        else:
            all_bars = self._fetch_bars_single(contract, bar_size,symbol,timeframe)
        
        if not all_bars:
            return None
        
        # Convert to DataFrame
        df = util.df(all_bars)

        # Clean data
        df = self._clean_data(df)

        # Validate data
        if not self._validation_data(df):
            return None
        
        # Save to cache
        if self.data_config.cache_data:
            self._save_to_cache(symbol,df,timeframe)

        # Return
        self.logger.info(f"Fetched {len(df)} bars for {symbol}_{timeframe}")
        return df

    def _fetch_bars_single(self,contract:Stock,bar_size:str,symbol:str,timeframe:str) -> list:
        """Single IB request for daily/weekly/monthly bars."""
        
        # Request data from IB
        try:
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=self.data_config.end_date,
                durationStr=self.data_config.lookback_period,
                barSizeSetting=bar_size,
                whatToShow='Trades',
                useRTH=True,
                formatDate=1,
            )
            return bars
        
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}_{timeframe}: {e}")
            return []

    def _fetch_bars_chunked(self,contract:Stock,bar_size:str,symbol:str,timeframe:str) -> list:
        """Fetch intraday bars in monthly chunks to avoid IB timeouts."""

        all_bars = []
        chunk_end = self.data_config.end_date

        # Calculate start date from lookback_period (e.g. "1 Y" -> 1 year back)
        amount, unit = self.data_config.lookback_period.split()
        amount = int(amount)
        unit_map = {'D': 'days', 'W': 'weeks', 'M': 'months', 'Y': 'years'}
        start_date = chunk_end - relativedelta(**{unit_map[unit]: amount})

        # Calculate total chunks for progress
        total_chunks = 0
        temp = chunk_end
        while temp > start_date:
            total_chunks += 1
            temp -= relativedelta(months=1)

        for i in range(total_chunks):
            self.logger.info(f"Chunk[{i+1}/{total_chunks}] ending {chunk_end.strftime('%Y-%m-%d')}")

            try:
                bars = self.ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=chunk_end,
                    durationStr='1 M',
                    barSizeSetting=bar_size,
                    whatToShow='Trades',
                    useRTH=True,
                    formatDate=1,
                    timeout=120,
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}_{timeframe}: {e}")
                return []

            if bars:
                all_bars.extend(bars)

            # Move back 1 month
            chunk_end -= relativedelta(months=1)

            # Rate limit between chunks (skip after last)
            if i < total_chunks - 1:
                time.sleep(self.config.request_pause)

        return all_bars

    def fetch_all_timeframes(self,symbol:str) -> Dict[str,pd.DataFrame]:
        """Fetch all configured timeframes for a symbol, returning a dict keyed by timeframe."""
        results = {}
        timeframes = self.data_config.timeframes

        for i, timeframe in enumerate(timeframes):
            self.logger.info(f"[{i+1}/{len(timeframes)}] Fetching {symbol} {timeframe}...")

            df = self.fetch_historical_data(symbol,timeframe)

            if df is not None:
                results[timeframe] = df
            if i < len(timeframes) - 1:
                time.sleep(self.config.request_pause)

        return results

    def disconnect(self) -> bool:
        """Close the IB TWS connection."""
        # Check if connected
        if not self.connected:
            self.logger.info("Not connected to IB")
            return True
        
        # Try to disconnect
        try:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IB")
            return True
        
        # Handle errors
        except Exception as e:
            self.logger.error(f"Error disconnecting from IB: {e}")
            self.connected = False
            return False

    def _clean_data(self,df: pd.DataFrame) -> pd.DataFrame:
        """Fix dtypes, remove duplicates, and handle OHLCV anomalies."""
        if df.empty:
            return df
        
        # Remove average column
        df = df[['date','open','high','low','close','volume']]

        # Remove duplicates
        df = df.drop_duplicates(subset=['date'], keep='first')

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Fix data types
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['close'] = df['close'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(int)

        # Check for missing values
        if df.isnull().any().any():
            self.logger.warning(f"Warning: Found {df.isnull().sum().sum()} missing values")
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')

        # Check for negative prices
        if (df[['open','high','close','low']] < 0).any().any():
            self.logger.warning("Warning: Found negative values")
        
        # Zero volume
        if (df['volume'] == 0).any():
            self.logger.warning(f"Warning: Found {(df['volume'] == 0).sum()} bars with zero volume")

        # Check for High < Low
        if (df['high'] < df['low']).any():
            self.logger.warning('Warning: Found High < Low')
            df[['high','low']] = df[['low','high']]

        # Check for close outside high-low range
        invalid = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid.any():
            self.logger.warning(f"Warning: Found {invalid.sum()} bars with close outside high-low range")

        self.logger.info(f"Cleaned Data: {len(df)} bars")
        return df
    
    def _validation_data(self,df: pd.DataFrame) -> bool:
        """Run integrity checks on OHLCV data, returning False on failure."""
        # Check if DataFrame is empty
        if df.empty:
            self.logger.error("Validation failed: Empty Dataframe")
            return False

        # Check required columns exist
        required_cols = ['date','open','high','low','close','volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Validation failed: Missing required columns. Expect {required_cols}")
            return False
        
        # Check minimum data points
        if len(df) < self.data_config.min_bars_required:
            self.logger.error(f"Validation failed: Only {len(df)} bars, need at least {self.data_config.min_bars_required}")
            return False
        
        # Check no NaN values remain
        if df.isnull().any().any():
            self.logger.error("Validation failed; Found Nan values")
            return False
        
        # Check if any High < Open
        if (df['high'] < df['open']).any():
            self.logger.error("Validation failed: Found High < Open")
            return False

        # Check if any High < Close
        if (df['high'] < df['close']).any():
            self.logger.error("Validation failed: Found High < Close")
            return False

        # Check if any Low > Open
        if (df['low'] > df['open']).any():
            self.logger.error("Validation failed: Found Low > Open")
            return False
        
        # Check if any Low > Close
        if (df['low'] > df['close']).any():
            self.logger.error(f"Validation failed: Found Low < Close")
            return False 
        
        # Check volume is positive
        if not (df['volume'] > 0).all():
            self.logger.error("Validation failed: Found non-positive volume")
            return False
        
        # Check latest date is not in the future
        if df['date'].max().date() > datetime.now().date():
            self.logger.error("Validation failed: Data contains future dates")
            return False
        
        # Check dates are sorted and in ascending order
        if not df['date'].is_monotonic_increasing:
            self.logger.error("Validation failed: Dates are not sorted in ascending order")
            return False

        self.logger.info("Validation passed")
        return True
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame, timeframe:str) -> bool:
        """Write DataFrame to CSV cache file."""
        cache_file = self.cache_dir / f"{symbol}_{timeframe}.csv"

        try:
            # Write DataFrame to CSV with specific options
            df.to_csv(cache_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            self.logger.info(f"Cached {symbol}_{timeframe} ({len(df)} bars) to {cache_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache {symbol}_{timeframe}: {e}")
            return False
        
    def _load_from_cache(self, symbol: str, timeframe:str) -> pd.DataFrame | None:
        """Load DataFrame from CSV cache file, or None if not cached."""
        cache_file = self.cache_dir / f"{symbol}_{timeframe}.csv"

        # Check if file exists
        if not cache_file.exists():
            self.logger.debug(f"No cache found for {symbol}_{timeframe}")
            return None

        try:
            # Read CSV and parse dates
            df = pd.read_csv(cache_file, parse_dates=['date'])
            self.logger.info(f"Loaded {symbol}_{timeframe} from cache ({len(df)} bars)")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load cache for {symbol}_{timeframe}: {e}")
            return None
        
    def fetch_all_symbols(self):
        pass
