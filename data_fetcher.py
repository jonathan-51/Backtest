import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from ib_insync import IB, Stock, util
from config import BacktestConfig, DataFetcherConfig, DataConfig

class DataFetcher:
    def __init__(self, config: DataFetcherConfig, data_config: DataConfig):
        self.config = config
        self.data_config = data_config
        self.ib = IB()
        self.connected = False
        self.cache_dir = Path(data_config.data_path)
        self.logger = None
        self.request_count = 0
        pass

    def connect(self):
        # Check if already connected
        if self.connected:
            print("Already connected to IB")
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
                print(f"Connected to IB at {self.config.tws_host}:{self.config.tws_port}")
                return True
            else:
                print("Connection failed")
                self.connected = False
                return False
        
        # Handle errors
        except Exception as e:
            print(f"Failed to connect to IB: {e}")
            self.connected = False
            return False
    
    def fetch_historical_data(self,symbol:str):

        # Check connection
        if not self.connected:
            raise ConnectionError("Not connected to IB. Call connect() first")
        
        # Try cache first
        if self.data_config.cache_data:
            cached_df = self.load_from_cache(symbol)
            if cached_df is not None:
                print(f"Loaded {symbol} from cache ({len(cached_df)} bars)")
                return cached_df
        print(f"Fetching {symbol} from IB...")

        # Create Contract
        contract = Stock(symbol, 'SMART', 'USD')

        # Map timeframe
        timeframe_map = {
            '1m': '1 min',
            '5m': '5 mins',
            '15m': '15 mins',
            '30m': '30 mins',
            '1h': '1 hour',
            '1d': '1 day',
            '1w': '1 week',
            '1M': '1 month'
        }

        bar_size = timeframe_map.get(self.data_config.timeframe,'1 day')

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
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        if not bars:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df = util.df(bars)

        # Clean data
        df = self._clean_data(df)

        # Validate data
        if not self._validation_data(df):
            print(f"Data validation failed for {symbol}")
            return pd.DataFrame()
        
        # Save to cache
        if self.data_config.cache_data:
            self.save_to_cache(symbol,df)

        # Return
        print(f"Fetched {len(df)} bars for {symbol}")
        return df

    def disconnect(self):

        # Check if connected 
        if not self.connected:
            print("Not connected to IB")
            return True
        
        # Try to disconnect
        try:
            self.ib.disconnect()
            self.connected = False
            print("Disconnected from IB")
            return True
        
        # Handle errors
        except Exception as e:
            print(f"Error disconnecting from IB: {e}")
            self.connected = False
            return False

    
    def _clean_data(self,data):
        return data
    def _validation_data(self,data):
        return True
    def save_to_cache(self,a,b):
        pass
    def load_from_cache(self,a):
        return None
    def fetch_all_symbols(self):
        pass

data_fetcher_config = DataFetcherConfig()
data_config = DataConfig()
config = BacktestConfig()

fetcher = DataFetcher(data_fetcher_config,data_config)
result = fetcher.connect()

aapl_data = fetcher.fetch_historical_data('AAPL')
fetcher.disconnect()
print(fetcher.connected)