import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from ib_insync import IB, Stock, util
from config import DataFetcherConfig, DataConfig

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
    
    def fetch_historical_data(self):
        pass
    
    def disconnect(self):
        pass
    
    def _clean_data(self):
        pass
    def _validation_data(self):
        pass
    def save_to_cache(self):
        pass
    def load_from_cache(self):
        pass
    def fetch_all_symbols(self):
        pass

data_fetcher_config = DataFetcherConfig()
data_config = DataConfig()

fetcher = DataFetcher(data_fetcher_config,data_config)
result = fetcher.connect()
print(result)
print(fetcher.connected)