import pandas as pd
import numpy as np
import yfinance as yf
import os
from config import DATA_PATH, TECH_STOCKS, DATA_PERIOD, INDICATORS

class DataCollection:

    """
    Collects and processes stock data with technical indicators.
    
    Why we need this class:
    - Centralizes all data collection logic
    - Ensures consistent data format across all stocks
    - Calculates technical indicators that help predict price movements
    """
    def __init__(self):
        self.data_path = DATA_PATH
        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)

    def stock_fetch_data(self, stock_symbol, period=DATA_PERIOD):
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            stock_symbol (str): Stock ticker symbol (e.g., 'AAPL').
            period (str): Data period to fetch (default is "2y").
        
        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period=period, interval='1d')

        if data.empty:
            raise ValueError(f"No data found for {stock_symbol} for the period {period}")
            return None
        
        data.columns = data.columns.str.strip()

        data['Symbol'] = stock_symbol
        print(f"  ✅ Fetched {len(data)} days of data for {symbol}")
        return data
    
    
        





