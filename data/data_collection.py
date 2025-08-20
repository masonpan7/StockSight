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
        print(f"  ✅ Fetched {len(data)} days of data for {stock_symbol}")
        return data
    
    def calculate_sma(self, data, window):
        """
        Calculate Simple Moving Average.
        
        SMA smooths out price data to identify trends.
        A stock above its SMA is often in an uptrend.
        """
        return data['Close'].rolling(window=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """
        Calculate Relative Strength Index.
        
        RSI measures how fast prices are changing.
        Values above 70 suggest overbought (might go down)
        Values below 30 suggest oversold (might go up)
        """
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD shows the relationship between two moving averages.
        When MACD crosses above signal line, it might indicate upward momentum.
        """
        # Calculate fast and slow EMAs
        exp_fast = data['Close'].ewm(span=fast).mean()
        exp_slow = data['Close'].ewm(span=slow).mean()
        
        # MACD is the difference
        macd = exp_fast - exp_slow
        
        # Signal line is EMA of MACD
        signal_line = macd.ewm(span=signal).mean()
        
        return macd, signal_line

    

        





