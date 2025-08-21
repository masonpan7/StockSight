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
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands show volatility around a moving average.
        When price hits upper band, it might be overbought.
        When price hits lower band, it might be oversold.
        """
        # Calculate middle band (SMA)
        sma = self.calculate_sma(data, window)
        
        # Calculate standard deviation
        std = data['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, lower_band

    def add_technical_indicators(self, data):
        """
        Add all technical indicators to the dataset.
        
        Why these indicators:
        - SMA: Shows trend direction
        - RSI: Shows momentum
        - MACD: Shows trend changes
        - Bollinger Bands: Shows volatility and potential reversal points
        - Volume MA: Shows buying/selling interest
        """
        print("  📊 Calculating technical indicators...")
        
        # Simple Moving Averages
        data['SMA_20'] = self.calculate_sma(data, 20)
        data['SMA_50'] = self.calculate_sma(data, 50)
        
        # RSI
        data['RSI'] = self.calculate_rsi(data)
        
        # MACD
        macd, signal = self.calculate_macd(data)
        data['MACD'] = macd
        data['MACD_Signal'] = signal
        
        # Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger_bands(data)
        data['BB_upper'] = bb_upper
        data['BB_lower'] = bb_lower
        
        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # Price change indicators
        data['Price_Change'] = data['Close'].pct_change()  # Daily return
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']  # Volatility
        
        print("  ✅ Technical indicators calculated")
        return data
    
    
    

        





