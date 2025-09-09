import pandas as pd
import numpy as np
import yfinance as yf
import os
from config import DATA_PATH, TECH_STOCKS, DATA_PERIOD, INDICATORS

class DataCollection:
    """
    Collects and processes stock data with technical indicators.
    Centralizes data collection logic and ensures consistent format across all stocks.
    """
    def __init__(self):
        self.data_path = DATA_PATH
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
        print(f"Fetched {len(data)} days of data for {stock_symbol}")
        return data
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average."""
        return data['Close'].rolling(window=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index."""
        delta = data['Close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp_fast = data['Close'].ewm(span=fast).mean()
        exp_slow = data['Close'].ewm(span=slow).mean()
        macd = exp_fast - exp_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(data, window)
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def add_technical_indicators(self, data):
        """Add technical indicators to the dataset."""
        print("Calculating technical indicators...")
        
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
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        print("Technical indicators calculated")
        return data
    
    def create_target_variable(self, data):
        """
        Create the target variable for classification.
        Target = 1 if tomorrow's close > today's close (UP)
        Target = 0 if tomorrow's close <= today's close (DOWN)
        """
        data['Next_Close'] = data['Close'].shift(-1)
        data['Target'] = (data['Next_Close'] > data['Close']).astype(int)
        data['Next_Day_Change'] = ((data['Next_Close'] - data['Close']) / data['Close']) * 100
        data = data[:-1].copy()
        
        print(f"Target distribution: {data['Target'].value_counts().to_dict()}")
        return data

    def clean_data(self, data):
        """Remove NaN values and extreme outliers from the dataset."""
        print("Cleaning data...")
        
        original_len = len(data)
        data = data.dropna()
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for column in ['Close', 'Volume', 'RSI']:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                data = data[abs(data[column] - mean) <= (3 * std)]
        
        print(f"Removed {original_len - len(data)} rows ({((original_len - len(data)) / original_len * 100):.1f}%)")
        return data
    
    def save_data(self, data, symbol):
        """Save processed data to CSV file."""
        filename = f"{self.data_path}{symbol}_processed.csv"
        data.to_csv(filename, index=True)
        print(f"Saved data to {filename}")

    def collect_all_stocks(self):
        """
        Collect and process data for all tech stocks.
        Main orchestration function that handles the complete data pipeline.
        """
        print("Starting data collection for all tech stocks...")
        print("=" * 50)
        
        all_data = []
        successful_stocks = []
        
        for symbol in TECH_STOCKS:
            print(f"\nProcessing {symbol}...")
            
            raw_data = self.stock_fetch_data(symbol)
            if raw_data is None:
                continue
            
            processed_data = self.add_technical_indicators(raw_data)
            processed_data = self.create_target_variable(processed_data)
            processed_data = self.clean_data(processed_data)
            
            self.save_data(processed_data, symbol)
            
            all_data.append(processed_data)
            successful_stocks.append(symbol)
            
            print(f"{symbol} completed: {len(processed_data)} samples")
        
        # Combine all stock data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_filename = f"{self.data_path}all_stocks_combined.csv"
            combined_data.to_csv(combined_filename, index=False)
            
            print(f"\nData collection completed!")
            print(f"Successfully processed: {successful_stocks}")
            print(f"Combined dataset: {len(combined_data)} samples")
            print(f"Saved to: {combined_filename}")
            
            return combined_data
        else:
            print("No data collected successfully")
            return None

    def get_data_summary(self):
        """Display a summary of collected data."""
        try:
            combined_file = f"{self.data_path}all_stocks_combined.csv"
            if not os.path.exists(combined_file):
                print("No combined data found. Run collect_all_stocks() first.")
                return
            
            data = pd.read_csv(combined_file)
            
            print("\nDATA SUMMARY")
            print("=" * 30)
            print(f"Total samples: {len(data):,}")
            print(f"Date range: {data['Date'].min() if 'Date' in data.columns else 'Unknown'} to {data['Date'].max() if 'Date' in data.columns else 'Unknown'}")
            print(f"Stocks: {data['Symbol'].unique().tolist()}")
            print(f"\nTarget distribution:")
            target_dist = data['Target'].value_counts()
            for target, count in target_dist.items():
                direction = "UP" if target == 1 else "DOWN"
                percentage = (count / len(data)) * 100
                print(f"  {direction}: {count:,} ({percentage:.1f}%)")
            
            print(f"\nFeatures available: {len(data.columns)}")
            print(f"Features: {list(data.columns)}")
            
        except Exception as e:
            print(f"Error loading data summary: {e}")

def main():
    """Main function to run data collection."""
    collector = DataCollection()
    data = collector.collect_all_stocks()
    collector.get_data_summary()
    
    if data is not None:
        print(f"\nReady for model training!")
        print(f"Next step: python model_trainer.py")

if __name__ == "__main__":
    main()