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
    
    def create_target_variable(self, data):
        """
        Create the target variable for classification.
        
        Target = 1 if tomorrow's close > today's close (UP)
        Target = 0 if tomorrow's close <= today's close (DOWN)
        
        This is what we're trying to predict!
        """
        # Shift close price to get next day's price
        data['Next_Close'] = data['Close'].shift(-1)
        
        # Create binary target: 1 for up, 0 for down
        data['Target'] = (data['Next_Close'] > data['Close']).astype(int)
        
        # Calculate the actual price change for evaluation
        data['Next_Day_Change'] = ((data['Next_Close'] - data['Close']) / data['Close']) * 100
        
        # Remove the last row as it doesn't have a next day
        data = data[:-1].copy()
        
        print(f"  🎯 Target distribution: {data['Target'].value_counts().to_dict()}")
        return data

    def clean_data(self, data):
        """
        Clean the dataset by removing NaN values and outliers.
        
        Why we clean data:
        - NaN values break machine learning algorithms
        - Outliers can mislead the model
        - Clean data = better model performance
        """
        print("  🧹 Cleaning data...")
        
        # Store original length
        original_len = len(data)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for column in ['Close', 'Volume', 'RSI']:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                data = data[abs(data[column] - mean) <= (3 * std)]
        
        print(f"  📉 Removed {original_len - len(data)} rows ({((original_len - len(data)) / original_len * 100):.1f}%)")
        return data
    
    def save_data(self, data, symbol):
        """
        Save processed data to CSV file.
        
        Why save data:
        - Avoid re-downloading same data
        - Faster model training
        - Data backup
        """
        filename = f"{self.data_path}{symbol}_processed.csv"
        data.to_csv(filename, index=True)
        print(f"  💾 Saved data to {filename}")

    def collect_all_stocks(self):
        """
        Collect and process data for all tech stocks.
        
        This is the main function that orchestrates everything:
        1. Fetch raw data
        2. Add technical indicators
        3. Create target variable
        4. Clean data
        5. Save data
        """
        print("🚀 Starting data collection for all tech stocks...")
        print("=" * 50)
        
        all_data = []
        successful_stocks = []
        
        for symbol in TECH_STOCKS:
            print(f"\n📈 Processing {symbol}...")
            
            # Step 1: Fetch raw stock data
            raw_data = self.fetch_stock_data(symbol)
            if raw_data is None:
                continue
            
            # Step 2: Add technical indicators
            processed_data = self.add_technical_indicators(raw_data)
            
            # Step 3: Create target variable (what we want to predict)
            processed_data = self.create_target_variable(processed_data)
            
            # Step 4: Clean the data
            processed_data = self.clean_data(processed_data)
            
            # Step 5: Save individual stock data
            self.save_data(processed_data, symbol)
            
            # Add to combined dataset
            all_data.append(processed_data)
            successful_stocks.append(symbol)
            
            print(f"✅ {symbol} completed: {len(processed_data)} samples")
        
        # Combine all stock data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_filename = f"{self.data_path}all_stocks_combined.csv"
            combined_data.to_csv(combined_filename, index=False)
            
            print(f"\n🎉 Data collection completed!")
            print(f"📊 Successfully processed: {successful_stocks}")
            print(f"📁 Combined dataset: {len(combined_data)} samples")
            print(f"💾 Saved to: {combined_filename}")
            
            return combined_data
        else:
            print("❌ No data collected successfully")
            return None
    def get_data_summary(self):
        """
        Display a summary of collected data.
        
        This helps us understand our dataset before training.
        """
        try:
            # Load combined data
            combined_file = f"{self.data_path}all_stocks_combined.csv"
            if not os.path.exists(combined_file):
                print("❌ No combined data found. Run collect_all_stocks() first.")
                return
            
            data = pd.read_csv(combined_file)
            
            print("\n📊 DATA SUMMARY")
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
            print(f"❌ Error loading data summary: {e}")

def main():
    """
    Main function to run data collection.
    """
    # Create data collector instance
    collector = DataCollection()
    
    # Collect all stock data
    data = collector.collect_all_stocks()
    
    # Show summary
    collector.get_data_summary()
    
    if data is not None:
        print(f"\n✅ Ready for model training!")
        print(f"Next step: python model_trainer.py")

if __name__ == "__main__":
    main()
        





