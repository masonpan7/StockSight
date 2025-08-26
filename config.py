# config.py
"""
Configuration file for Stock Price Movement Classifier
All project settings are centralized here for easy management.
"""

# Top 10 tech stocks to analyze
TECH_STOCKS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Google (Alphabet)
    'AMZN',   # Amazon
    'TSLA',   # Tesla
    'META',   # Meta (Facebook)
    'NVDA',   # NVIDIA
    'NFLX',   # Netflix
    'CRM',    # Salesforce
    'ORCL'    # Oracle
]

# Data settings
DATA_PERIOD = "2y"        # How much historical data to fetch
SEQUENCE_LENGTH = 60      # Days of data to use for prediction
TRAIN_SPLIT = 0.8        # 80% for training, 20% for testing

# Model settings
EPOCHS = 50              # Number of training iterations
BATCH_SIZE = 32          # Number of samples per training batch
LEARNING_RATE = 0.005    # How fast the model learns

# Technical indicators to calculate
INDICATORS = [
    'SMA_20',    # 20-day Simple Moving Average
    'SMA_50',    # 50-day Simple Moving Average
    'RSI',       # Relative Strength Index
    'MACD',      # Moving Average Convergence Divergence
    'BB_upper',  # Bollinger Band Upper
    'BB_lower',  # Bollinger Band Lower
    'Volume_MA'  # Volume Moving Average
]

# File paths
DATA_PATH = 'data/'
MODEL_PATH = 'models/'
RESULTS_PATH = 'results/'
