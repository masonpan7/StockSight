# Configuration file for dashboard
# These values must match your training configuration

# Data paths
DATA_PATH = './data/'
MODEL_PATH = './models/'
RESULTS_PATH = './results/'

# Model parameters
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8

# Stock symbols
TECH_STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'META', 'NFLX', 'NVDA', 'AMD', 'INTC'
]

DATA_PERIOD = "2y"
INDICATORS = ['SMA', 'RSI', 'MACD', 'BB']