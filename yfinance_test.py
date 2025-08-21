import yfinance as yf
stock = yf.Ticker('AAPL')  # Apple Inc.
data = stock.history()  # Try 1 month first
print(f"Shape: {data.shape}")
print(data.head())
