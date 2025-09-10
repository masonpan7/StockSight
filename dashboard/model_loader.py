import torch
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import os
from models import LSTMModel, GRUModel
from config import MODEL_PATH, SEQUENCE_LENGTH

class ModelLoader:
    """Handles loading trained models and making predictions for the dashboard."""
    
    def __init__(self, model_path=MODEL_PATH, sequence_length=SEQUENCE_LENGTH):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = self._get_device()
        
        # Use OrderedDict to maintain insertion order, with GRU first
        from collections import OrderedDict
        self.models = OrderedDict()
        self.scalers = {}
        
        self.stock_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'META', 'NFLX', 'NVDA', 'AMD', 'INTC'
        ]
    
    def _get_device(self):
        """Determine the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_models(self):
        """Load all trained models with desired order: GRU, Random Forest, LSTM, Logistic Regression."""
        try:
            # Load GRU first
            gru_file = f"{self.model_path}gru_model.pth"
            if os.path.exists(gru_file):
                checkpoint = torch.load(gru_file, map_location=self.device)
                input_size = checkpoint['input_size']
                model = GRUModel(input_size).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models['GRU'] = model
                print("Loaded GRU model")
            
            # Load Random Forest second
            rf_file = f"{self.model_path}random_forest.pkl"
            if os.path.exists(rf_file):
                with open(rf_file, 'rb') as f:
                    model, scaler = pickle.load(f)
                    self.models['Random Forest'] = (model, scaler)
                print("Loaded Random Forest")
            
            # Load LSTM third
            '''
            lstm_file = f"{self.model_path}lstm_model.pth"
            if os.path.exists(lstm_file):
                checkpoint = torch.load(lstm_file, map_location=self.device)
                input_size = checkpoint['input_size']
                model = LSTMModel(input_size).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models['LSTM'] = model
                print("Loaded LSTM model")
            '''
            
            # Load Logistic Regression last
            lr_file = f"{self.model_path}logistic_regression.pkl"
            if os.path.exists(lr_file):
                with open(lr_file, 'rb') as f:
                    model, scaler = pickle.load(f)
                    self.models['Logistic Regression'] = (model, scaler)
                print("Loaded Logistic Regression")
            
            # Load feature scaler
            scaler_file = f"{self.model_path}feature_scaler.pkl"
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scalers['feature'] = pickle.load(f)
                print("Loaded feature scaler")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_stock_data(self, symbol, period='6mo'):
        """Fetch recent stock data for a symbol."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if len(data) == 0:
                return None
                
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, data):
        """Add the exact same technical indicators used in training."""
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        rs = avg_gains / avg_losses
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp_fast = data['Close'].ewm(span=12).mean()
        exp_slow = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp_fast - exp_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = sma + (std * 2)
        data['BB_lower'] = sma - (std * 2)
        
        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # Price change indicators
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        return data
    
    def predict_stock(self, symbol, model_name='GRU', threshold=0.5):
        """Make a prediction for a single stock using the specified model."""
        if model_name not in self.models:
            print(f"Model {model_name} not available")
            return None
            
        data = self.get_stock_data(symbol)
        if data is None or len(data) < self.sequence_length:
            print(f"Insufficient data for {symbol}")
            return None
        
        try:
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
                'BB_upper', 'BB_lower', 'Volume_MA',
                'Price_Change', 'High_Low_Pct'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].dropna()
            
            if len(X) < self.sequence_length:
                print(f"Not enough clean data for {symbol}")
                return None
            
            X_recent = X.iloc[-self.sequence_length:].values
            
            if 'feature' in self.scalers:
                X_scaled = self.scalers['feature'].transform(X_recent)
            else:
                print("Warning: No feature scaler found")
                X_scaled = X_recent
            
            if model_name in ['LSTM', 'GRU']:
                X_sequence = X_scaled.reshape(1, self.sequence_length, -1)
                X_tensor = torch.FloatTensor(X_sequence).to(self.device)
                
                model = self.models[model_name]
                model.eval()
                
                with torch.no_grad():
                    prediction = model(X_tensor, return_logits=False).cpu().numpy()[0][0]
            
            else:
                model, scaler = self.models[model_name]
                X_flat = X_scaled.reshape(1, -1)
                X_scaled_flat = scaler.transform(X_flat)
                prediction = model.predict_proba(X_scaled_flat)[0][1]
            
            direction = "UP" if prediction > threshold else "DOWN"
            confidence = prediction if prediction > threshold else 1 - prediction
            
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            price_change = ((current_price - previous_price) / previous_price) * 100
            
            result = {
                'symbol': symbol,
                'prediction': direction,
                'probability': float(prediction),
                'confidence': float(confidence),
                'current_price': float(current_price),
                'price_change_pct': float(price_change),
                'model_used': model_name,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting {symbol} with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_all_stocks(self, model_name='GRU', threshold=0.5):
        """Make predictions for all stocks in the portfolio."""
        predictions = []
        
        for symbol in self.stock_symbols:
            prediction = self.predict_stock(symbol, model_name, threshold)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def get_available_models(self):
        """Return list of available models with GRU first."""
        return list(self.models.keys())