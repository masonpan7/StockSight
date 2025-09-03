import torch
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import os

# Import your model classes (you'll need to copy these to the dashboard folder)
from model_trainer import LSTMModel, GRUModel  # Adjust import path

class ModelLoader:
    """
    Handles loading trained models and making predictions for the dashboard.
    """
    
    def __init__(self, model_path='models/', sequence_length=30):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = self._get_device()
        
        # Storage for loaded models
        self.models = {}
        self.scalers = {}
        
        # Stock symbols to predict (customize this list)
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
    
    @st.cache_data
    def load_models(_self):
        """
        Load all trained models. Uses Streamlit caching for performance.
        The _self parameter with underscore prevents Streamlit from hashing the object.
        """
        try:
            # Load PyTorch models
            for model_type in ['lstm', 'gru']:
                model_file = f"{_self.model_path}{model_type}_model.pth"
                if os.path.exists(model_file):
                    checkpoint = torch.load(model_file, map_location=_self.device)
                    
                    # Create model instance
                    input_size = checkpoint['input_size']
                    if model_type == 'lstm':
                        model = LSTMModel(input_size).to(_self.device)
                    else:
                        model = GRUModel(input_size).to(_self.device)
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    _self.models[model_type.upper()] = model
                    print(f"✅ Loaded {model_type.upper()} model")
            
            # Load traditional ML models
            for model_name in ['random_forest', 'logistic_regression']:
                model_file = f"{_self.model_path}{model_name}.pkl"
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model, scaler = pickle.load(f)
                        _self.models[model_name.replace('_', ' ').title()] = (model, scaler)
                    print(f"✅ Loaded {model_name}")
            
            # Load feature scaler
            scaler_file = f"{_self.model_path}feature_scaler.pkl"
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    _self.scalers['feature'] = pickle.load(f)
                print("✅ Loaded feature scaler")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    @st.cache_data
    def get_stock_data(_self, symbol, period='3mo'):
        """
        Fetch recent stock data for a symbol.
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if len(data) == 0:
                return None
                
            # Add technical indicators
            data = _self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, data):
        """Add the same technical indicators used in training."""
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gains = delta.where(delta > 0, 0).rolling(14).mean()
        losses = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gains / losses
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = data['Close'].ewm(span=12).mean()
        exp26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp12 - exp26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        data['BB_upper'] = sma20 + (2 * std20)
        data['BB_lower'] = sma20 - (2 * std20)
        
        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        
        # Price Change and High-Low Percentage
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        return data
    
    def predict_stock(self, symbol, model_name='GRU', threshold=0.5):
        """
        Make a prediction for a single stock using the specified model.
        """
        if model_name not in self.models:
            return None
            
        # Get fresh data
        data = self.get_stock_data(symbol)
        if data is None or len(data) < self.sequence_length:
            return None
        
        try:
            # Prepare features (same as training)
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
                'BB_upper', 'BB_lower', 'Volume_MA',
                'Price_Change', 'High_Low_Pct'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].dropna()
            
            if len(X) < self.sequence_length:
                return None
            
            # Take the last sequence_length days
            X_recent = X.iloc[-self.sequence_length:].values
            
            # Scale features
            if 'feature' in self.scalers:
                X_scaled = self.scalers['feature'].transform(X_recent)
            else:
                X_scaled = X_recent  # Fallback if no scaler
            
            # Make prediction based on model type
            if model_name in ['LSTM', 'GRU']:
                # Neural network prediction
                X_sequence = X_scaled.reshape(1, self.sequence_length, -1)
                X_tensor = torch.FloatTensor(X_sequence).to(self.device)
                
                model = self.models[model_name]
                model.eval()
                
                with torch.no_grad():
                    prediction = model(X_tensor, return_logits=False).cpu().numpy()[0][0]
            
            else:
                # Traditional ML prediction
                model, scaler = self.models[model_name]
                X_flat = X_scaled.reshape(1, -1)
                X_scaled_flat = scaler.transform(X_flat)
                prediction = model.predict_proba(X_scaled_flat)[0][1]
            
            # Determine direction and confidence
            direction = "UP" if prediction > threshold else "DOWN"
            confidence = prediction if prediction > threshold else 1 - prediction
            
            # Get current price and other info
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
            print(f"❌ Error predicting {symbol} with {model_name}: {e}")
            return None
    
    def predict_all_stocks(self, model_name='LSTM', threshold=0.5):
        """
        Make predictions for all stocks in the portfolio.
        """
        predictions = []
        
        for symbol in self.stock_symbols:
            prediction = self.predict_stock(symbol, model_name, threshold)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def get_available_models(self):
        """Return list of available models."""
        return list(self.models.keys())