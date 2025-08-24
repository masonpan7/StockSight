import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from config import *

class StockPredictor:
    """
    Complete stock movement prediction system.
    
    This class handles:
    1. Data preprocessing 
    2. Feature engineering
    3. Model training (LSTM, Traditional ML)
    4. Model comparison
    5. Prediction and evaluation
    """
    
    def __init__(self):
        """Initialize the predictor with paths and scalers."""
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.results_path = RESULTS_PATH
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        # Scalers for normalizing data
        # MinMaxScaler: scales features to 0-1 range (good for neural networks)
        # StandardScaler: standardizes features to mean=0, std=1 (good for traditional ML)
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = StandardScaler()
        
        # Store models for comparison
        self.models = {}
        self.results = {}

    def load_data(self):
        """
        Load the preprocessed stock data.
        
        Returns the combined dataset from all stocks.
        """
        print("📊 Loading stock data...")
        
        data_file = f"{self.data_path}all_stocks_combined.csv"
        if not os.path.exists(data_file):
            print("❌ No data found! Run data_collector.py first.")
            return None
        
        data = pd.read_csv(data_file)
        print(f"✅ Loaded {len(data):,} samples from {len(data['Symbol'].unique())} stocks")
        
        return data
    
    def prepare_features(self, data):
        """
        Select and prepare features for machine learning.
        
        Feature selection is crucial:
        - Use technical indicators that actually predict price movements
        - Remove features that cause data leakage (future information)
        - Handle missing values properly
        """
        print("🔧 Preparing features...")
        
        # Select features for prediction
        # We exclude: Date, Symbol, Next_Close, Next_Day_Change (these contain future info)
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',  # Basic OHLCV
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',  # Technical indicators
            'BB_upper', 'BB_lower', 'Volume_MA',
            'Price_Change', 'High_Low_Pct'  # Price-based features
        ]
        
        # Filter to only include available columns
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"📈 Using {len(available_features)} features: {available_features}")
        
        # Extract features and target
        X = data[available_features].copy()
        y = data['Target'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        print(f"✅ Prepared {len(X)} samples with {X.shape[1]} features")
        print(f"🎯 Target distribution: UP={sum(y)}, DOWN={len(y)-sum(y)}")
        
        return X, y, available_features
    
    def create_sequences(self, X, y, sequence_length=SEQUENCE_LENGTH):
        """
        Create sequences for LSTM training.
        
        LSTM needs sequential data:
        - Input: Last N days of features
        - Output: Next day direction (up/down)
        
        Example: Use days 1-60 to predict day 61
        """
        print(f"🔄 Creating sequences of length {sequence_length}...")
        
        X_sequences = []
        y_sequences = []
        
        # Group by stock symbol to maintain temporal order
        if 'Symbol' in X.columns:
            symbols = X['Symbol'].unique()
            X_no_symbol = X.drop('Symbol', axis=1)
        else:
            # If no symbol column, treat as one continuous sequence
            symbols = ['ALL']
            X_no_symbol = X
        
        for symbol in symbols:
            if symbol == 'ALL':
                symbol_data = X_no_symbol
                symbol_targets = y
            else:
                symbol_mask = X['Symbol'] == symbol
                symbol_data = X_no_symbol[symbol_mask].reset_index(drop=True)
                symbol_targets = y[symbol_mask].reset_index(drop=True)
            
            # Create sequences for this stock
            for i in range(sequence_length, len(symbol_data)):
                # Take sequence_length days of features
                sequence_X = symbol_data.iloc[i-sequence_length:i].values
                sequence_y = symbol_targets.iloc[i]
                
                X_sequences.append(sequence_X)
                y_sequences.append(sequence_y)
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        print(f"✅ Created {len(X_seq)} sequences")
        print(f"📐 Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM neural network model.
        
        Architecture explanation:
        1. LSTM Layer 1: 50 units, returns sequences for next layer
        2. Dropout: Prevents overfitting by randomly turning off neurons
        3. LSTM Layer 2: 50 units, returns sequences  
        4. LSTM Layer 3: 50 units, final sequence processing
        5. Dense layers: Traditional neural network for final classification
        
        Why LSTM?
        - Remembers long-term dependencies in time series
        - Can capture patterns over days/weeks
        - Good for sequential financial data
        """
        print("🧠 Building LSTM model...")
        
        model = Sequential([
            # First LSTM layer: captures initial patterns
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),  # 20% dropout to prevent overfitting
            
            # Second LSTM layer: learns more complex patterns
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer: final sequence processing
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for classification
            Dense(25, activation='relu'),  # Hidden layer
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Output: probability of going up
        ])
        
        # Compile model
        # Adam optimizer: adaptive learning rate
        # Binary crossentropy: good for binary classification
        # Accuracy: easy to understand metric
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ LSTM model built")
        model.summary()
        
        return model
    
    def build_gru_model(self, input_shape):
        """
        Build GRU (Gated Recurrent Unit) neural network model.
        
        GRU is similar to LSTM but simpler and faster:
        - Uses fewer parameters than LSTM
        - Often performs similarly to LSTM
        - Faster training and inference
        - Good alternative if LSTM is too slow
        """
        print("🧠 Building GRU model...")
        
        model = Sequential([
            # First GRU layer
            GRU(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second GRU layer
            GRU(50, return_sequences=True),
            Dropout(0.2),
            
            # Third GRU layer  
            GRU(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for classification
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ GRU model built")
        return model
    
    def train_lstm_model(self, X_train, y_train, X_test, y_test):
        """
        Train LSTM model and track performance.
        
        Why we use validation data:
        - Monitor overfitting during training
        - Stop early if model stops improving  
        - Get unbiased performance estimate
        """
        print("🚀 Training LSTM model...")
        
        # Build model
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Create callbacks for better training
        callbacks = [
            # Stop early if validation loss stops improving
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,  # Wait 10 epochs before stopping
                restore_best_weights=True
            ),
            # Reduce learning rate if stuck
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Cut learning rate in half
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        model_path = f"{self.model_path}lstm_model.h5"
        model.save(model_path)
        print(f"💾 LSTM model saved to {model_path}")
        
        # Store results
        self.models['LSTM'] = model
        self.results['LSTM'] = {
            'history': history.history,
            'model_path': model_path
        }
        
        return model, history
    
    def train_traditional_models(self, X_train, y_train, X_test, y_test):
        """
        Train traditional machine learning models for comparison.
        
        Why compare with traditional ML:
        - Sometimes simpler models work better
        - Faster to train and predict
        - More interpretable results
        - Good baseline to beat with neural networks
        """
        print("📊 Training traditional ML models...")
        
        # Flatten sequences for traditional ML (they don't handle sequences)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Scale features for traditional ML
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Model 1: Random Forest
        print("  🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,  # 100 trees
            max_depth=10,      # Prevent overfitting
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Model 2: Logistic Regression
        print("  📈 Training Logistic Regression...")
        lr_model = LogisticRegression(
            max_iter=1000,     # Ensure convergence
            random_state=42
        )
        lr_model.fit(X_train_scaled, y_train)
        
        # Save traditional models
        with open(f"{self.model_path}random_forest.pkl", 'wb') as f:
            pickle.dump((rf_model, scaler), f)
        
        with open(f"{self.model_path}logistic_regression.pkl", 'wb') as f:
            pickle.dump((lr_model, scaler), f)
        
        # Store results
        self.models['Random Forest'] = (rf_model, scaler)
        self.models['Logistic Regression'] = (lr_model, scaler)
        
        print("✅ Traditional models trained and saved")
        return rf_model, lr_model, scaler
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and compare performance.
        
        Metrics we track:
        - Accuracy: Overall correctness
        - Precision: Of predicted UPs, how many were correct
        - Recall: Of actual UPs, how many did we catch
        - F1-score: Balance between precision and recall
        """
        print("📊 Evaluating all models...")
        
        results_summary = {}
        
        # Evaluate LSTM
        if 'LSTM' in self.models:
            print("  🧠 Evaluating LSTM...")
            lstm_model = self.models['LSTM']
            lstm_pred = lstm_model.predict(X_test)
            lstm_pred_binary = (lstm_pred > 0.5).astype(int).flatten()
            
            results_summary['LSTM'] = {
                'accuracy': accuracy_score(y_test, lstm_pred_binary),
                'classification_report': classification_report(y_test, lstm_pred_binary),
                'predictions': lstm_pred_binary
            }
        
        # Evaluate traditional models
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        for model_name in ['Random Forest', 'Logistic Regression']:
            if model_name in self.models:
                print(f"  📊 Evaluating {model_name}...")
                model, scaler = self.models[model_name]
                X_test_scaled = scaler.transform(X_test_flat)
                pred = model.predict(X_test_scaled)
                
                results_summary[model_name] = {
                    'accuracy': accuracy_score(y_test, pred),
                    'classification_report': classification_report(y_test, pred),
                    'predictions': pred
                }
        
        # Print comparison
        print(f"\n📈 MODEL COMPARISON:")
        print("=" * 40)
        for model_name, metrics in results_summary.items():
            accuracy = metrics['accuracy']
            print(f"{model_name:<20}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Find best model
        best_model = max(results_summary.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} ({best_model[1]['accuracy']*100:.2f}%)")
        
        return results_summary
    
    def plot_training_history(self):
        """
        Plot training history for neural networks.
        
        This helps us understand:
        - Is the model learning?
        - Is it overfitting?
        - Should we train longer or stop earlier?
        """
        if 'LSTM' not in self.results:
            print("No LSTM training history to plot")
            return
        
        history = self.results['LSTM']['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss', color='blue')
        ax2.plot(history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Model Loss Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = f"{self.results_path}training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training plots saved to {plot_path}")

    def create_confusion_matrices(self, results_summary, y_test):
        """
        Create confusion matrices for all models.
        
        Confusion matrix shows:
        - True Positives: Correctly predicted UP
        - True Negatives: Correctly predicted DOWN  
        - False Positives: Predicted UP but was DOWN
        - False Negatives: Predicted DOWN but was UP
        """
        n_models = len(results_summary)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(results_summary.items()):
            y_pred = results['predictions']
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['DOWN', 'UP'],
                       yticklabels=['DOWN', 'UP'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plot_path = f"{self.results_path}confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Confusion matrices saved to {plot_path}")

    def predict_next_day(self, symbol, model_name='LSTM'):
        """
        Predict tomorrow's direction for a specific stock.
        
        This function:
        1. Gets the latest data for the stock
        2. Prepares it in the same format as training
        3. Makes a prediction using the trained model
        4. Returns probability and direction
        """
        print(f"🔮 Predicting next day for {symbol} using {model_name}...")
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found!")
            return None
        
        try:
            # Get fresh data
            import yfinance as yf
            stock = yf.Ticker(symbol)
            recent_data = stock.history(period='3mo')  # Get 3 months of recent data
            
            if len(recent_data) < SEQUENCE_LENGTH:
                print(f"❌ Not enough recent data for {symbol}")
                return None
            
            # Prepare features (same as training)
            processed_data = self.add_technical_indicators(recent_data.copy())
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
            
            X = processed_data[feature_columns].dropna()
            
            if len(X) < SEQUENCE_LENGTH:
                print(f"❌ Not enough processed data for {symbol}")
                return None
            
            # Take the last SEQUENCE_LENGTH days
            X_recent = X.iloc[-SEQUENCE_LENGTH:].values
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, SEQUENCE_LENGTH, -1)
            
            # Make prediction
            if model_name == 'LSTM':
                model = self.models[model_name]
                prediction = model.predict(X_sequence)[0][0]
                direction = "UP" if prediction > 0.5 else "DOWN"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            else:
                model, scaler = self.models[model_name]
                X_flat = X_sequence.reshape(1, -1)
                prediction = model.predict_proba(X_flat)[0][1]  # Probability of UP
                direction = "UP" if prediction > 0.5 else "DOWN"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            
            result = {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'probability': prediction,
                'current_price': recent_data['Close'].iloc[-1]
            }
            
            print(f"  📈 {symbol}: {direction} ({confidence*100:.1f}% confidence)")
            return result
            
        except Exception as e:
            print(f"❌ Error predicting for {symbol}: {e}")
            return None
        
    def add_technical_indicators(self, data):
        """Add technical indicators to new data (same as training)."""
        # SMA
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
        
        # Bollinger Bands
        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        data['BB_upper'] = sma20 + (2 * std20)
        data['BB_lower'] = sma20 - (2 * std20)
        
        return data
    
    
