import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# Set device for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

class LSTMModel(nn.Module):
    """PyTorch LSTM model for stock price prediction."""
    
    def __init__(self, input_size, hidden_size=50, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.bn1 = nn.BatchNorm1d(25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, x, return_logits=False):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        if self.training and last_output.size(0) > 1:
            last_output = self.batch_norm(last_output)
        
        out = self.dropout(last_output)
        out = self.fc1(out)
        if self.training and out.size(0) > 1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)

class GRUModel(nn.Module):
    """PyTorch GRU model for stock price prediction."""
    
    def __init__(self, input_size, hidden_size=50, num_layers=3, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_logits=False):
        gru_out, hidden = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)

class StockPredictor:
    """Complete stock movement prediction system using PyTorch."""
    
    def __init__(self):
        """Initialize the predictor with paths and scalers."""
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.results_path = RESULTS_PATH
        
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = StandardScaler()
        
        self.models = {}
        self.results = {}
        self.device = device

    def load_data(self):
        """Load the preprocessed stock data."""
        print("Loading stock data...")
        
        data_file = f"{self.data_path}all_stocks_combined.csv"
        if not os.path.exists(data_file):
            print("No data found! Run data_collector.py first.")
            return None
        
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data):,} samples from {len(data['Symbol'].unique())} stocks")
        
        return data
    
    def prepare_features(self, data):
        """Select and prepare features for machine learning."""
        print("Preparing features...")
        
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_upper', 'BB_lower', 'Volume_MA',
            'Price_Change', 'High_Low_Pct'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        X = data[available_features].copy()
        y = data['Target'].copy()
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        print(f"Prepared {len(X)} samples with {X.shape[1]} features")
        print(f"Target distribution: UP={sum(y)}, DOWN={len(y)-sum(y)}")
        
        return X, y, available_features
    
    def create_sequences(self, X, y, sequence_length=SEQUENCE_LENGTH):
        """Create sequences for LSTM training."""
        print(f"Creating sequences of length {sequence_length}...")
        
        X_sequences = []
        y_sequences = []
        
        if 'Symbol' in X.columns:
            symbols = X['Symbol'].unique()
            X_no_symbol = X.drop('Symbol', axis=1)
        else:
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
            
            for i in range(sequence_length, len(symbol_data)):
                sequence_X = symbol_data.iloc[i-sequence_length:i].values
                sequence_y = symbol_targets.iloc[i]
                
                X_sequences.append(sequence_X)
                y_sequences.append(sequence_y)
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        print(f"Created {len(X_seq)} sequences")
        print(f"Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def train_pytorch_model(self, X_train, y_train, X_test, y_test, model_type='LSTM'):
        """Train PyTorch model (LSTM or GRU)."""
        print(f"Training {model_type} model...")
        
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"Training class distribution: {class_distribution}")
        
        if len(counts) > 1 and 1 in class_distribution and 0 in class_distribution:
            pos_weight = torch.FloatTensor([class_distribution[0] / class_distribution[1]]).to(self.device)
        else:
            pos_weight = torch.FloatTensor([1.0]).to(self.device)
        print(f"Positive class weight: {pos_weight.item():.3f}")
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        input_size = X_train.shape[2]
        if model_type == 'LSTM':
            model = LSTMModel(input_size).to(self.device)
        else:
            model = GRUModel(input_size).to(self.device)
        
        print(f"Model architecture:")
        print(f"   Input size: {input_size}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print(f"Starting training for {EPOCHS} epochs...")
        
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_predictions = []
            train_targets = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                
                logits = model(batch_X, return_logits=True)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                train_predictions.extend(predicted.cpu().numpy().flatten())
                train_targets.extend(batch_y.cpu().numpy().flatten())
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    logits = model(batch_X, return_logits=True)
                    loss = criterion(logits, batch_y)
                    
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits)
                    predicted = (probs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    val_predictions.extend(predicted.cpu().numpy().flatten())
                    val_targets.extend(batch_y.cpu().numpy().flatten())
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            val_loss /= len(test_loader)
            val_acc = val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            train_pred_dist = np.bincount(np.array(train_predictions, dtype=int))
            val_pred_dist = np.bincount(np.array(val_predictions, dtype=int))
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{self.model_path}{model_type.lower()}_best.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1:3d}/{EPOCHS}] - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'   Train Pred Dist: {train_pred_dist}, Val Pred Dist: {val_pred_dist}')
                print(f'   LR: {optimizer.param_groups[0]["lr"]:.2e}, Patience: {patience_counter}/{patience}')
        
        model.load_state_dict(torch.load(f"{self.model_path}{model_type.lower()}_best.pth"))
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'input_size': input_size,
            'history': history
        }, f"{self.model_path}{model_type.lower()}_model.pth")
        
        print(f"{model_type} model saved")
        
        self.models[model_type] = model
        self.results[model_type] = {
            'history': history,
            'model_path': f"{self.model_path}{model_type.lower()}_model.pth"
        }
        
        return model, history
    
    def train_traditional_models(self, X_train, y_train, X_test, y_test):
        """Train traditional machine learning models for comparison."""
        print("Training traditional ML models...")
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        lr_model.fit(X_train_scaled, y_train)
        
        with open(f"{self.model_path}random_forest.pkl", 'wb') as f:
            pickle.dump((rf_model, scaler), f)
        
        with open(f"{self.model_path}logistic_regression.pkl", 'wb') as f:
            pickle.dump((lr_model, scaler), f)
        
        self.models['Random Forest'] = (rf_model, scaler)
        self.models['Logistic Regression'] = (lr_model, scaler)
        
        print("Traditional models trained and saved")
        return rf_model, lr_model, scaler
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models and compare performance."""
        print("Evaluating all models...")
        
        results_summary = {}
        
        for model_type in ['LSTM', 'GRU']:
            if model_type in self.models:
                print(f"Evaluating {model_type}...")
                model = self.models[model_type]
                model.eval()
                
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                with torch.no_grad():
                    probabilities = model(X_test_tensor, return_logits=False).cpu().numpy()
                
                pred_binary = (probabilities > 0.5).astype(int).flatten()
                
                print(f"   Sample probabilities: {probabilities[:5].flatten()}")
                print(f"   Sample predictions: {pred_binary[:5]}")
                print(f"   Prediction distribution: UP={np.sum(pred_binary)}, DOWN={len(pred_binary) - np.sum(pred_binary)}")
                
                results_summary[model_type] = {
                    'accuracy': accuracy_score(y_test, pred_binary),
                    'classification_report': classification_report(y_test, pred_binary),
                    'predictions': pred_binary
                }
        
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        for model_name in ['Random Forest', 'Logistic Regression']:
            if model_name in self.models:
                print(f"Evaluating {model_name}...")
                model, scaler = self.models[model_name]
                X_test_scaled = scaler.transform(X_test_flat)
                pred = model.predict(X_test_scaled)
                
                results_summary[model_name] = {
                    'accuracy': accuracy_score(y_test, pred),
                    'classification_report': classification_report(y_test, pred),
                    'predictions': pred
                }
        
        print(f"\nMODEL COMPARISON:")
        print("=" * 40)
        for model_name, metrics in results_summary.items():
            accuracy = metrics['accuracy']
            print(f"{model_name:<20}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if results_summary:
            best_model = max(results_summary.items(), key=lambda x: x[1]['accuracy'])
            print(f"\nBest Model: {best_model[0]} ({best_model[1]['accuracy']*100:.2f}%)")
        
        return results_summary
    
    def plot_training_history(self):
        """Plot training history for neural networks."""
        pytorch_models = ['LSTM', 'GRU']
        available_models = [m for m in pytorch_models if m in self.results]
        
        if not available_models:
            print("No PyTorch training history to plot")
            return
        
        fig, axes = plt.subplots(len(available_models), 2, figsize=(15, 5*len(available_models)))
        if len(available_models) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, model_name in enumerate(available_models):
            history = self.results[model_name]['history']
            
            # Plot accuracy
            axes[idx, 0].plot(history['train_acc'], label='Training Accuracy', color='blue')
            axes[idx, 0].plot(history['val_acc'], label='Validation Accuracy', color='red')
            axes[idx, 0].set_title(f'{model_name} - Accuracy Over Time')
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Accuracy')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True)
            
            # Plot loss
            axes[idx, 1].plot(history['train_loss'], label='Training Loss', color='blue')
            axes[idx, 1].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[idx, 1].set_title(f'{model_name} - Loss Over Time')
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('Loss')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True)
        
        plt.tight_layout()
        plot_path = f"{self.results_path}training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training plots saved to {plot_path}")

    def create_confusion_matrices(self, results_summary, y_test):
        """Create confusion matrices for all models."""
        n_models = len(results_summary)
        if n_models == 0:
            print("No models to create confusion matrices for")
            return
            
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(results_summary.items()):
            y_pred = results['predictions']
            
            cm = confusion_matrix(y_test, y_pred)
            
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
        print(f"Confusion matrices saved to {plot_path}")

    def predict_next_day(self, symbol, model_name='LSTM'):
        """Predict tomorrow's direction for a specific stock."""
        print(f"Predicting next day for {symbol} using {model_name}...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            recent_data = stock.history(period='3mo')
            
            if len(recent_data) < SEQUENCE_LENGTH:
                print(f"Not enough recent data for {symbol}")
                return None
            
            processed_data = self.add_technical_indicators(recent_data.copy())
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
            
            X = processed_data[feature_columns].dropna()
            
            if len(X) < SEQUENCE_LENGTH:
                print(f"Not enough processed data for {symbol}")
                return None
            
            X_recent = X.iloc[-SEQUENCE_LENGTH:].values
            X_scaled = self.feature_scaler.transform(X_recent)
            X_sequence = X_scaled.reshape(1, SEQUENCE_LENGTH, -1)
            
            if model_name in ['LSTM', 'GRU']:
                model = self.models[model_name]
                model.eval()
                X_tensor = torch.FloatTensor(X_sequence).to(self.device)
                with torch.no_grad():
                    prediction = model(X_tensor, return_logits=False).cpu().numpy()[0][0]
                
                print(f"  Raw model output (probability): {prediction}")
                
                direction = "UP" if prediction > 0.5 else "DOWN"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            else:
                model, scaler = self.models[model_name]
                X_flat = X_sequence.reshape(1, -1)
                X_scaled_flat = scaler.transform(X_flat)
                prediction = model.predict_proba(X_scaled_flat)[0][1]
                direction = "UP" if prediction > 0.5 else "DOWN"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            
            result = {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'probability': prediction,
                'current_price': recent_data['Close'].iloc[-1]
            }
            
            print(f"  {symbol}: {direction} ({confidence*100:.1f}% confidence)")
            return result
            
        except Exception as e:
            print(f"Error predicting for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def add_technical_indicators(self, data):
        """Add technical indicators to new data (same as training)."""
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        delta = data['Close'].diff()
        gains = delta.where(delta > 0, 0).rolling(14).mean()
        losses = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gains / losses
        data['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = data['Close'].ewm(span=12).mean()
        exp26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp12 - exp26
        
        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        data['BB_upper'] = sma20 + (2 * std20)
        data['BB_lower'] = sma20 - (2 * std20)
        
        return data
    
    def run_full_pipeline(self):
        """Run the complete model training pipeline."""
        print("STARTING FULL MODEL TRAINING PIPELINE (PyTorch - FIXED)")
        print("=" * 60)
        
        data = self.load_data()
        if data is None:
            return
        
        X, y, feature_names = self.prepare_features(data)
        X_seq, y_seq = self.create_sequences(X, y)
        
        split_idx = int(len(X_seq) * TRAIN_SPLIT)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        self.feature_scaler.fit(X_train_reshaped)
        X_train_scaled = self.feature_scaler.transform(X_train_reshaped)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped)
        
        X_train = X_train_scaled.reshape(X_train.shape)
        X_test = X_test_scaled.reshape(X_test.shape)
        
        lstm_model, lstm_history = self.train_pytorch_model(X_train, y_train, X_test, y_test, 'LSTM')
        gru_model, gru_history = self.train_pytorch_model(X_train, y_train, X_test, y_test, 'GRU')
        
        rf_model, lr_model, scaler = self.train_traditional_models(X_train, y_train, X_test, y_test)
        
        results = self.evaluate_models(X_test, y_test)
        
        self.plot_training_history()
        self.create_confusion_matrices(results, y_test)
        
        with open(f"{self.model_path}feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        print(f"\nPIPELINE COMPLETE!")
        print(f"All models saved to: {self.model_path}")
        print(f"All results saved to: {self.results_path}")
        
        return results

def main():
    """Main function to run model training."""
    predictor = StockPredictor()
    results = predictor.run_full_pipeline()
    
    if results:
        print(f"\nFINAL RESULTS:")
        print("=" * 30)
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['accuracy']*100:.2f}% accuracy")
        
        print(f"\nNext steps:")
        print(f"1. Check plots in {predictor.results_path}")
        print(f"2. Run predictions: predictor.predict_next_day('AAPL')")
        print(f"3. Create dashboard: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()