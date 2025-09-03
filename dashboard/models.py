import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for stock price prediction.
    
    FIXED: Model now outputs probabilities directly for consistency and handles class imbalance better.
    """
    def __init__(self, input_size, hidden_size=50, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        # LSTM layers with improved configuration
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False  # Keep unidirectional for time series
        )
        
        # Batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer with higher rate
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers with batch norm
        self.fc1 = nn.Linear(hidden_size, 25)
        self.bn1 = nn.BatchNorm1d(25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        
        # FIXED: Add sigmoid for probability output during inference
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights properly (Xavier initialization)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization for better gradients."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with vanishing gradients)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
        
        # Initialize linear layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        
    def forward(self, x, return_logits=False):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Apply batch normalization (only during training with batch size > 1)
        if self.training and last_output.size(0) > 1:
            last_output = self.batch_norm(last_output)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Dense layers with batch norm
        out = self.fc1(out)
        if self.training and out.size(0) > 1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        # FIXED: Return logits for training (BCEWithLogitsLoss), probabilities for inference
        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)

class GRUModel(nn.Module):
    """
    PyTorch GRU model for stock price prediction.
    
    FIXED: Model now outputs probabilities directly for consistency.
    """
    def __init__(self, input_size, hidden_size=50, num_layers=3, dropout=0.2):
        super(GRUModel, self).__init__()
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        # FIXED: Add sigmoid for probability output during inference
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_logits=False):
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Take the last output from the sequence
        last_output = gru_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        # FIXED: Return logits for training (BCEWithLogitsLoss), probabilities for inference
        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)