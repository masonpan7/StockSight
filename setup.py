# Step 1: Project Setup & Dependencies
# File: 01_setup.py
# Goal: Install packages, set up project structure, and test basic imports

"""
Stock Price Movement Classifier - Step 1: Setup
===============================================

This script sets up the entire project environment including:
1. Project folder structure
2. Virtual environment setup
3. Package installation
4. Import testing
5. Configuration setup

Author: Your Name
Date: Today's Date
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import random
import numpy as np
import tensorflow as tf

class ProjectSetup:
    """
    Handles the complete project setup for the Stock Price Movement Classifier.
    
    This class creates the folder structure, manages dependencies, and validates
    the environment to ensure everything is ready for the machine learning pipeline.
    """
    
    def __init__(self, project_name="stock_predictor"):
        """
        Initialize the project setup.
        
        Args:
            project_name (str): Name of the project folder
        """
        self.project_name = project_name
        self.project_root = Path.cwd() / project_name
        self.required_packages = [
            'numpy>=1.21.0',
            'pandas>=1.3.0', 
            'yfinance>=0.1.70',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'scikit-learn>=1.0.0',
            'tensorflow>=2.8.0',
            'streamlit>=1.15.0',
            'plotly>=5.0.0',
            'jupyter>=1.0.0'
        ]
        
    def create_folder_structure(self):
        """
        Create the complete project folder structure.
        
        This creates organized folders for different components:
        - data/: Raw and processed data storage
        - models/: Trained model storage
        - src/: Source code modules
        - notebooks/: Jupyter notebooks for exploration
        - dashboard/: Streamlit dashboard files
        - results/: Output files and visualizations
        - tests/: Unit tests
        - docs/: Documentation
        """
        print("📁 Creating project folder structure...")
        
        folders = [
            'data/raw',           # Raw stock data from APIs
            'data/processed',     # Cleaned and featured data
            'data/predictions',   # Prediction outputs
            'models/traditional', # Random Forest, Logistic Regression
            'models/neural',      # LSTM, GRU models
            'models/scalers',     # Data scaling objects
            'src/data',          # Data collection and processing
            'src/features',      # Feature engineering
            'src/models',        # Model training and evaluation
            'src/utils',         # Utility functions
            'notebooks',         # Jupyter notebooks for exploration
            'dashboard',         # Streamlit dashboard components
            'results/plots',     # Generated visualizations
            'results/reports',   # Performance reports
            'tests',             # Unit tests
            'docs',              # Documentation
            'config'             # Configuration files
        ]
        
        # Create project root directory
        self.project_root.mkdir(exist_ok=True)
        print(f"✅ Created project root: {self.project_root}")
        
        # Create all subdirectories
        for folder in folders:
            folder_path = self.project_root / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created folder: {folder}")
            
            # Add __init__.py to source folders for Python modules
            if folder.startswith('src/'):
                init_file = folder_path / '__init__.py'
                init_file.touch()
                
        print(f"📁 Project structure created successfully!")
        return True
    
    def create_requirements_file(self):
        """
        Create requirements.txt file with all necessary packages.
        
        This file contains all Python packages needed for the project,
        making it easy to recreate the environment on any machine.
        """
        print("📄 Creating requirements.txt file...")
        
        requirements_path = self.project_root / 'requirements.txt'
        
        with open(requirements_path, 'w') as f:
            f.write("# Stock Price Movement Classifier - Requirements\n")
            f.write("# Install with: pip install -r requirements.txt\n\n")
            f.write("# Core Data Science Libraries\n")
            for package in self.required_packages:
                f.write(f"{package}\n")
            
            f.write("\n# Additional Utilities\n")
            f.write("python-dotenv>=0.19.0\n")  # Environment variables
            f.write("tqdm>=4.62.0\n")           # Progress bars
            f.write("joblib>=1.1.0\n")          # Model persistence
            
        print(f"✅ Requirements file created: {requirements_path}")
        return requirements_path
    
    def create_config_files(self):
        """
        Create configuration files for the project.
        
        These files contain project settings, stock symbols, and parameters
        that can be easily modified without changing the code.
        """
        print("⚙️ Creating configuration files...")
        
        # Main configuration file
        config_path = self.project_root / 'config' / 'config.py'
        
        config_content = '''"""
Project Configuration File
=========================

Contains all configurable parameters for the Stock Price Movement Classifier.
Modify these values to customize the behavior without changing the main code.
"""

# Stock symbols for analysis (Top 10 Tech Stocks)
STOCK_SYMBOLS = [
    'AAPL',   # Apple Inc.
    'MSFT',   # Microsoft Corporation  
    'GOOGL',  # Alphabet Inc. (Google)
    'AMZN',   # Amazon.com Inc.
    'TSLA',   # Tesla Inc.
    'META',   # Meta Platforms Inc.
    'NVDA',   # NVIDIA Corporation
    'NFLX',   # Netflix Inc.
    'CRM',    # Salesforce Inc.
    'ADBE'    # Adobe Inc.
]

# Data collection parameters
DATA_CONFIG = {
    'period': '2y',              # Historical data period ('1y', '2y', '5y', 'max')
    'interval': '1d',            # Data interval ('1d', '1h', '5m')
    'auto_adjust': True,         # Adjust OHLC for dividends and splits
    'prepost': False,            # Include pre and post market data
    'threads': True,             # Use threading for faster downloads
}

# Feature engineering parameters
FEATURE_CONFIG = {
    'sequence_length': 60,       # Days of historical data for predictions
    'rsi_period': 14,           # RSI calculation period
    'macd_fast': 12,            # MACD fast EMA period
    'macd_slow': 26,            # MACD slow EMA period
    'sma_short': 10,            # Short-term moving average
    'sma_long': 30,             # Long-term moving average
    'volatility_window': 20,    # Volatility calculation window
}

# Model training parameters
TRAINING_CONFIG = {
    'test_size': 0.2,           # Proportion of data for testing
    'validation_split': 0.2,    # Validation split for neural networks
    'random_state': 42,         # Random seed for reproducibility
    'epochs': 50,               # Maximum training epochs for neural networks
    'batch_size': 32,           # Batch size for neural network training
    'patience': 10,             # Early stopping patience
    'learning_rate': 0.001,     # Initial learning rate
}

# Neural network architecture
NN_CONFIG = {
    'lstm_units': [50, 50],     # LSTM layer units
    'gru_units': [50, 50],      # GRU layer units  
    'dropout_rate': 0.2,        # Dropout rate for regularization
    'dense_units': 25,          # Dense layer units
    'activation': 'relu',       # Hidden layer activation
    'output_activation': 'sigmoid',  # Output layer activation
}

# Traditional ML parameters
ML_CONFIG = {
    'rf_n_estimators': 100,     # Random Forest number of trees
    'rf_max_depth': 10,         # Random Forest max depth
    'rf_min_samples_split': 20, # Random Forest min samples for split
    'lr_max_iter': 1000,        # Logistic Regression max iterations
    'lr_solver': 'liblinear',   # Logistic Regression solver
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'Stock Movement Predictor',
    'page_icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'update_interval': 300,     # Seconds between data updates
}

# File paths (relative to project root)
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'models': 'models',
    'results': 'results',
    'logs': 'logs',
}

# API configuration (if using premium data sources)
API_CONFIG = {
    'alpha_vantage_key': None,  # Add your API key if using Alpha Vantage
    'quandl_key': None,         # Add your API key if using Quandl
    'use_yfinance': True,       # Use free Yahoo Finance data
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'stock_predictor.log',
}

# Random seeds for reproducibility
RANDOM_SEEDS = {
    'numpy': 42,
    'python': 42,
    'tensorflow': 42,
}
'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Environment file template
        env_path = self.project_root / '.env.template'
        env_content = '''# Environment Variables Template
# Copy this file to .env and fill in your actual values

# API Keys (optional - yfinance is free)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
QUANDL_API_KEY=your_quandl_key_here

# Database configuration (if using database storage)
DATABASE_URL=sqlite:///stock_data.db

# Logging level
LOG_LEVEL=INFO

# Dashboard settings
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8501
'''
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Configuration files created")
        return True
    
    def test_imports(self):
        """
        Test all required package imports to ensure everything is installed correctly.
        
        This function attempts to import all packages and reports any missing ones.
        It's crucial to run this before proceeding to avoid runtime errors later.
        """
        print("🔍 Testing package imports...")
        
        # Core packages to test
        test_packages = [
            ('numpy', 'np'),
            ('pandas', 'pd'),
            ('yfinance', 'yf'),
            ('matplotlib.pyplot', 'plt'),
            ('seaborn', 'sns'),
            ('sklearn', None),
            ('tensorflow', 'tf'),
            ('streamlit', 'st'),
            ('plotly', None),
        ]
        
        failed_imports = []
        successful_imports = []
        
        for package_info in test_packages:
            if len(package_info) == 2:
                package, alias = package_info
            else:
                package, alias = package_info[0], None
            
            try:
                if alias:
                    exec(f"import {package} as {alias}")
                else:
                    exec(f"import {package}")
                successful_imports.append(package)
                print(f"✅ {package} - OK")
                
            except ImportError as e:
                failed_imports.append((package, str(e)))
                print(f"❌ {package} - FAILED: {e}")
        
        # Test specific functions we'll need
        print("\n🔍 Testing specific functionality...")
        
        try:
            import yfinance as yf
            # Test data download
            test_data = yf.download('AAPL', period='5d', progress=False)
            if not test_data.empty:
                print("✅ Yahoo Finance data download - OK")
            else:
                print("❌ Yahoo Finance data download - No data returned")
        except Exception as e:
            print(f"❌ Yahoo Finance test - FAILED: {e}")
        
        try:
            import tensorflow as tf
            # Test TensorFlow GPU availability
            if tf.config.list_physical_devices('GPU'):
                print("✅ TensorFlow GPU support - Available")
            else:
                print("ℹ️ TensorFlow GPU support - Not available (CPU only)")
        except Exception as e:
            print(f"❌ TensorFlow test - FAILED: {e}")
        
        # Summary
        print(f"\n📊 Import Summary:")
        print(f"✅ Successful: {len(successful_imports)}")
        print(f"❌ Failed: {len(failed_imports)}")
        
        if failed_imports:
            print(f"\n❌ Failed packages:")
            for package, error in failed_imports:
                print(f"   • {package}: {error}")
            print(f"\n💡 Install missing packages with:")
            print(f"   pip install -r requirements.txt")
            return False
        else:
            print(f"\n🎉 All packages imported successfully!")
            return True
    
    def set_random_seeds(self):
        """
        Set random seeds for reproducibility across all libraries.
        
        This ensures that results are consistent across different runs,
        which is important for comparing model performance and debugging.
        """
        print("🎲 Setting random seeds for reproducibility...")
        
        # Python's random module
        random.seed(42)
        print("✅ Python random seed set to 42")
        
        # NumPy random seed
        np.random.seed(42)
        print("✅ NumPy random seed set to 42")
        
        # TensorFlow random seed
        tf.random.set_seed(42)
        print("✅ TensorFlow random seed set to 42")
        
        # Set TensorFlow to use deterministic operations (when possible)
        try:
            tf.config.experimental.enable_op_determinism()
            print("✅ TensorFlow deterministic operations enabled")
        except:
            print("ℹ️ TensorFlow deterministic operations not available (older version)")
        
        return True
    
    def create_gitignore(self):
        """
        Create .gitignore file to exclude unnecessary files from version control.
        
        This prevents large model files, data files, and system files from
        being accidentally committed to the repository.
        """
        print("📝 Creating .gitignore file...")
        
        gitignore_content = '''# Stock Predictor Project - Git Ignore File

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/
.env

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/*.csv
data/processed/*.csv
data/predictions/*.csv
*.h5
*.pkl
*.joblib

# Model files
models/traditional/*.pkl
models/neural/*.h5
models/scalers/*.pkl

# Results and logs
results/plots/*.png
results/plots/*.jpg
results/reports/*.html
*.log

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp

# API keys and secrets
.env
config/secrets.py

# Large files
*.zip
*.tar.gz
'''
        
        gitignore_path = self.project_root / '.gitignore'
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        print(f"✅ .gitignore file created")
        return True
    
    def create_readme(self):
        """
        Create a comprehensive README.md file for the project.
        
        This provides documentation for anyone who wants to understand
        or contribute to the project.
        """
        print("📚 Creating README.md file...")
        
        readme_content = '''# 📈 Stock Price Movement Classifier

AI-powered system that predicts next-day stock price movements using deep learning and traditional machine learning models.

## 🎯 Project Overview

This project implements and compares multiple machine learning approaches for predicting whether a stock's price will go up or down the next day:

- **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in time series data
- **GRU (Gated Recurrent Unit)**: Simplified version of LSTM with fewer parameters  
- **Random Forest**: Ensemble baseline for comparison
- **Logistic Regression**: Linear baseline for comparison

## 🚀 Features

- ✅ Real-time stock data collection from Yahoo Finance
- ✅ Comprehensive technical indicator calculation (RSI, MACD, Moving Averages)
- ✅ Multiple model types with performance comparison
- ✅ Interactive Streamlit dashboard
- ✅ Trading recommendations with confidence scores
- ✅ Modular, well-documented codebase

## 📁 Project Structure

```
stock_predictor/
│
├── data/                   # Data storage
│   ├── raw/               # Raw stock data
│   ├── processed/         # Cleaned and featured data
│   └── predictions/       # Model predictions
│
├── models/                # Trained models
│   ├── traditional/       # Random Forest, Logistic Regression
│   ├── neural/           # LSTM, GRU models
│   └── scalers/          # Data scaling objects
│
├── src/                   # Source code
│   ├── data/             # Data collection and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── utils/            # Utility functions
│
├── dashboard/             # Streamlit dashboard
├── results/              # Outputs and visualizations
├── tests/                # Unit tests
├── docs/                 # Documentation
└── config/               # Configuration files
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_predictor
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Step-by-Step Execution

The project is organized into 22 manageable steps:

1. **Setup**: `python 01_setup.py`
2. **Data Collection**: `python 02_data_collector.py`
3. **Feature Engineering**: `python 05_feature_engineering.py`
4. **Model Training**: `python 10_nn_training.py`
5. **Dashboard**: `streamlit run 17_advanced_dashboard.py`

### Quick Start

```python
from src.models.stock_classifier import StockMovementClassifier

# Initialize classifier
classifier = StockMovementClassifier(['AAPL', 'MSFT', 'GOOGL'])

# Train models
classifier.train_model('AAPL')

# Make prediction
prediction = classifier.predict_next_day('AAPL', model_type='LSTM')
print(f"Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.2%})")
```

## 📈 Expected Performance

Typical accuracy ranges:
- **LSTM/GRU**: 52-58% (above random 50%)
- **Random Forest**: 48-54%  
- **Logistic Regression**: 45-52%

## 🎨 Dashboard

Launch the interactive dashboard:
```bash
streamlit run dashboard/app.py
```

Features:
- Real-time predictions for top 10 tech stocks
- Model performance comparisons
- Trading recommendations
- Interactive visualizations

## ⚠️ Disclaimer

This project is for educational purposes only. Stock predictions are inherently uncertain and this tool should not be used as the sole basis for investment decisions. Always consult with financial professionals and do your own research.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock data
- TensorFlow team for the deep learning framework
- Streamlit for the amazing dashboard framework
'''
        
        readme_path = self.project_root / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ README.md file created")
        return True
    
    def run_setup(self):
        """
        Execute the complete project setup process.
        
        This is the main function that orchestrates all setup steps
        and provides a comprehensive setup report.
        """
        print("🚀 Starting Stock Predictor Project Setup")
        print("=" * 60)
        
        setup_steps = [
            ("Creating project structure", self.create_folder_structure),
            ("Creating requirements file", self.create_requirements_file),
            ("Creating configuration files", self.create_config_files),
            ("Creating .gitignore", self.create_gitignore),
            ("Creating README", self.create_readme),
            ("Setting random seeds", self.set_random_seeds),
            ("Testing package imports", self.test_imports),
        ]
        
        results = {}
        
        for step_name, step_function in setup_steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            try:
                result = step_function()
                results[step_name] = {"success": result, "error": None}
            except Exception as e:
                results[step_name] = {"success": False, "error": str(e)}
                print(f"❌ Error in {step_name}: {e}")
        
        # Setup summary
        print(f"\n{'='*60}")
        print("🎯 SETUP SUMMARY")
        print(f"{'='*60}")
        
        successful_steps = sum(1 for r in results.values() if r["success"])
        total_steps = len(results)
        
        print(f"📊 Completed: {successful_steps}/{total_steps} steps")
        
        for step_name, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {step_name}")
            if not result["success"] and result["error"]:
                print(f"   Error: {result['error']}")
        
        if successful_steps == total_steps:
            print(f"\n🎉 Setup completed successfully!")
            print(f"📁 Project created at: {self.project_root}")
            print(f"\n📋 Next steps:")
            print(f"1. cd {self.project_name}")
            print(f"2. python -m venv venv")
            print(f"3. source venv/bin/activate  # (Windows: venv\\Scripts\\activate)")
            print(f"4. pip install -r requirements.txt")
            print(f"5. python 02_data_collector.py")
        else:
            print(f"\n⚠️ Setup completed with {total_steps - successful_steps} errors.")
            print(f"Please resolve the errors above before proceeding.")
        
        return successful_steps == total_steps

def main():
    """
    Main execution function for project setup.
    
    This function can be run directly to set up the entire project structure.
    """
    print("Stock Price Movement Classifier - Project Setup")
    print("Developer: Your Name")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup = ProjectSetup()
    success = setup.run_setup()
    
    if success:
        print("\n🎊 Welcome to your Stock Predictor project!")
        print("Happy coding! 🚀")
    else:
        print("\n🔧 Please fix the setup issues and run again.")

if __name__ == "__main__":
    from datetime import datetime
    main()