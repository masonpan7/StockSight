# project_setup.py
"""
Stock Price Movement Classifier - Project Setup
This script creates the project structure and installs dependencies automatically.
"""

import os
import subprocess
import sys
import platform

def create_project_structure():
    """
    Create the project directory structure.
    We organize code into logical folders for better maintainability.
    """
    print("📁 Creating project structure...")
    
    # Define our project structure
    folders = [
        'data',           # Store downloaded stock data
        'models',         # Save trained models
        'notebooks',      # Jupyter notebooks for exploration
        'src',           # Source code
        'dashboard',     # Streamlit dashboard files
        'results',       # Model results and plots
    ]
    
    # Create each folder
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✅ Created: {folder}/")
    
    print("📁 Project structure created!\n")

def detect_system():
    """
    Detect the operating system and processor type.
    This helps us install the right TensorFlow version.
    """
    system = platform.system()
    machine = platform.machine()
    
    print(f"🖥️  System: {system}")
    print(f"🔧 Architecture: {machine}")
    
    # Determine Mac type for TensorFlow installation
    if system == "Darwin":  # macOS
        if machine == "arm64":
            return "apple_silicon"
        else:
            return "intel_mac"
    else:
        return "other"

def install_packages():
    """
    Install required packages based on system type.
    We use different TensorFlow versions for different systems to avoid AVX errors.
    """
    system_type = detect_system()
    print(f"\n📦 Installing packages for {system_type}...")
    
    # Base packages that work on all systems
    base_packages = [
        "pandas>=1.3.0",      # Data manipulation
        "numpy>=1.21.0",      # Numerical computing
        "yfinance>=0.1.70",   # Yahoo Finance API for stock data
        "matplotlib>=3.5.0",  # Basic plotting
        "seaborn>=0.11.0",    # Statistical visualization
        "scikit-learn>=1.0.0", # Traditional ML algorithms
        "streamlit>=1.15.0",  # Dashboard framework
        "plotly>=5.10.0",     # Interactive plots
        "jupyter>=1.0.0",     # Notebook environment
    ]
    
    # TensorFlow installation based on system
    if system_type == "apple_silicon":
        tf_packages = ["tensorflow-macos>=2.9.0", "tensorflow-metal>=0.5.0"]
        print("🍎 Installing Apple Silicon optimized TensorFlow...")
    elif system_type == "intel_mac":
        tf_packages = ["tensorflow>=2.10.0,<2.11.0"]
        print("🍎 Installing Intel Mac compatible TensorFlow...")
    else:
        tf_packages = ["tensorflow>=2.10.0"]
        print("💻 Installing standard TensorFlow...")
    
    # Combine all packages
    all_packages = base_packages + tf_packages
    
    # Install packages
    for package in all_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {package}")
            print(f"     Try: pip install {package}")
    
    print("\n📦 Package installation complete!")

def create_requirements_file():
    """
    Create a requirements.txt file for easy future setup.
    This allows others to recreate your environment.
    """
    print("\n📝 Creating requirements.txt...")
    
    requirements = """# Stock Price Movement Classifier Requirements
# Data manipulation and analysis
pandas>=1.3.0
numpy>=1.21.0

# Stock data
yfinance>=0.1.70

# Machine learning
tensorflow>=2.10.0  # Use tensorflow-macos for Apple Silicon
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Dashboard
streamlit>=1.15.0

# Development
jupyter>=1.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("  ✅ requirements.txt created")

def create_config_file():
    """
    Create a configuration file for the project.
    This centralizes all our settings and makes the project more maintainable.
    """
    print("\n⚙️  Creating config.py...")
    
    config_content = '''# config.py
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
LEARNING_RATE = 0.001    # How fast the model learns

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
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("  ✅ config.py created")

def test_installation():
    """
    Test if all packages are installed correctly.
    This helps catch any installation issues early.
    """
    print("\n🧪 Testing installation...")
    
    # Essential packages (must work)
    essential_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('yfinance', 'yf'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('sklearn', None),
        ('plotly', None),
    ]
    
    # Optional packages (nice to have, but not required for setup)
    '''
    optional_imports = [
        ('streamlit', 'st'),
        ('tensorflow', 'tf'),
    ]
    '''
    
    failed_essential = []
    failed_optional = []
    
    # Test essential packages
    print("📦 Testing essential packages...")
    for module, alias in essential_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {str(e)}")
            failed_essential.append(module)
    '''
    # Test optional packages
    print("\n🔧 Testing optional packages...")
    for module, alias in optional_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ⚠️  {module} - {str(e)}")
            print(f"     (Optional - can install later)")
            failed_optional.append(module)
    '''
    # Summary
    if failed_essential:
        print(f"\n❌ CRITICAL: Essential packages failed: {failed_essential}")
        print("   These are required! Try: pip install <package_name>")
        return False
    elif failed_optional:
        print(f"\n✅ Essential packages work! Optional failed: {failed_optional}")
        print("💡 You can continue and install optional packages later")
        return True
    else:
        print("\n🎉 All packages imported successfully!")
        return True

def main():
    """
    Main setup function that runs all setup steps.
    """
    print("🚀 Stock Price Movement Classifier - Project Setup")
    print("=" * 50)
    
    # Run all setup steps
    create_project_structure()
    install_packages()
    create_requirements_file()
    create_config_file()
    
    # Test installation and check if we can proceed
    setup_success = test_installation()
    
    print("\n" + "=" * 50)
    
    if setup_success:
        print("🎉 Project setup complete!")
        print("\nNext steps:")
        print("1. Run: python data_collector.py")
        print("2. Run: python model_trainer.py (requires TensorFlow)")
        print("3. Run: streamlit run dashboard/app.py")
        print("\n💡 Tip: You can start with data collection even if TensorFlow isn't working yet!")
        
        if 'tensorflow' in locals():
            try:
                import tensorflow as tf
                print(f"\n🤖 TensorFlow version: {tf.__version__}")
                print(f"🔧 GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
            except:
                print(f"\n💡 TensorFlow will be needed later for model training")
                print("   Install with: pip install tensorflow-cpu (for Intel Mac)")
                print("   Or: pip install tensorflow-macos tensorflow-metal (for Apple Silicon)")
    else:
        print("❌ Setup incomplete - fix essential packages first!")
        print("\n💡 Try running the setup again after installing missing packages")

if __name__ == "__main__":
    main()