# 📈 Stocksight

**Stocksight** is a machine learning project for analyzing stock market data, engineering features, and predicting trading signals. It combines classical ML models with deep learning architectures (**GRU and LSTM networks**) to capture temporal patterns in stock price movements.

---

## 🚀 Features

- Historical stock data preprocessing (OHLCV, technical indicators).  
- Exploratory Data Analysis (EDA) with visualizations.  
- Feature importance and correlation analysis.  
- Model training with:
  - Random Forest, XGBoost, Logistic Regression  
  - **Recurrent Neural Networks (GRU & LSTM)** for time-series modeling  
- Evaluation metrics (Precision, Recall, F1, ROC-AUC, RMSE, etc.).  
- **Interactive Streamlit app** for running predictions and visualizing results.  

---

## 📂 Project Structure

.
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks for EDA, modeling, evaluation
├── src/ # Core Python modules (data, features, models, utils)
├── scripts/ # CLI scripts for training and backtesting
├── results/ # Model outputs and evaluation plots
├── app/ # Streamlit app files
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚡ Installation

Clone the repo:

```bash
git clone https://github.com/masonpan7/stocksight.git
cd stocksight
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🛠 Usage
Run the Streamlit app:

bash
Copy code
streamlit run app/app.py
This will launch a local web interface where you can:

Upload or select a stock dataset

Run trained models (Random Forest, XGBoost, GRU, LSTM)

Visualize predictions vs. actual stock movements

Explore feature correlations and importance

📊 Models Used
Classical ML Models:

Random Forest

XGBoost

Logistic Regression

Deep Learning Models:

GRU (Gated Recurrent Unit)

LSTM (Long Short-Term Memory)

These neural networks model sequential dependencies in stock time series, capturing both short- and long-term trends more effectively than classical models.

📈 Results
Interactive dashboards for prediction and analysis via Streamlit

Model comparison metrics (F1, ROC-AUC, RMSE, etc.)

Plots showing predicted vs. actual values and cumulative returns

🤝 Contributing
Contributions are welcome!
Feel free to fork the repo, open issues, or submit PRs.

📜 License
This project is licensed under the MIT License — see the LICENSE file.

📬 Contact
Created by Mason Pan
GitHub: @masonpan7
