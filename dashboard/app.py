import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from model_loader import ModelLoader

st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_model_loader():
    """Initialize the model loader."""
    return ModelLoader()

def load_models(model_loader):
    """Load models and show status."""
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading models..."):
            success = model_loader.load_models()
            st.session_state.models_loaded = success
            st.session_state.model_loader = model_loader
    
    return st.session_state.models_loaded

def main():
    """Main dashboard application."""
    st.title("Daily Stock Movement Predictor")
    st.markdown("Predict whether your stocks will go UP or DOWN tomorrow using machine learning models")
    
    model_loader = initialize_model_loader()
    models_loaded = load_models(model_loader)
    
    if not models_loaded:
        st.error("Could not load models. Please ensure model files are in the correct directory.")
        
        with st.expander("Debug Information", expanded=True):
            st.write(f"**Model path:** {model_loader.model_path}")
            st.write(f"**Available models:** {model_loader.get_available_models()}")
            
            import os
            if os.path.exists(model_loader.model_path):
                files = os.listdir(model_loader.model_path)
                st.write(f"**Files in model directory:** {files}")
            else:
                st.error(f"Model directory does not exist: {model_loader.model_path}")
        
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    available_models = model_loader.get_available_models()
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Choose Model:",
            available_models,
            index=0
        )
    else:
        st.error("No models available")
        st.stop()
    
    threshold = st.sidebar.slider(
        "Prediction Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher threshold = more conservative UP predictions"
    )
    
    st.sidebar.subheader("Stock Portfolio")
    all_stocks = model_loader.stock_symbols
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks to Monitor:",
        all_stocks,
        default=all_stocks[:5],
        help="Choose which stocks to include in predictions"
    )
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Daily Predictions", "Individual Analysis", "Model Performance", "Settings"])
    
    with tab1:
        display_daily_predictions(model_loader, selected_model, selected_stocks, threshold)
    
    with tab2:
        display_individual_analysis(model_loader, selected_model, threshold)
    
    with tab3:
        display_model_performance(model_loader)
    
    with tab4:
        display_settings()
    
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

def display_daily_predictions(model_loader, selected_model, selected_stocks, threshold):
    """Display daily predictions for all selected stocks."""
    st.header(f"Today's Predictions ({selected_model})")
    
    if not selected_stocks:
        st.warning("Please select at least one stock from the sidebar.")
        return
    
    with st.spinner("Making predictions..."):
        predictions = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(selected_stocks):
            prediction = model_loader.predict_stock(symbol, selected_model, threshold)
            if prediction:
                predictions.append(prediction)
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        progress_bar.empty()
    
    if not predictions:
        st.error("Could not make predictions. Please check if models are loaded correctly.")
        st.write("**Debug Info:**")
        st.write(f"Selected model: {selected_model}")
        st.write(f"Available models: {model_loader.get_available_models()}")
        st.write(f"Selected stocks: {selected_stocks}")
        return
    
    df_predictions = pd.DataFrame(predictions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        up_count = len(df_predictions[df_predictions['prediction'] == 'UP'])
        st.metric("Predicted UP", up_count, f"{up_count/len(df_predictions)*100:.1f}%")
    
    with col2:
        down_count = len(df_predictions[df_predictions['prediction'] == 'DOWN'])
        st.metric("Predicted DOWN", down_count, f"{down_count/len(df_predictions)*100:.1f}%")
    
    with col3:
        avg_confidence = df_predictions['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col4:
        total_stocks = len(predictions)
        st.metric("Total Stocks", total_stocks)
    
    st.subheader("Detailed Predictions")
    
    display_df = df_predictions.copy()
    display_df['Probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['Change %'] = display_df['price_change_pct'].apply(lambda x: f"{x:+.2f}%")
    
    def style_prediction(val):
        if val == 'UP':
            return 'background-color: #90EE90; color: black'
        elif val == 'DOWN':
            return 'background-color: #FFB6C1; color: black'
        return ''
    
    styled_df = display_df[['symbol', 'prediction', 'Probability', 'Confidence', 'Current Price', 'Change %']].style.applymap(
        style_prediction, subset=['prediction']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.subheader("Prediction Visualization")
    
    fig_dist = px.pie(
        df_predictions, 
        names='prediction', 
        title='Prediction Distribution',
        color_discrete_map={'UP': '#90EE90', 'DOWN': '#FFB6C1'}
    )
    
    fig_conf = px.histogram(
        df_predictions,
        x='confidence',
        nbins=10,
        title='Confidence Distribution',
        color='prediction',
        color_discrete_map={'UP': '#90EE90', 'DOWN': '#FFB6C1'}
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        st.plotly_chart(fig_conf, use_container_width=True)

def display_individual_analysis(model_loader, selected_model, threshold):
    """Display detailed analysis for a single stock."""
    st.header("Individual Stock Analysis")
    
    selected_symbol = st.selectbox(
        "Select Stock for Detailed Analysis:",
        model_loader.stock_symbols
    )
    
    if st.button("Analyze Stock"):
        with st.spinner(f"Analyzing {selected_symbol}..."):
            prediction = model_loader.predict_stock(selected_symbol, selected_model, threshold)
            
            if not prediction:
                st.error(f"Could not analyze {selected_symbol}")
                return
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                direction_color = "green" if prediction['prediction'] == 'UP' else "red"
                st.markdown(f"<h2 style='color: {direction_color}'>{prediction['prediction']}</h2>", unsafe_allow_html=True)
                st.metric("Prediction", prediction['prediction'])
            
            with col2:
                st.metric("Confidence", f"{prediction['confidence']:.1%}")
                st.metric("Probability", f"{prediction['probability']:.1%}")
            
            with col3:
                st.metric("Current Price", f"${prediction['current_price']:.2f}")
                st.metric("Daily Change", f"{prediction['price_change_pct']:+.2f}%")
            
            stock_data = model_loader.get_stock_data(selected_symbol, period='1mo')
            
            if stock_data is not None:
                st.subheader("Recent Price Movement")
                
                fig = go.Figure(data=go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                ))
                
                fig.update_layout(
                    title=f"{selected_symbol} - Last 30 Days",
                    yaxis_title="Price ($)",
                    xaxis_title="Date"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if 'RSI' in stock_data.columns:
                    st.subheader("Technical Indicators")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("RSI", f"{stock_data['RSI'].iloc[-1]:.1f}")
                        if stock_data['RSI'].iloc[-1] > 70:
                            st.caption("Potentially Overbought")
                        elif stock_data['RSI'].iloc[-1] < 30:
                            st.caption("Potentially Oversold")
                    
                    with col2:
                        if 'MACD' in stock_data.columns:
                            st.metric("MACD", f"{stock_data['MACD'].iloc[-1]:.3f}")

def display_model_performance(model_loader):
    """Display model performance and comparison."""
    st.header("Model Performance")
    
    st.info("Model performance metrics would be displayed here based on historical backtesting results.")
    
    models = model_loader.get_available_models()
    
    st.subheader("Available Models")
    for model in models:
        st.write(f"✅ {model}")
    
    st.subheader("Performance Metrics")
    st.write("This section would show:")
    st.write("- Historical accuracy")
    st.write("- Precision and recall")
    st.write("- Confusion matrices")
    st.write("- ROC curves")

def display_settings():
    """Display app settings and configuration."""
    st.header("Settings")
    
    st.subheader("About")
    st.write("""
    This dashboard uses machine learning models to predict stock price movements:
    
    **Models Available:**
    - LSTM (Long Short-Term Memory): Neural network good at learning from time series
    - GRU (Gated Recurrent Unit): Lighter alternative to LSTM
    - Random Forest: Ensemble method using decision trees
    - Logistic Regression: Linear classification model
    
    **Features Used:**
    - Price data (Open, High, Low, Close, Volume)
    - Technical indicators (SMA, RSI, MACD, Bollinger Bands)
    - Price change and volatility measures
    """)
    
    st.subheader("Disclaimer")
    st.warning("""
    **Investment Warning:**
    - These predictions are for educational purposes only
    - Past performance does not guarantee future results
    - Always do your own research before making investment decisions
    - Consider consulting with a financial advisor
    - Never invest more than you can afford to lose
    """)
    
    st.subheader("Technical Details")
    st.write(f"- Sequence Length: {30} days")
    st.write(f"- Update Frequency: Real-time (market hours)")
    st.write(f"- Data Source: Yahoo Finance")

if __name__ == "__main__":
    main()