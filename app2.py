import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple, List, Optional

# --- Page Configuration ---
st.set_page_config(page_title="Institutional AI Trader", layout="wide", page_icon="📈")

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-container {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Institutional AI Trading System")
st.markdown("Automated Technical Analysis & Machine Learning Pipeline.")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input("Asset Ticker", "BTC-USD")
start_date = st.sidebar.date_input("Training Start Date", pd.to_datetime("2020-01-01"))
training_window = st.sidebar.slider("Backtest Window (Days)", 50, 365, 100)

st.sidebar.markdown("---")
st.sidebar.caption("**Disclaimer:** This tool is for educational purposes only. Not financial advice.")

def get_data_and_features(ticker: str, start_date: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Retrieves data and performs Feature Engineering using Pandas-TA.
    Returns: DataFrame with indicators and lagged features.
    """
    try:
        df = yf.Ticker(ticker).history(start=start_date)
        
        if df.empty:
            return None, f"No data found for '{ticker}'. Check spelling or suffix (e.g., .ST)."
        
        if len(df) < 300:
            return None, f"Insufficient data ({len(df)} rows). Need at least 300 days."

        # --- Technical Indicators ---
        # Trend
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.macd(append=True)
        # Momentum
        df.ta.rsi(length=14, append=True)
        # Volatility
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        # Volume
        df.ta.vwap(append=True)

        # --- Feature Engineering (The Alpha) ---
        df['Return'] = df['Close'].pct_change()
        df['Lag_1'] = df['Return'].shift(1)
        df['Lag_2'] = df['Return'].shift(2)
        df['Volatility_5'] = df['Return'].rolling(window=5).std()
        
        df.dropna(inplace=True)
        
        # Target: 1 if Close(t+1) > Close(t)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df[:-1] # Drop last row (no target)
        
        return df, None

    except Exception as e:
        return None, f"Error: {str(e)}"

def train_and_evaluate(df: pd.DataFrame, test_days: int) -> Tuple[any, float, pd.DataFrame, any, any, List[str]]:
    """
    Trains a Random Forest Classifier with a Time-Series Split.
    """
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target']
    features = [c for c in df.columns if c not in exclude]
    
    X = df[features]
    y = df['Target']
    
    # Split Data
    split_idx = -test_days
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Model (Limited depth to prevent overfitting)
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=5, 
        min_samples_split=20, 
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    
    return model, acc, X_test, preds, proba, features

# --- Main Logic ---
if st.sidebar.button("Run System Analysis"):
    with st.spinner('Analyzing market structure & training models...'):
        df, error = get_data_and_features(ticker, start_date)
        
        if error:
            st.error(error)
        else:
            model, acc, X_test, preds, proba, feature_names = train_and_evaluate(df, training_window)
            
            # Prediction for Tomorrow
            last_row = df.iloc[[-1]]
            next_pred = model.predict(last_row[feature_names])[0]
            next_prob = model.predict_proba(last_row[feature_names])[0]
            
            # --- Dashboard UI ---
            
            # 1. Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            signal = "BUY" if next_pred == 1 else "SELL"
            signal_color = "#00ff00" if next_pred == 1 else "#ff0000"
            conf = next_prob[1] if next_pred == 1 else next_prob[0]
            
            col1.metric("Asset Price", f"{current_price:.2f}")
            col2.metric("Model Accuracy", f"{acc:.1%}")
            col3.markdown(f"**AI Signal:** <span style='color:{signal_color}; font-weight:bold'>{signal}</span>", unsafe_allow_html=True)
            col4.metric("Confidence", f"{conf:.1%}")

            # 2. Charts
            tab1, tab2 = st.tabs(["Technical Chart", "Model Performance"])
            
            with tab1:
                st.subheader(f"Price Action & Indicators: {ticker}")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                
                # Candles
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
                # VWAP
                if 'VWAP_D' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_D'], line=dict(color='#ffa500', width=1), name='VWAP'), row=1, col=1)
                # BB
                if 'BBU_20_2.0' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='gray', dash='dot'), name='Upper BB'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], line=dict(color='gray', dash='dot'), name='Lower BB'), row=1, col=1)
                # Volume
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
                
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Feature Importance")
                imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
                st.bar_chart(imp_df.set_index('Feature').sort_values(by='Importance', ascending=False).head(10))

            # 3. Data Export
            st.markdown("---")
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Analysis Data (CSV)",
                data=csv,
                file_name=f'{ticker}_ai_analysis.csv',
                mime='text/csv',
            )

else:
    st.info("System Ready. Adjust parameters in the sidebar and click 'Run System Analysis'.")