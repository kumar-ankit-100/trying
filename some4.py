import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="StockVision Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS with glassmorphism, gradients, and animations
def get_advanced_theme_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #4a6741 100%);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            --backdrop-filter: blur(20px);
        }
        
        .stApp {
            background: var(--primary-gradient);
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
            text-align: center;
            color: white;
        }
        
        .main-header h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 3rem;
            margin: 0;
            # background: linear-gradient(45deg, #fff, #f093fb);
            # -webkit-background-clip: text;
            # -webkit-text-fill-color: transparent;
            # background-clip: text;
            color: white;
        }
        
        .metric-card {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 10px 0;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
            color: white;
            text-align: center;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.5);
        }
        
        .metric-card h4 {
            margin: 0 0 10px 0;
            font-weight: 500;
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        .metric-card h2 {
            margin: 0;
            font-weight: 700;
            font-size: 1.8rem;
        }
        
        .prediction-card {
            background: var(--accent-gradient);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 10px 0;
            box-shadow: var(--shadow);
            color: white;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .prediction-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px 0 rgba(79, 172, 254, 0.4);
        }
        
        .feature-card {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: var(--shadow);
            color: white;
        }
        
        .stButton > button {
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6);
        }
        
        .sidebar .stSelectbox > div > div {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
        }
        
        .stSelectbox > div > div {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            color: white;
        }
        
        .stDataFrame {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border-radius: 15px;
            overflow: hidden;
        }
        
        .prediction-summary {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
            color: white;
        }
        
        .section-header {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: white;
            margin: 2rem 0 1rem 0;
            font-size: 1.5rem;
        }
        
        .chart-container {
            background: var(--glass-bg);
            backdrop-filter: var(--backdrop-filter);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%);
            backdrop-filter: blur(20px);
        }
        
        /* Animation keyframes */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeInUp 0.6s ease-out;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            .metric-card {
                padding: 1rem;
            }
        }
    </style>
    """

# Initialize session state
if 'current_stock' not in st.session_state:
    st.session_state.current_stock = 'ASIANPAINT'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Apply theme
st.markdown(get_advanced_theme_css(), unsafe_allow_html=True)

# ... (Previous imports and CSS remain unchanged)

# Load dataset
@st.cache_data
def load_stock_data(file_path='ASIANPAINT.csv'):
    """Load and prepare the ASIANPAINT dataset"""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.sort_index()
    return data

# Load model and scaler
@st.cache_resource
def load_model_and_scaler(model_path='stock_prediction_model.h5', scaler_path='scaler.pkl'):
    """Load the pre-trained model and scaler"""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Predict future prices
def predict_future(model, data, scaler, horizon, time_steps=60):
    """Generate future price predictions using the trained model"""
    last_sequence = data['Close'][-time_steps:].values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    current_sequence = last_sequence_scaled.reshape(1, time_steps, 1)
    
    predicted_prices = []
    for _ in range(horizon):
        pred_scaled = model.predict(current_sequence, verbose=0)
        predicted_prices.append(pred_scaled[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = pred_scaled[0, 0]
    
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    return scaler.inverse_transform(predicted_prices).flatten()

# Calculate technical indicators
def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    return data

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.markdown("### 🧭 Navigation")
    st.markdown('</div>', unsafe_allow_html=True)
    
    page = st.selectbox(
        "Select Page",
        ["📊 Dashboard", "🔮 Predictions", "📈 Technical Analysis", "🔍 Multi-Stock Comparison", "📋 Reports & Insights"],
        key="page_selector"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Stock Selection")
    stock_symbol = st.selectbox(
        "Choose Stock", 
        ['ASIANPAINT'], 
        key="stock_selector"
    )
    st.session_state.current_stock = stock_symbol
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    dark_mode = st.checkbox("🌙 Dark Mode", value=True)
    st.session_state.dark_mode = dark_mode
    
    st.markdown("### 🕐 Market Status")
    current_time = datetime.now().time()
    if datetime.now().weekday() < 5 and 9 <= current_time.hour < 16:
        st.success("🟢 Market Open (IST)")
    else:
        st.error("🔴 Market Closed (IST)")

# Load data and model
data = load_stock_data()
model, scaler = load_model_and_scaler()
data = calculate_technical_indicators(data)

# Main content based on selected page
if "Dashboard" in page:
    st.markdown(f'''
    <div class="main-header">
        <h1>📈 {st.session_state.current_stock}</h1>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">Advanced Stock Market Prediction Platform</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📊 Real-Time Market Data</h2>', unsafe_allow_html=True)
    
    last_data = data.iloc[-1]
    prev_data = data.iloc[-2]
    
    price_change = last_data['Close'] - prev_data['Close']
    price_change_pct = (price_change / prev_data['Close']) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'''
        <div class="metric-card fade-in">
            <h4>Current Price</h4>
            <h2>₹{last_data["Close"]:.2f}</h2>
            <small style="color: {'#00ff88' if price_change >= 0 else '#ff6b6b'};">
                {'+' if price_change >= 0 else ''}{price_change:.2f} ({price_change_pct:+.2f}%)
            </small>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card fade-in"><h4>Day High</h4><h2>₹{last_data["High"]:.2f}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card fade-in"><h4>Day Low</h4><h2>₹{last_data["Low"]:.2f}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card fade-in"><h4>Opening</h4><h2>₹{last_data["Open"]:.2f}</h2></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card fade-in"><h4>Volume</h4><h2>{last_data["Volume"]:,}</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📈 Price Movement & Predictions</h2>', unsafe_allow_html=True)
    
    pred_prices_30 = predict_future(model, data, scaler, 30)
    pred_dates_30 = pd.date_range(start=data.index[-1] + pd.offsets.BDay(1), periods=30, freq='B')
    
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:], name="Historical", line=dict(color='#00ff88', width=3)))
    fig_main.add_trace(go.Scatter(x=pred_dates_30, y=pred_prices_30, name="30-Day Prediction", line=dict(color='#ff6b6b', width=3, dash='dash')))
    upper_bound = pred_prices_30 * 1.05
    lower_bound = pred_prices_30 * 0.95
    fig_main.add_trace(go.Scatter(x=pred_dates_30, y=upper_bound, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig_main.add_trace(go.Scatter(x=pred_dates_30, y=lower_bound, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='95% Confidence', fillcolor='rgba(255, 107, 107, 0.2)'))
    # Fix for add_vline by converting Timestamp to string
    fig_main.add_vline(x=data.index[-1].strftime("%Y-%m-%d %H:%M:%S"), line_dash="dot", line_color="rgb(255,0,0)")
    fig_main.update_layout(title="Historical Performance vs Future Predictions", xaxis_title="Date", yaxis_title="Price (₹)", template="plotly_dark", height=600)
    st.plotly_chart(fig_main, use_container_width=True)
    
    st.markdown('<h2 class="section-header">🔮 Prediction Summary</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    horizons = [7, 15, 30]
    for i, days in enumerate(horizons):
        pred_prices = predict_future(model, data, scaler, days)
        predicted_price = pred_prices[-1]
        current_price = data['Close'].iloc[-1]
        change = ((predicted_price - current_price) / current_price) * 100
        with [col1, col2, col3][i]:
            st.markdown(f'''
            <div class="prediction-card">
                <h4>{days} Days Forecast</h4>
                <h2>₹{predicted_price:.2f}</h2>
                <p style="margin: 5px 0; font-size: 0.9rem;">
                    Expected Change: <strong style="color: {'#00ff88' if change >= 0 else '#ffaa44'};">
                    {'+' if change >= 0 else ''}{change:.1f}%</strong>
                </p>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📅 Next 7 Days Detailed Forecast</h2>', unsafe_allow_html=True)
    pred_dates_7 = pd.date_range(start=data.index[-1] + pd.offsets.BDay(1), periods=7, freq='B')
    pred_prices_7 = predict_future(model, data, scaler, 7)
    detailed_predictions = []
    for i, (date, price) in enumerate(zip(pred_dates_7, pred_prices_7)):
        if i == 0:
            change = ((price - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
        else:
            change = ((price - pred_prices_7[i-1]) / pred_prices_7[i-1]) * 100
        detailed_predictions.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Day': date.strftime('%A'),
            'Predicted Price (₹)': f"₹{price:.2f}",
            'Expected Change (%)': f"{change:+.2f}%",
            'Confidence': 'High' if abs(change) < 2 else 'Medium' if abs(change) < 4 else 'Low'
        })
    df_detailed = pd.DataFrame(detailed_predictions)
    st.dataframe(df_detailed, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 class="section-header">📊 Market Sentiment</h3>', unsafe_allow_html=True)
        sentiment_data = pd.DataFrame({'Sentiment': ['Bullish', 'Bearish', 'Neutral'], 'Percentage': [45, 30, 25]})
        fig_sentiment = px.pie(sentiment_data, values='Percentage', names='Sentiment', color_discrete_sequence=['#00ff88', '#ff6b6b', '#ffd93d'])
        fig_sentiment.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    with col2:
        st.markdown('<h3 class="section-header">📈 Performance Metrics</h3>', unsafe_allow_html=True)
        returns_30d = ((data['Close'].iloc[-1] - data['Close'].iloc[-31]) / data['Close'].iloc[-31]) * 100
        returns_90d = ((data['Close'].iloc[-1] - data['Close'].iloc[-91]) / data['Close'].iloc[-91]) * 100
        volatility = data['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
        st.markdown(f'''
        <div class="feature-card">
            <h4>📊 Key Metrics</h4>
            <div style="margin: 1rem 0;">
                <p><strong>30-Day Return:</strong> <span style="color: {'#00ff88' if returns_30d >= 0 else '#ff6b6b'};">{returns_30d:+.2f}%</span></p>
                <p><strong>90-Day Return:</strong> <span style="color: {'#00ff88' if returns_90d >= 0 else '#ff6b6b'};">{returns_90d:+.2f}%</span></p>
                <p><strong>Volatility (Annualized):</strong> <span style="color: #ffd93d;">{volatility:.2f}%</span></p>
                <p><strong>Current RSI:</strong> <span style="color: {'#ff6b6b' if data['RSI'].iloc[-1] > 70 else '#00ff88' if data['RSI'].iloc[-1] < 30 else '#ffd93d'};">{data['RSI'].iloc[-1]:.1f}</span></p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

elif "Predictions" in page:
    st.markdown(f'''
    <div class="main-header">
        <h1>🔮 Advanced Predictions</h1>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">AI-Powered Stock Price Forecasting</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">⚙️ Prediction Settings</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pred_duration = st.selectbox("Forecast Duration", ["7 days", "15 days", "30 days", "60 days", "90 days"])
    with col2:
        confidence_level = st.slider("Confidence Level", 80, 99, 95, step=1)
    with col3:
        show_technical = st.checkbox("Include Technical Indicators", value=True)
    
    days = int(pred_duration.split()[0])
    pred_prices = predict_future(model, data, scaler, days)
    pred_dates = pd.date_range(start=data.index[-1] + pd.offsets.BDay(1), periods=days, freq='B')
    confidence_factor = confidence_level / 100
    confidence_range = 0.1 * (1 - confidence_factor)
    confidence_upper = pred_prices * (1 + confidence_range)
    confidence_lower = pred_prices * (1 - confidence_range)
    
    st.markdown('<h2 class="section-header">📈 Prediction Visualization</h2>', unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=data.index[-60:], y=data['Close'][-60:], name="Historical", line=dict(color='#00ff88', width=3)))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_prices, name=f"{days}-Day Prediction", line=dict(color='#ff6b6b', width=3)))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=confidence_upper, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=confidence_lower, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name=f'{confidence_level}% Confidence', fillcolor='rgba(255, 107, 107, 0.3)'))
    if show_technical:
        fig_pred.add_trace(go.Scatter(x=data.index[-60:], y=data['MA20'][-60:], name="MA20", line=dict(color='#ffd93d', width=2, dash='dot')))
        fig_pred.add_trace(go.Scatter(x=data.index[-60:], y=data['MA50'][-60:], name="MA50", line=dict(color='#ff9500', width=2, dash='dot')))
    fig_pred.add_vline(x=data.index[-1].to_pydatetime(), line_dash="dot", line_color="rgba(255, 0, 0, 0.5)")
    fig_pred.update_layout(title=f"{days}-Day Price Prediction with {confidence_level}% Confidence", xaxis_title="Date", yaxis_title="Price (₹)", template="plotly_dark", height=700)
    st.plotly_chart(fig_pred, use_container_width=True)

elif "Technical Analysis" in page:
    st.markdown(f'''
    <div class="main-header">
        <h1>📈 Technical Analysis</h1>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">Advanced Charting & Technical Indicators</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">⚙️ Indicator Settings</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_ma20 = st.checkbox("20-Day MA", value=True)
        show_ma50 = st.checkbox("50-Day MA", value=True)
    with col2:
        show_ma200 = st.checkbox("200-Day MA", value=False)
        show_bb = st.checkbox("Bollinger Bands", value=True)
    with col3:
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
    with col4:
        timeframe = st.selectbox("Timeframe", ["30 days", "90 days", "180 days", "1 year"])
    
    days_back = int(timeframe.split()[0]) if timeframe.split()[0].isdigit() else 365
    chart_data = data.tail(days_back)
    
    rows = 1
    if show_rsi: rows += 1
    if show_macd: rows += 1
    subplot_titles = ['Price & Technical Indicators']
    if show_rsi: subplot_titles.append('RSI')
    if show_macd: subplot_titles.append('MACD')
    
    fig_tech = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=subplot_titles)
    fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], name="Close", line=dict(color='#00ff88')), row=1, col=1)
    if show_ma20: fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MA20'], name="MA20", line=dict(color='#ffd93d', dash='dash')), row=1, col=1)
    if show_ma50: fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MA50'], name="MA50", line=dict(color='#ff9500', dash='dash')), row=1, col=1)
    if show_ma200: fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MA200'], name="MA200", line=dict(color='#ff6b6b', dash='dash')), row=1, col=1)
    if show_bb:
        fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['BB_Upper'], name="BB Upper", line=dict(color='rgba(128,128,128,0.5)')), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['BB_Lower'], name="BB Lower", line=dict(color='rgba(128,128,128,0.5)'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    current_row = 2
    if show_rsi:
        fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], name="RSI", line=dict(color='#4facfe')), row=current_row, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    if show_macd:
        fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD'], name="MACD", line=dict(color='#ff6b6b')), row=current_row, col=1)
        fig_tech.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD_Signal'], name="Signal", line=dict(color='#ffd93d')), row=current_row, col=1)
        macd_histogram = chart_data['MACD'] - chart_data['MACD_Signal']
        colors = ['green' if val >= 0 else 'red' for val in macd_histogram]
        fig_tech.add_trace(go.Bar(x=chart_data.index, y=macd_histogram, name="Histogram", marker_color=colors, opacity=0.6), row=current_row, col=1)
    fig_tech.update_layout(title="Technical Analysis Dashboard", template="plotly_dark", height=600 + (rows-1)*200)
    st.plotly_chart(fig_tech, use_container_width=True)
    
    st.markdown('<h2 class="section-header">📊 Technical Analysis Summary</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        current_rsi = chart_data['RSI'].iloc[-1]
        current_macd = chart_data['MACD'].iloc[-1]
        current_signal = chart_data['MACD_Signal'].iloc[-1]
        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
        st.markdown(f'''
        <div class="feature-card">
            <h3>🎯 Current Signals</h3>
            <div style="margin: 1rem 0;">
                <p><strong>RSI Signal:</strong> <span style="color: {'#ff6b6b' if rsi_signal == 'Overbought' else '#00ff88' if rsi_signal == 'Oversold' else '#ffd93d'};">{rsi_signal} ({current_rsi:.1f})</span></p>
                <p><strong>MACD Signal:</strong> <span style="color: {'#00ff88' if macd_signal == 'Bullish' else '#ff6b6b'};">{macd_signal}</span></p>
                <p><strong>Price vs MA20:</strong> <span style="color: {'#00ff88' if chart_data['Close'].iloc[-1] > chart_data['MA20'].iloc[-1] else '#ff6b6b'};">{'Above' if chart_data['Close'].iloc[-1] > chart_data['MA20'].iloc[-1] else 'Below'}</span></p>
                <p><strong>Price vs MA50:</strong> <span style="color: {'#00ff88' if chart_data['Close'].iloc[-1] > chart_data['MA50'].iloc[-1] else '#ff6b6b'};">{'Above' if chart_data['Close'].iloc[-1] > chart_data['MA50'].iloc[-1] else 'Below'}</span></p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        resistance = chart_data['High'].tail(30).max()
        support = chart_data['Low'].tail(30).min()
        st.markdown(f'''
        <div class="feature-card">
            <h3>📈 Key Levels</h3>
            <div style="margin: 1rem 0;">
                <p><strong>Resistance Level:</strong> <span style="color: #ff6b6b;">₹{resistance:.2f}</span></p>
                <p><strong>Support Level:</strong> <span style="color: #00ff88;">₹{support:.2f}</span></p>
                <p><strong>Current Position:</strong> <span style="color: #ffd93d;">{((chart_data['Close'].iloc[-1] - support) / (resistance - support) * 100):.1f}% of range</span></p>
                <p><strong>Volatility (30D):</strong> <span style="color: #4facfe;">{chart_data['Close'].pct_change().tail(30).std() * 100:.2f}%</span></p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

elif "Multi-Stock Comparison" in page:
    st.markdown(f'''
    <div class="main-header">
        <h1>🔍 Multi-Stock Analysis</h1>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">Compare Multiple Stocks & Portfolio Management</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.info("📌 Currently showing simulated comparison with ASIANPAINT variations.")
    
    stocks_data = {
        'ASIANPAINT': data,
        'ASIANPAINT_Bullish': data.copy(),
        'ASIANPAINT_Conservative': data.copy()
    }
    stocks_data['ASIANPAINT_Bullish']['Close'] = stocks_data['ASIANPAINT_Bullish']['Close'] * 1.15
    stocks_data['ASIANPAINT_Conservative']['Close'] = stocks_data['ASIANPAINT_Conservative']['Close'] * 0.92
    
    st.markdown('<h2 class="section-header">📊 Stock Selection</h2>', unsafe_allow_html=True)
    selected_stocks = st.multiselect("Select stocks to compare:", list(stocks_data.keys()), default=list(stocks_data.keys()))
    
    if selected_stocks:
        st.markdown('<h2 class="section-header">📈 Price Comparison</h2>', unsafe_allow_html=True)
        fig_multi = go.Figure()
        colors = ['#00ff88', '#ff6b6b', '#ffd93d', '#4facfe', '#ff9500']
        for i, stock in enumerate(selected_stocks):
            stock_data = stocks_data[stock]
            fig_multi.add_trace(go.Scatter(x=stock_data.index[-90:], y=stock_data['Close'][-90:], name=stock, line=dict(color=colors[i % len(colors)], width=2)))
        fig_multi.update_layout(title="Multi-Stock Price Comparison (Last 90 Days)", xaxis_title="Date", yaxis_title="Price (₹)", template="plotly_dark", height=600)
        st.plotly_chart(fig_multi, use_container_width=True)
        
        st.markdown('<h2 class="section-header">📊 Performance Metrics</h2>', unsafe_allow_html=True)
        perf_data = []
        for stock in selected_stocks:
            stock_data = stocks_data[stock]
            current_price = stock_data['Close'].iloc[-1]
            returns_30d = ((current_price - stock_data['Close'].iloc[-31]) / stock_data['Close'].iloc[-31]) * 100
            returns_90d = ((current_price - stock_data['Close'].iloc[-91]) / stock_data['Close'].iloc[-91]) * 100
            volatility = stock_data['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
            perf_data.append({
                'Stock': stock,
                'Current Price': f"₹{current_price:.2f}",
                '30D Return (%)': f"{returns_30d:+.2f}%",
                '90D Return (%)': f"{returns_90d:+.2f}%",
                'Volatility (%)': f"{volatility:.2f}%",
                'Volume': f"{stock_data['Volume'].iloc[-1]:,}"
            })
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, use_container_width=True)
        
        st.markdown('<h2 class="section-header">💼 Portfolio Allocation</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            weights = {}
            total_weight = 0
            st.markdown("### Set Portfolio Weights")
            for stock in selected_stocks:
                weight = st.slider(f"{stock} allocation", 0, 100, 100 // len(selected_stocks))
                weights[stock] = weight
                total_weight += weight
            if total_weight != 100:
                st.warning(f"⚠️ Total allocation: {total_weight}%. Please adjust to 100%.")
        with col2:
            if total_weight > 0:
                fig_portfolio = px.pie(values=list(weights.values()), names=list(weights.keys()), title="Portfolio Allocation", color_discrete_sequence=colors)
                fig_portfolio.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_portfolio, use_container_width=True)

elif "Reports & Insights" in page:
    st.markdown(f'''
    <div class="main-header">
        <h1>📋 Reports & Insights</h1>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">Comprehensive Analysis Reports</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">⚙️ Report Settings</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        report_type = st.selectbox("Report Type", ["Prediction Report", "Technical Analysis", "Performance Summary"])
    with col2:
        report_period = st.selectbox("Time Period", ["7 days", "15 days", "30 days", "60 days"])
    with col3:
        include_charts = st.checkbox("Include Charts", value=True)
    
    for days in [7, 15, 30]:
        with st.expander(f"📊 {days}-Day Comprehensive Report", expanded=(days == 7)):
            col1, col2 = st.columns([2, 1])
            with col1:
                pred_prices = predict_future(model, data, scaler, days)
                pred_dates = pd.date_range(start=data.index[-1] + pd.offsets.BDay(1), periods=days, freq='B')
                fig_report = go.Figure()
                fig_report.add_trace(go.Scatter(x=data.index[-30:], y=data['Close'][-30:], name="Historical", line=dict(color='#00ff88')))
                fig_report.add_trace(go.Scatter(x=pred_dates, y=pred_prices, name="Prediction", line=dict(color='#ff6b6b', dash='dash')))
                fig_report.add_vline(x=str(data.index[-1]), line_dash="dot", line_color="rgba(255,255,255,0.5)")
                fig_report.update_layout(title=f"{days}-Day Forecast Overview", template="plotly_dark", height=350)
                st.plotly_chart(fig_report, use_container_width=True)
            with col2:
                current_price = data['Close'].iloc[-1]
                predicted_price = pred_prices[-1]
                change = ((predicted_price - current_price) / current_price) * 100
                volatility = data['Close'].pct_change().tail(30).std() * 100
                risk_level = "High" if volatility > 3 else "Medium" if volatility > 1.5 else "Low"
                st.markdown(f'''
                <div class="feature-card">
                    <h3>📈 Report Summary</h3>
                    <div style="margin: 1rem 0;">
                        <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
                        <p><strong>Predicted Price:</strong> ₹{predicted_price:.2f}</p>
                        <p><strong>Expected Change:</strong> <span style="color: {'#00ff88' if change >= 0 else '#ff6b6b'};">{'+' if change >= 0 else ''}{change:.2f}%</span></p>
                        <p><strong>Risk Level:</strong> <span style="color: {'#ff6b6b' if risk_level == 'High' else '#ffd93d' if risk_level == 'Medium' else '#00ff88'};">{risk_level}</span></p>
                        <p><strong>Confidence:</strong> <span style="color: #4facfe;">{85 if days <= 7 else 75 if days <= 15 else 65}%</span></p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                report_content = f"""
ASIANPAINT Stock Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}

=== {days}-Day Forecast ===
Current Price: ₹{current_price:.2f}
Predicted Price: ₹{predicted_price:.2f}
Expected Change: {change:+.2f}%
Risk Level: {risk_level}
Volatility: {volatility:.2f}%

=== Technical Indicators ===
RSI: {data['RSI'].iloc[-1]:.1f}
MACD: {data['MACD'].iloc[-1]:.4f}
MA20: ₹{data['MA20'].iloc[-1]:.2f}
MA50: ₹{data['MA50'].iloc[-1]:.2f}

=== Recommendation ===
{'BUY' if change > 2 else 'SELL' if change < -2 else 'HOLD'}
                """
                st.download_button(f"📥 Download {days}D Report", data=report_content, file_name=f"ASIANPAINT_{days}day_report.txt", mime="text/plain")
    
    st.markdown('<h2 class="section-header">🔍 Market Insights</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="feature-card">
            <h3>📊 Key Market Insights</h3>
            <div style="margin: 1rem 0;">
                <p>• <strong>Current Trend:</strong> {'Bullish' if data['Close'].iloc[-1] > data['MA20'].iloc[-1] else 'Bearish'}</p>
                <p>• <strong>Support Level:</strong> ₹{data['Close'].tail(30).min():.2f}</p>
                <p>• <strong>Resistance Level:</strong> ₹{data['Close'].tail(30).max():.2f}</p>
                <p>• <strong>Volume Trend:</strong> {'Increasing' if data['Volume'].iloc[-1] > data['Volume'].tail(5).mean() else 'Decreasing'}</p>
                <p>• <strong>Momentum:</strong> {'Strong' if abs(data['RSI'].iloc[-1] - 50) > 20 else 'Moderate'}</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="feature-card">
            <h3>⚠️ Risk Factors</h3>
            <div style="margin: 1rem 0;">
                <p>• <strong>Market Volatility:</strong> {'High' if data['Close'].pct_change().tail(30).std() > 0.03 else 'Normal'}</p>
                <p>• <strong>RSI Status:</strong> {'Overbought' if data['RSI'].iloc[-1] > 70 else 'Oversold' if data['RSI'].iloc[-1] < 30 else 'Normal'}</p>
                <p>• <strong>Price Position:</strong> {'Near High' if data['Close'].iloc[-1] > data['Close'].tail(30).quantile(0.8) else 'Near Low' if data['Close'].iloc[-1] < data['Close'].tail(30).quantile(0.2) else 'Mid-Range'}</p>
                <p>• <strong>Prediction Confidence:</strong> {'High (>80%)' if data['Close'].pct_change().tail(30).std() < 0.02 else 'Medium (60-80%)'}</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('''
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
    <p>🚀 <strong>StockVision Pro</strong> - Advanced AI-Powered Stock Market Prediction Platform</p>
    <p>💡 Built with cutting-edge machine learning algorithms and modern web technologies</p>
    <p style="font-size: 0.9rem;">⚠️ <em>This is a demo application. Always consult with financial advisors for investment decisions.</em></p>
</div>
''', unsafe_allow_html=True)