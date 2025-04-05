# AI Powered Technical Analysis Of Stock
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# Configure the Gemini API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
gen_model = genai.GenerativeModel(MODEL_NAME)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "RSI (14)"],
    default=["20-Day SMA"]
)

n_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)

# Fetch button
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                stock_data[ticker] = data
            else:
                st.warning(f"No data found for {ticker}.")
        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {e}")
    st.session_state["stock_data"] = stock_data
    if stock_data:
        st.success("Stock data loaded successfully.")

# RSI Calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# LSTM Forecasting
def predict_future_prices_lstm(data, n_days, look_back=60):
    df = data[['Close']].copy().dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
    future_predictions = []
    for _ in range(n_days):
        pred = model.predict(last_sequence, verbose=0)
        future_predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })
    return prediction_df

# Analysis function
def analyze_ticker(ticker, data, prediction_df):
    show_rsi = "RSI (14)" in indicators
    if show_rsi:
        fig, (ax_price, ax_rsi) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
        )
    else:
        fig, ax_price = plt.subplots(figsize=(12, 6))

    ax_price.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1)

    if "20-Day SMA" in indicators:
        sma = data['Close'].rolling(window=20).mean()
        ax_price.plot(data.index, sma, label='SMA (20)', linestyle='--', color='blue')

    if "20-Day EMA" in indicators:
        ema = data['Close'].ewm(span=20).mean()
        ax_price.plot(data.index, ema, label='EMA (20)', linestyle='--', color='green')

    if "20-Day Bollinger Bands" in indicators:
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        ax_price.plot(data.index, upper, label='Upper BB', linestyle=':', color='red')
        ax_price.plot(data.index, lower, label='Lower BB', linestyle=':', color='red')

    if "VWAP" in indicators:
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        ax_price.plot(data.index, data['VWAP'], label='VWAP', linestyle='-', color='purple')

    if not prediction_df.empty:
        ax_price.plot(prediction_df['Date'], prediction_df['Predicted_Close'], label='LSTM Prediction', linestyle='--', color='magenta')

    ax_price.set_title(f"{ticker} Technical Chart")
    ax_price.set_ylabel("Price")
    ax_price.legend()
    ax_price.grid(True)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    if show_rsi:
        rsi = calculate_rsi(data['Close'])
        ax_rsi.plot(data.index, rsi, label='RSI (14)', color='orange')
        ax_rsi.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax_rsi.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True)

    # Save figure
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, bbox_inches='tight')
        tmpfile_path = tmpfile.name
    with open(tmpfile_path, "rb") as f:
        image_bytes = f.read()
    os.remove(tmpfile_path)
    plt.close(fig)

    image_part = {
        "data": image_bytes,
        "mime_type": "image/png"
    }

    prompt = (
        f"You are a Stock Trader specializing in Technical Analysis. "
        f"Analyze the stock chart for {ticker} using the indicators shown. "
        f"Give a detailed explanation of the trend, patterns, and your recommendation from: "
        f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', 'Strong Sell'. "
        f"Respond in JSON with 'action' and 'justification'."
    )

    contents = [
        {"role": "user", "parts": [prompt]},
        {"role": "user", "parts": [image_part]}
    ]

    response = gen_model.generate_content(contents=contents)

    try:
        result_text = response.text
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        json_string = result_text[json_start:json_end]
        result = json.loads(json_string)
    except Exception as e:
        result = {
            "action": "Error",
            "justification": f"Error: {e}. Raw response: {response.text}"
        }

    return fig, result

# Render analysis
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    stock_data = st.session_state["stock_data"]
    tab_names = ["📋 Summary"] + list(stock_data.keys())
    tabs = st.tabs(tab_names)

    summary = []

    for i, ticker in enumerate(stock_data):
        data = stock_data[ticker]
        prediction_df = predict_future_prices_lstm(data, n_days)
        fig, result = analyze_ticker(ticker, data, prediction_df)
        summary.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})

        with tabs[i + 1]:
            st.subheader(f"📈 {ticker} Analysis")
            st.pyplot(fig)
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No explanation provided."))

            st.write("**📅 Forecasted Prices:**")
            st.dataframe(prediction_df.set_index('Date'))

            # Excel download
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                prediction_df.to_excel(writer, sheet_name='Forecast', index=False)
            st.download_button(
                label="📥 Download Forecast as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{ticker}_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # PDF download
            pdf_buffer = BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
            st.download_button(
                label="📥 Download Chart as PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"{ticker}_chart.pdf",
                mime="application/pdf"
            )

    with tabs[0]:
        st.subheader("📊 Overall Recommendations")
        df_summary = pd.DataFrame(summary)
        st.table(df_summary)
else:
    st.info("Please fetch stock data using the sidebar.")
