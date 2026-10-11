# AI-Powered Technical Stock Analysis

## Project Overview
AI-Powered Technical Stock Analysis is a Python-based web application that analyzes stock market trends using technical indicators and AI-generated insights. It provides interactive stock charts, technical analysis, price predictions, and AI-assisted market interpretations through a Streamlit interface. Analyzes the stock based on the its open and close price and generate a AI based recommendation for 'Buy', 'Sell' and 'Hold' with a custom prediction of its future prices

## Objectives
- Analyze historical stock price movements.
- Visualize trends using technical indicators.
- Generate AI-assisted stock market insights.
- Explore potential price movements using an LSTM model.
- Export analysis results for further review.

## Technologies Used
- **Python** – Data processing and analysis
- **Streamlit** – Interactive web application
- **Pandas & NumPy** – Data manipulation
- **yfinance** – Historical stock market data
- **Matplotlib & mplfinance** – Financial charts and candlestick visualizations
- **Scikit-learn** – Data preprocessing and scaling
- **TensorFlow/Keras** – LSTM-based price prediction
- **Google Gemini API** – AI-generated stock analysis
- **Excel/PDF** – Exporting analysis results

## Key Features
- Fetch historical stock market data.
- Display candlestick charts and moving averages.
- Calculate technical indicators such as RSI, Bollinger Bands, SMA, and EMA.
- Generate AI-assisted market analysis.
- Predict potential stock price trends using an LSTM model.
- Export analysis reports in Excel and PDF formats.

## Installation and Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it on Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API credentials
Create a `.env` file or configure your API key using Streamlit secrets, according to your application code. Never commit API keys or passwords to GitHub.

### 5. Run the application
```bash
streamlit run "stock analyzer.py"
```

## Important Disclaimer
This project is intended for educational and analytical purposes only. Stock predictions and AI-generated insights are not guaranteed to be accurate and should not be treated as financial advice.

## Future Improvements
- Improve model evaluation and prediction accuracy.
- Add more technical indicators.
- Enhance interactive visualizations.
- Integrate additional market data sources.

## Author
Nikhil Bhardwaj

- GitHub:https://github.com/nik9999999/Level-up
- LinkedIn: www.linkedin.com/in/nikhil-bhardwaj-188762312
