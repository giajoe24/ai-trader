import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def add_features(input_file, output_file):
    print(f"Loading data from {input_file}...")
    try:
        # Load with index_col=0. yfinance CSVs often have multiple header rows.
        # We'll clean it up by coercing the index to datetime and columns to numeric.
        df = pd.read_csv(input_file, index_col=0)
        
        # Clean Index: Convert to datetime, turn errors (like 'Ticker', 'Date' rows) into NaT
        df.index = pd.to_datetime(df.index, errors='coerce')
        # Drop rows with invalid index (metadata rows)
        df = df[df.index.notna()]
        
        # Clean Columns: Ensure they are lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure Numeric: Convert all columns to numeric, coercing errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        print(f"Data loaded and cleaned. Shape: {df.shape}")
        
    except Exception as e:
        print(f"Error loading or cleaning data: {e}")
        return 
    # yfinance often gives 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    # Sometimes multi-index if multiple tickers. Assuming single ticker for now.

    # 1. Moving Averages
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    
    # 2. RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # 3. MACD (Moving Average Convergence Divergence)
    from ta.trend import MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # 4. Bollinger Bands
    from ta.volatility import BollingerBands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # Distance from bands can be a feature
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']

    # 5. Returns & Lag Features
    df['return'] = df['close'].pct_change()
    # Lag 1: Return of yesterday
    df['return_lag1'] = df['return'].shift(1)
    # Lag 2: Return of 2 days ago
    df['return_lag2'] = df['return'].shift(2)

    # 6. Target Variable: Will price go UP significantly tomorrow?
    # Shift(-1) means looking at the NEXT row's return
    df['target_return'] = df['return'].shift(-1)
    
    # Threshold: meaningful gain (e.g., > 0.1% to cover costs/spread)
    # If we just predict > 0, we get too much noise.
    df['target'] = (df['target_return'] > 0.000).astype(int) # Keep 0 for now, but XGboost handles it better

    # Drop NaNs created by rolling windows and shift
    df.dropna(inplace=True)

    print("Features added:")
    print(df[['close', 'sma_20', 'rsi', 'target']].tail())

    df.to_csv(output_file)
    print(f"Feature-rich data saved to {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "data/stock_data.csv"
    OUTPUT_FILE = "data/stock_data_features.csv"
    add_features(INPUT_FILE, OUTPUT_FILE)
