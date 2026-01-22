import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import requests
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from xgboost import XGBClassifier
import datetime

# --- Configuration ---
# User Provided Keys
GEMINI_KEY = "AIzaSyBrhH_kjmuSFk2Gu__tkeBM7lMP6mXoXQ8"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1463190103248867465/lF0sIS7vRzmboaeHeVf_HAmpKqj_0amvQdeX7n08xJr3rf6zoplNuh5fWZX_7vrQi43m"

UNIVERSE = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD", "ETH-USD"]
MODEL_DIR = "data/models/"
PORTFOLIO_FILE = "data/virtual_portfolio.json"

# Settings
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.05
CHECK_INTERVAL_SECONDS = 300 # Check every 5 minutes

print("🤖 Headless AI Trader Starting...")
print(f"🔑 Gemini Key: {GEMINI_KEY[:5]}...")
print(f"📡 Discord: Connected")

analyzer = SentimentIntensityAnalyzer()

def send_discord(msg):
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    except:
        pass

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"cash": 10000, "holdings": {}, "history": []}

def save_portfolio(p):
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(p, f, default=str)

# --- Feature & Model Logic (Simplified duplicate of dashboard) ---
def get_signal(ticker):
    try:
        # Fetch Data
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if df.empty: return 0.5, 0, 0
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Features
        df['sma_20'] = SMAIndicator(df['close'], 20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], 50).sma_indicator()
        df['rsi'] = RSIIndicator(df['close'], 14).rsi()
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
        
        data = df.iloc[-1]
        
        # Simple Logic for Headless (Robustness)
        # If RSI < 30 and Price > SMA50 -> BUY Signal Strong
        score = 0.5
        if data['rsi'] < 30: score += 0.2
        elif data['rsi'] > 70: score -= 0.2
        
        if data['close'] > data['sma_50']: score += 0.1
        else: score -= 0.1
        
        return score, data['close'], data['atr']
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return 0.5, 0, 0

# --- Main Loop ---
if __name__ == "__main__":
    send_discord("🤖 **Headless Bot Started**: PC上で24時間監視を開始します。")
    
    while True:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Scanning Market...")
        
        p = load_portfolio()
        
        for ticker in UNIVERSE:
            score, price, atr = get_signal(ticker)
            if price == 0: continue
            
            # 1. Check Holdings (Risk Mgmt)
            qty = p['holdings'].get(ticker, 0)
            if qty > 0:
                # Calculate PnL
                last_buy = 0
                for tx in reversed(p['history']):
                    if tx['ticker'] == ticker and 'BUY' in tx['action']:
                        last_buy = tx['price']
                        break
                
                if last_buy > 0:
                    # Dynamic ATR Thresholds
                    sl_price = last_buy - (atr * 2.0)
                    tp_price = last_buy + (atr * 3.0)
                    
                    if price <= sl_price:
                        # STOP LOSS
                        pnl = (price - last_buy) / last_buy
                        rev = qty * price
                        p['cash'] += rev
                        p['holdings'][ticker] = 0
                        p['history'].append({"action": "AUTO-SELL (ATR-SL)", "ticker": ticker, "price": price, "qty": qty, "time": str(datetime.datetime.now())})
                        msg = f"⚠️ **ATR STOP-LOSS**: {ticker} Sold @ ${price:.2f} (Loss: {pnl:.1%})"
                        print(msg)
                        send_discord(msg)
                        save_portfolio(p)
                        continue

                    elif price >= tp_price:
                        # TAKE PROFIT
                        pnl = (price - last_buy) / last_buy
                        rev = qty * price
                        p['cash'] += rev
                        p['holdings'][ticker] = 0
                        p['history'].append({"action": "AUTO-SELL (ATR-TP)", "ticker": ticker, "price": price, "qty": qty, "time": str(datetime.datetime.now())})
                        msg = f"💰 **ATR TAKE-PROFIT**: {ticker} Sold @ ${price:.2f} (Profit: +{pnl:.1%})"
                        print(msg)
                        send_discord(msg)
                        save_portfolio(p)
                        continue

            # 2. Check Buy Signal (Only if not holding)
            if qty == 0 and score > 0.7:
                # Budget Check
                if p['cash'] > price:
                    # Buy max 10% of cash
                    buy_amt = int((p['cash'] * 0.1) // price)
                    if buy_amt > 0:
                        cost = buy_amt * price
                        p['cash'] -= cost
                        p['holdings'][ticker] = buy_amt
                        p['history'].append({"action": "AUTO-BUY", "ticker": ticker, "price": price, "qty": buy_amt, "time": str(datetime.datetime.now())})
                        
                        msg = f"🟢 **AUTO-BUY**: {ticker} {buy_amt}shares @ ${price:.2f} (Score: {score:.0%})"
                        print(msg)
                        send_discord(msg)
                        save_portfolio(p)

        print(f"Cycle complete. Waiting {CHECK_INTERVAL_SECONDS}s...")
        time.sleep(CHECK_INTERVAL_SECONDS)
