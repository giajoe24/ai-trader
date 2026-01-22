import os
import json
import requests
import yfinance as yf
import google.generativeai as genai
from datetime import datetime

print("🔍 SYSTEM DIAGNOSTIC TOOL v1.0")
print("=============================")

# 1. Connectivity
try:
    print("[1/6] Checking Internet...", end=" ")
    r = requests.get("https://www.google.com", timeout=5)
    if r.status_code == 200: print("✅ ONLINE")
    else: print("❌ HTTP ERROR")
except: print("❌ OFFLINE")

# 2. Dependencies
try:
    print("[2/6] Checking AI Models (sklearn)...", end=" ")
    import sklearn
    from sklearn.linear_model import LinearRegression
    print("✅ INSTALLED")
except ImportError: print("❌ MISSING (pip install scikit-learn)")

# 3. Market Data
try:
    print("[3/6] Checking yfinance (AAPL)...", end=" ")
    t = yf.Ticker("AAPL")
    hist = t.history(period="1d")
    if not hist.empty: print(f"✅ OK (Price: {hist['Close'].iloc[-1]:.2f})")
    else: print("❌ NO DATA")
except Exception as e: print(f"❌ ERROR: {e}")

# 4. Settings & Portfolio
print("[4/6] Checking Files...", end=" ")
settings_ok = os.path.exists("settings.json")
portfolio_ok = os.path.exists("data/virtual_portfolio.json")
if settings_ok and portfolio_ok: print(f"✅ OK (Settings & Portfolio found)")
elif settings_ok: print("⚠️ PARTIAL (Portfolio missing - New Install?)")
else: print("⚠️ MISSING (Will be created on launch)")

# 5. Neural Network (Gemini)
print("[5/6] Checking Gemini AI...", end=" ")
try:
    # Read key from 05_live_dashboard.py (hacky but effective)
    with open("05_live_dashboard.py", "r", encoding='utf-8') as f:
        content = f.read()
        # Find DEFAULT_GEMINI_KEY
        import re
        match = re.search(r'DEFAULT_GEMINI_KEY\s*=\s*"([^"]+)"', content)
        if match:
            key = match.group(1)
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            res = model.generate_content("Say 'OK' in 2 chars.")
            if "OK" in res.text: print("✅ CONNECTED")
            else: print(f"⚠️ RESPONSE: {res.text.strip()}")
        else:
            print("⚠️ Key Not Found in Code")
except Exception as e: print(f"❌ ERROR: {e}")

# 6. News Feed
try:
    print("[6/6] Checking News Feed...", end=" ")
    news = yf.Ticker("SPY").news
    if len(news) > 0: print("✅ OK")
    else: print("⚠️ EMPTY (Might be after hours or blocked)")
except: print("❌ FAILED")

print("\n=============================")
print("System Verification Complete.")
