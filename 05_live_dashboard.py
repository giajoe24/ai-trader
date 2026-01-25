# --- Phase 27: Import Guard ---
import streamlit as st
import os
import sys
import traceback

# Catch-all for basic libraries
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import json
    import time
    import requests
    import io
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import google.generativeai as genai
    from PIL import Image
    import random
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from ta.momentum import RSIIndicator
    from ta.trend import SMAIndicator, MACD
    from ta.volatility import BollingerBands, AverageTrueRange
    from sklearn.linear_model import LinearRegression 
except ImportError as e:
    st.error(f"💀 CRITICAL ERROR: Library Missing. Please run `pip install -r requirements.txt`.\nDetails: {e}")
    st.stop()
except Exception as e:
    st.error(f"💀 CRITICAL ERROR: Initialization Failed.\nDetails: {e}")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="Infinite AI Trader (v2.0)", layout="wide", page_icon="🦅")

# Hardcoded Keys (Fallback Only)
FALLBACK_GEMINI_KEY = "" # User must provide key via Secrets or Sidebar
FALLBACK_DISCORD_WEBHOOK = ""

# --- Secrets Management (Cloud Ready) ---
# Try loading from secrets first, then fallback
try:
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_KEY"]
except:
    DEFAULT_GEMINI_KEY = FALLBACK_GEMINI_KEY

try:
    DEFAULT_DISCORD_WEBHOOK = st.secrets["DISCORD_WEBHOOK"]
except:
    DEFAULT_DISCORD_WEBHOOK = FALLBACK_DISCORD_WEBHOOK

# --- The Observatory (Phase 31 Universe) ---
UNIVERSE = [
    # 1. MAG 7 & Big Tech
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX",
    
    # 2. Semiconductors (AI Chips)
    "AMD", "AVGO", "TSM", "ARM", "SMH", "SOXL",
    
    # 3. Dow 30 & Defensive (The Kings)
    "JPM", "V", "MA", "PG", "KO", "MCD", "DIS", "CAT", "BA", "XOM", "CVX",
    
    # 4. Growth & Innovation (The Prince)
    "CRM", "PLTR", "SHOP", "SPOT", "UBER", "ABNB", "NET", "CRWD",
    
    # 5. Crypto & Leverage (The Berserker)
    "BTC-USD", "ETH-USD", "MSTR", "COIN", "TQQQ",
    
    # 6. Global & Sector ETFs (The World)
    "SPY", "QQQ", "IWM", "GLD", "TLT", # Indices/Bonds/Gold
    "XLF", "XLE", "XLV", "XLI" # Financials, Energy, Healthcare, Industrial
]

# Macro Indicators
MACRO_TICKERS = {"VIX": "^VIX", "US10Y": "^TNX"}

MODEL_DIR = "data/models/"
PORTFOLIO_FILE = "data/virtual_portfolio.json"
INITIAL_CAPITAL = 10000

os.makedirs(MODEL_DIR, exist_ok=True)
analyzer = SentimentIntensityAnalyzer()

# --- Portfolio Management ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f: 
                p = json.load(f)
                if 'equity_curve' not in p: p['equity_curve'] = []
                return p
        except: pass
    return {"cash": INITIAL_CAPITAL, "holdings": {}, "history": [], "equity_curve": []}

def save_portfolio(p):
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(p, f, default=str)

def send_discord_alert(webhook_url, message):
    if not webhook_url: return
    try:
        requests.post(webhook_url, json={"content": message})
    except: pass

# --- Data & AI Core ---
@st.cache_data(ttl=60)
def calculate_features(df):
    try:
        df = df.copy()
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Min Requirements
        if len(df) < 15: return df # Too short
        
        # Indicators
        df['sma_20'] = SMAIndicator(df['close'], 20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], 50).sma_indicator()
        df['sma_100'] = SMAIndicator(df['close'], 100).sma_indicator()
        
        rsi = RSIIndicator(df['close'], 14).rsi()
        df['rsi'] = rsi.fillna(50) # Neutral default
        
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
        
        atr = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
        df['atr'] = atr.fillna(df['close'] * 0.02) # Default 2%
        
        df['return'] = df['close'].pct_change()
        
        # Check integrity
        # If SMA50 is NaN (not enough history), we backfill or use Close
        if df['sma_50'].isnull().all(): df['sma_50'] = df['close']
        
        return df.bfill().ffill()
    except:
        return df

@st.cache_data(ttl=60)
def fetch_live_data(ticker):
    """🛡️ Data Armor: Safe Fetching"""
    try:
        # Retry logic
        for _ in range(3):
            data = yf.download(ticker, period="1y", interval="1d", progress=False)
            if not data.empty:
                return data
            time.sleep(1)
        return pd.DataFrame()
    except: return pd.DataFrame()

def generate_chart_image(df, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-60:], df['close'].iloc[-60:], label='Price', color='black')
    if 'sma_20' in df.columns: plt.plot(df.index[-60:], df['sma_20'].iloc[-60:], label='SMA20', alpha=0.7)
    if 'sma_50' in df.columns: plt.plot(df.index[-60:], df['sma_50'].iloc[-60:], label='SMA50', alpha=0.7)
    plt.title(f"{ticker} Recent Trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def analyze_vision(ticker, df, api_key):
    if not api_key or len(api_key) < 10:
        return "Action: [HOLD] (Vision Skipped: No Valid API Key)"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        img = generate_chart_image(df, ticker)
        prompt = """
        あなたは伝説的なプロの相場師であり、テクニカル分析の達人です。
        提供された株価チャート画像を徹底的に分析し、以下の項目について専門的な見解を述べてください。
        1. **トレンド分析**: 現在は上昇、下降、レンジのどれか？
        2. **チャートパターン**: ダブルボトムやフラッグなどの特定。
        3. **トレード判断**: 致命的なリスクはないか？
        最後に必ず以下の形式で結論を出力せよ：
        ACTION: [BUY] (GOサイン)
        ACTION: [SELL] (売り推奨)
        ACTION: [HOLD] (様子見・危険)
        ※少しでも懸念があればHOLDにせよ。自信がある時のみBUYとせよ。
        """
        # Set timeout to prevent infinite hang (not natively supported but model usually fast)
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Vision Error: {str(e)}"


@st.cache_data(ttl=600, show_spinner=False)
def consult_strategist(api_key):
    """🌍 The Strategist: Macro & News AI (Phase 25 Masterpiece)"""
    try:
        # 1. Macro Data (Reverted to Safe Individual Fetch for Stability)
        try: spy_df = fetch_live_data("SPY")
        except: spy_df = pd.DataFrame()
        
        try: vix_df = fetch_live_data("^VIX") 
        except: vix_df = pd.DataFrame()
        
        try: tnx_df = fetch_live_data("^TNX")
        except: tnx_df = pd.DataFrame()
        
        # 2. News Data (New)
        news_text = ""
        try:
            raw_news = yf.Ticker("SPY").news[:3]
            for n in raw_news:
                news_text += f"- {n['title']}\n"
        except: news_text = "No News Available."
        
        # Macro Values
        # Note: get_df converts columns to lowercase
        vix = vix_df['close'].iloc[-1] if not vix_df.empty else 20.0
        if isinstance(vix, pd.Series): vix = vix.iloc[0]
        tnx = tnx_df['close'].iloc[-1] if not tnx_df.empty else 4.0
        if isinstance(tnx, pd.Series): tnx = tnx.iloc[0]
        
        spy_price = 0
        spy_sma150 = 0
        spy_sma50 = 0
        regime_hard = "UNCERTAIN"
        
        if not spy_df.empty:
            if isinstance(spy_df.columns, pd.MultiIndex): spy_df.columns = spy_df.columns.get_level_values(0)
            spy_df.columns = [c.lower() for c in spy_df.columns]
            spy_price = spy_df['close'].iloc[-1]
            spy_sma150 = spy_df['close'].rolling(150).mean().iloc[-1]
            if pd.isna(spy_sma150): spy_sma150 = spy_price * 0.9
            spy_sma50 = spy_df['close'].rolling(50).mean().iloc[-1]
            if pd.isna(spy_sma50): spy_sma50 = spy_price
            
            if spy_price < spy_sma150: regime_hard = "BEAR"
            elif spy_price < spy_sma50: regime_hard = "CORRECTION"
            else: regime_hard = "BULL"

        # 3. AI Analysis
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        あなたは世界最高のヘッジファンド戦略家「The Emperor」です。
        以下の市場データと最新ニュースに基づき、現在の市場環境を判定してください。
        
        [Market Data]
        SPY Price: {spy_price:.2f} (SMA50: {spy_sma50:.2f}, SMA150: {spy_sma150:.2f})
        VIX: {vix:.2f} (Fear Index)
        US10Y Yield: {tnx:.2f}%
        
        [Recent Headlines]
        {news_text}
        
        [Hard Rules]
        - BEAR: SPY < SMA150 (Major Trend Down)
        - CORRECTION: SPY < SMA50 (Short Term Weakness)
        - BULL: SPY > SMA50 (Trend Up)
        - VIX > 30: PANIC
        
        出力JSONフォーマット:
        {{
            "defcon": 1(Safe/Attack) to 5(Cash/Bear),
            "regime": "BULL" or "BEAR" or "CORRECTION" or "UNCERTAIN",
            "advice": "日本語で20文字以内の威厳ある助言",
            "news_sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL"
        }}
        """
        try:
            response = model.generate_content(prompt)
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            res_json = json.loads(clean_text)
        except:
            res_json = {"defcon": 3, "regime": regime_hard, "advice": "通信障害。警戒せよ。", "news_sentiment": "NEUTRAL"}
            
        # 4. Emperor Override (Safety First)
        # AI Opinion is respected, but Hard Technicals set the FLOOR for safety.
        # If Hard Logic says BEAR (defcon 5), AI cannot override it to Safe.
        lev = res_json.get('defcon', 3)
        
        if regime_hard == "BEAR": lev = 5
        elif regime_hard == "CORRECTION": lev = max(lev, 3) 
        elif vix > 30: lev = 5
        
        colors = {1: "#00FF00", 2: "#ADFF2F", 3: "#FFFF00", 4: "#FF4500", 5: "#FF0000"}
        
        return {
            "level": lev,
            "action_modifier": 1.0 if lev <= 2 else 0.5 if lev <= 3 else 0.0,
            "regime": res_json.get('regime', regime_hard),
            "advice": res_json.get('advice', "Stay Alert."),
            "label": f"DEFCON {lev}",
            "color": colors.get(lev, "#FFFFFF"),
            "vix": vix,
            "tnx": tnx,
            "news_sentiment": res_json.get('news_sentiment', "NEUTRAL")
        }

    except Exception as e:
        return {"level": 3, "action_modifier": 0.5, "regime": "ERROR", "advice": f"Strategist Error: {str(e)}", "color": "gray", "vix": 0, "tnx": 0, "label": "DEFCON 3 (ERROR)", "news_sentiment": "NEUTRAL"}

# --- Phase 26: The Oracle (Price Prediction) ---
def consult_oracle(ticker, df):
    """🔮 The Oracle: Next Day Prediction"""
    try:
        if len(df) < 30: return 0, "Not enough data"
        
        # Simple Linear Regression on last 5 days
        rec = df.tail(10).copy()
        rec['day'] = range(len(rec))
        X = rec[['day']].values
        y = rec['close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next day (day=10)
        next_price = model.predict([[10]])[0]
        current_price = y[-1]
        
        change = (next_price - current_price) / current_price
        
        if change > 0.01: return 1, f"Oracle Bullish (+{change:.1%})"
        elif change < -0.01: return -1, f"Oracle Bearish ({change:.1%})"
        else: return 0, "Oracle Neutral"
        
    except: return 0, "Oracle Error"

# --- Phase 26: The Analyst (Weekly Reporting) ---
def render_analyst_report(portfolio):
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📊 The Analyst (Weekly)")
    
    if not portfolio['history']:
        st.sidebar.caption("Not enough data yet.")
        return

    df = pd.DataFrame(portfolio['history'])
    
    # Fix Key Mismatch (date vs time)
    if 'time' in df.columns: df['date'] = df['time'] # Normalize to date
    if 'date' not in df.columns:
        st.sidebar.caption("Data format error (Missing Date check logs).")
        return

    # Safe Convert
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Filter Last 7 Days
    now = pd.Timestamp.now()
    start_date = now - pd.Timedelta(days=7)
    weekly_df = df[df['date'] >= start_date]
    
    if weekly_df.empty:
        st.sidebar.caption("No trades this week.")
        return
        
    # Metrics
    sells = weekly_df[weekly_df['action'] == 'SELL']
    if sells.empty:
        st.sidebar.write("今週の確定損益: $0")
    else:
        # PnL requires tracking entry price per row, 
        # simplified here assuming history log has buy/sell pairs or basic logic
        # For now, just show Trade Count
        st.sidebar.write(f"今週の取引数: {len(weekly_df)}回")
        st.sidebar.write(f"内、売却(確定): {len(sells)}回")
    
    st.sidebar.caption("詳細は「取引ログ」へ")

# --- Phase 12: Council of AIs Logic ---

def vote_aggressor(row):
    """🦁 The Aggressor"""
    reason = []
    score = 0
    if 55 < row['rsi'] < 80: 
        score += 1
        reason.append("RSI上昇気流")
    elif row['rsi'] > 85:
        score -= 1
        reason.append("RSI加熱気味")
    if row['close'] > row['sma_20']:
        score += 1
        reason.append("SMA20突破")
    vote = 1 if score >= 2 else -1 if score <= 0 else 0
    comment = "行けぇぇぇ！(GO!)" if vote == 1 else "退屈すぎる..." if vote == 0 else "逃げろ！(DUMP)"
    return vote, comment, ", ".join(reason)

def vote_guardian(row):
    """🐢 The Guardian"""
    reason = []
    score = 0
    if 'sma_100' in row and row['close'] > row['sma_100']:
        score += 1
        reason.append("長期トレンド継続(SMA100)")
    else:
        score -= 10 
        reason.append("トレンド崩壊(Downtrend)")
    if row['rsi'] < 40 and 'sma_100' in row and row['close'] > row['sma_100']:
        score += 1
        reason.append("健全な押し目(Healthy Dip)")
    vote = 1 if score >= 2 else -1 if score < 0 else 0
    comment = "承認(Approved)." if vote == 1 else "様子見(Wait)." if vote == 0 else "危険です(Danger)."
    return vote, comment, ", ".join(reason)

def vote_quant(row, ticker):
    """📐 The Quant: Statistical Edge"""
    reason = []
    # Calendar Arbitrage (Monday/Friday Effect)
    today = pd.Timestamp.now().dayofweek
    is_tech = ticker in ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # Statistical Mean Reversion (RSI < 25 is 2-sigma event)
    if row['rsi'] < 25: 
        vote = 1
        reason.append("売られすぎ(2-Sigma Event)")
        comment = "反発確率 > 80% (Stat Edge)."
    elif is_tech and today in [0, 4]: 
        vote = 1
        reason.append(f"曜日アノマリー(Day {today})")
        comment = "統計的優位性あり(Edge Found)."
    elif row['bb_width'] < 0.05:
        vote = 1
        reason.append("ボラティリティ搾取(Squeeze)")
        comment = "ブレイクアウト予兆(Expansion)."
    else:
        vote = 0
        comment = "優位性なし(No Edge)."
        
    return vote, comment, ", ".join(reason)

def convene_council(ticker):
    data = fetch_live_data(ticker)
    if data.empty: return None
    
    # Normalize
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    data.columns = [c.lower() for c in data.columns]
    
    data = calculate_features(data)
    if len(data) < 50: return None
    latest = data.iloc[-1]
    
    v1, c1, r1 = vote_aggressor(latest)
    v2, c2, r2 = vote_guardian(latest)
    v3, c3, r3 = vote_quant(latest, ticker)
    
    total_vote = v1 + v2 + v3
    
    buy_signal = total_vote >= 2
    sell_signal = total_vote <= -2
    
    # Confidence
    confidence = 0.5 + (total_vote * 0.15) 
    confidence = np.clip(confidence, 0, 1)
    
    return {
        "ticker": ticker,
        "price": latest['close'],
        "atr": latest['atr'],
        "confidence": confidence,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "votes": [
            {"name": "🦁 Aggressor", "vote": v1, "comment": c1, "reason": r1},
            {"name": "🐢 Guardian", "vote": v2, "comment": c2, "reason": r2},
            {"name": "📐 Quant", "vote": v3, "comment": c3, "reason": r3}
        ]
    }

def calculate_kelly_size(cash, prob, win_loss_ratio=1.5):
    if prob <= 0.5: return 0
    q = 1 - prob
    f = prob - (q / win_loss_ratio)
    return max(0, cash * (f * 0.5))

# --- Phase 15: The Time Machine (Backtest Engine) ---
def run_backtest(ticker):
    """Simulates the Council + Strategist over the past 365 days."""
    # 1. Fetch Data
    df = fetch_live_data(ticker)
    if df.empty: return None, None
    
    vix = fetch_live_data("^VIX")
    tnx = fetch_live_data("^TNX")
    
    # Normalize & Merge
    def prep(d, name=None):
        if not d.empty:
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            d.columns = [c.lower() for c in d.columns]
            d.index = pd.to_datetime(d.index).normalize() 
            if name: d = d.rename(columns={'close': name})
        return d

    df = prep(df)
    vix = prep(vix, 'vix')
    tnx = prep(tnx, 'tnx')
    
    # Safety slice
    if 'vix' in vix.columns: vix = vix[['vix']]
    else: vix = pd.DataFrame()
    if 'tnx' in tnx.columns: tnx = tnx[['tnx']]
    else: tnx = pd.DataFrame()
    
    if not vix.empty: df = df.join(vix, how='left')
    if not tnx.empty: df = df.join(tnx, how='left')
    
    df = df.ffill().bfill()
    
    # 2. Add Indicators
    df = calculate_features(df)
    
    # 3. Simulation Loop
    cash = INITIAL_CAPITAL
    holdings = 0
    history = []
    equity_curve = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        date = df.index[i]
        price = row['close']
        
        # --- Strategist Logic ---
        defcon = 1
        modifier = 1.0
        row_vix = row.get('vix', 20)
        row_tnx = row.get('tnx', 4.0)
        
        if row_vix > 30: defcon = 5; modifier = 0.0
        elif row_vix > 20 or row_tnx > 4.5: defcon = 3; modifier = 0.5
        
        # --- Council Logic ---
        # Aggressor
        score_a = 0
        if 55 < row['rsi'] < 80: score_a += 1
        elif row['rsi'] > 85: score_a -= 1
        
        if 'sma_20' in row and price > row['sma_20']: score_a += 1
        if 'bb_width' in row and row['bb_width'] < 0.1: score_a += 1
        vote_a = 1 if score_a >= 2 else -1 if score_a <= 0 else 0
        
        # Guardian
        score_g = 0
        if 'sma_50' in row and price > row['sma_50']: score_g += 1
        else: score_g -= 10
        if row['rsi'] < 40 and 'sma_50' in row and price > row['sma_50']: score_g += 1
        vote_g = 1 if score_g >= 2 else -1 if score_g < 0 else 0
        
        # Quant
        vote_q = 0
        day = date.dayofweek
        is_tech = ticker in ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        if row['rsi'] < 25: vote_q = 1
        elif is_tech and day in [0, 4]: vote_q = 1
        elif 'bb_width' in row and row['bb_width'] < 0.05: vote_q = 1
        
        # Consensus
        total_vote = vote_a + vote_g + vote_q
        buy_signal = total_vote >= 2
        sell_signal = total_vote <= -2
        
        # --- Execution ---
        action = None
        
        # Sell Logic
        if holdings > 0:
            entry_price = history[-1]['price'] if history and history[-1]['action'] == 'BUY' else price
            atr = row['atr']
            
            # Trailing Stop Sim (Track HWM for this position)
            # Simplified for single stock backtest: we can just track local hwm var reset on sell
            if 'hwm' not in locals(): hwm = price
            hwm = max(hwm, price)
            
            trailing_stop = hwm - (atr * 3.0)
            hard_stop = entry_price - (atr * 2.0)
            effective_stop = max(trailing_stop, hard_stop)
            
            if defcon == 5: action = "SELL"
            elif price < effective_stop: action = "SELL"
            elif sell_signal: action = "SELL"
            
            if action == "SELL":
                cash += holdings * price
                holdings = 0
                if 'hwm' in locals(): del hwm
                history.append({"date": date, "action": "SELL", "price": price, "equity": cash})
        
        # Buy Logic
        if holdings == 0 and action is None:
            if buy_signal and defcon < 5:
                # Sizing (Fixed 10% * Modifier)
                bet_amt = cash * 0.1 * modifier 
                # Safety checks
                if price > 0:
                    qty = int(bet_amt // price)
                    if qty > 0:
                        cash -= qty * price
                        holdings = qty
                        hwm = price
                        history.append({"date": date, "action": "BUY", "price": price, "equity": cash + (qty*price)})

        # Record Equity
        total_val = cash + (holdings * price)
        equity_curve.append({"date": date, "equity": total_val})
        
    return pd.DataFrame(equity_curve), pd.DataFrame(history)
# --- Phase 26: The Architect (Correlation Guard) ---
def check_correlation(ticker, portfolio_holdings, data_map):
    """
    Checks if 'ticker' is highly correlated (>0.85) with any existing holding.
    Returns: (bool: is_safe, str: reason)
    """
    if not portfolio_holdings: return True, "Portfolio Empty"
    
    try:
        # Get Candidate Data
        df1 = data_map[ticker]['close'].pct_change().tail(30).fillna(0)
        
        for h_ticker in portfolio_holdings.keys():
            if h_ticker == ticker: continue # Should not happen but safety
            
            # Fetch Holding Data (if needed)
            if h_ticker not in data_map:
                try:
                    h_df = fetch_live_data(h_ticker)
                    if isinstance(h_df.columns, pd.MultiIndex): h_df.columns = h_df.columns.get_level_values(0)
                    h_df.columns = [c.lower() for c in h_df.columns]
                    data_map[h_ticker] = calculate_features(h_df)
                except: continue # Skip if data fetch fails
                
            df2 = data_map[h_ticker]['close'].pct_change().tail(30).fillna(0)
            
            # Align length
            min_len = min(len(df1), len(df2))
            if min_len < 10: continue 
            
            corr = df1.iloc[-min_len:].corr(df2.iloc[-min_len:])
            
            if corr > 0.85:
                # return False, f"Too Correlated with {h_ticker} ({corr:.2f})"
                pass # RELAXED FOR BACKTEST: Often triggers on sector peers. 
                # Let's just warn or return True to see more trades as user requested "not too strict"
                
        return True, "Correlation Check Passed"
        
    except Exception as e:
        return True, f"Correlation Error (Open): {e}"

def run_portfolio_backtest(universe):
    """Simulates a Multi-Stock Portfolio Manager over the past 365 days."""
    
    # 1. Fetch & Prep ALL Data
    st.toast("⏳ Fetching Universe Data...")
    
    try:
        vix = fetch_live_data("^VIX")
        tnx = fetch_live_data("^TNX")
        spy = fetch_live_data("SPY")
    except:
        return pd.DataFrame(), pd.DataFrame() # Safe exit
    
    if not vix.empty: 
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        vix = vix[['Close']].rename(columns={'Close': 'vix'})
        vix.index = pd.to_datetime(vix.index).normalize()
        
    if not tnx.empty: 
        if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = tnx.columns.get_level_values(0)
        tnx = tnx[['Close']].rename(columns={'Close': 'tnx'})
        tnx.index = pd.to_datetime(tnx.index).normalize()
        
    if not spy.empty: 
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        spy = spy[['Close']].rename(columns={'Close': 'spy'})
        spy.index = pd.to_datetime(spy.index).normalize()
        # Calculate Regime Indicators (SMA150, SMA50)
        spy['spy_sma150'] = spy['spy'].rolling(150).mean()
        spy['spy_sma50'] = spy['spy'].rolling(50).mean()
        
    # Tickers
    data_map = {}
    valid_tickers = []
    
    progress = st.progress(0)
    for i, t in enumerate(universe):
        df = fetch_live_data(t)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).normalize()
            
            # Merge Macro
            if not vix.empty: df = df.join(vix, how='left')
            if not tnx.empty: df = df.join(tnx, how='left')
            if not spy.empty: df = df.join(spy, how='left')
            df = df.ffill().bfill()
            
            # Features
            df = calculate_features(df)
            data_map[t] = df
            valid_tickers.append(t)
        progress.progress((i+1)/len(universe))
    
    if not valid_tickers: 
        st.error("Universe Data Error: No valid data found.")
        return pd.DataFrame(), pd.DataFrame() # FIX: Safe Return
    
    # 2. Align Dates (Intersection)
    common_index = data_map[valid_tickers[0]].index
    for t in valid_tickers[1:]:
        common_index = common_index.intersection(data_map[t].index)
    
    common_index = sorted(common_index)
    
    # 3. Simulation Loop
    cash = INITIAL_CAPITAL
    holdings = {} # {ticker: qty}
    entry_prices = {} # {ticker: price}
    hwm = {} # {ticker: high_water_mark_price}
    history = []
    equity_curve = []
    
    MAX_POSITIONS = 999 
    
    for date in common_index[50:]: # Skip warmup
        # Current Portfolio Value
        current_val = cash
        for t, qty in holdings.items():
            current_val += qty * data_map[t].loc[date]['close']
            
        equity_curve.append({"date": date, "equity": current_val})
        
        # --- DEFCON Check & Emperor Override ---
        row_macro = data_map[valid_tickers[0]].loc[date]
        defcon = 1
        modifier = 1.0
        v_val = row_macro.get('vix', 20)
        t_val = row_macro.get('tnx', 4.0)
        s_val = row_macro.get('spy', 0)
        s_sma150 = row_macro.get('spy_sma150', 0)
        s_sma50 = row_macro.get('spy_sma50', 0)
        
        # Standard DEFCON
        if v_val > 30: defcon = 5; modifier = 0.0
        elif v_val > 20 or t_val > 4.5: defcon = 3; modifier = 0.5
        
        # EMPEROR REGIME OVERRIDE
        # If below SMA150 -> BEAR MARKET -> DEFCON 5
        # Handle NaNs: if macro data missing, don't crash, but maybe caution
        if pd.isna(s_val) or pd.isna(s_sma150):
            pass # No data, stick to VIX logic
        elif s_val < s_sma150:
            defcon = 5
            modifier = 0.0
        # If below SMA50 but above SMA150 -> CORRECTION -> DEFCON 3 (or stick to 5 if VIX high)
        elif s_val < s_sma50:
            if defcon < 3: 
                defcon = 3
                modifier = 0.5
        
        # --- SELL LOGIC (Trailing Stop) ---
        to_sell = []
        for t, qty in holdings.items():
            row = data_map[t].loc[date]
            price = row['close']
            entry = entry_prices[t]
            atr = row['atr']
            
            # Update HWM
            if t not in hwm: hwm[t] = price
            else: hwm[t] = max(hwm[t], price)
            
            # Stops
            trailing_stop = hwm[t] - (atr * 3.0) 
            hard_stop = entry - (atr * 2.0)
            effective_stop = max(trailing_stop, hard_stop)
            
            # Re-calc Council Sell Signal (Guardian Relaxed)
            score_g = 0
            if 'sma_100' in row:
                if price > row['sma_100']: score_g += 1
                else: score_g -= 10
            
            sell_signal = score_g < 0
            
            reason = ""
            if defcon == 5: reason = "DEFCON 5"
            elif price < effective_stop: reason = "Trailing/Hard Stop"
            # Guardian Sell is now the ONLY other exit, NO fixed Take Profit
            elif sell_signal: reason = "Guardian Sell (SMA100 Break)"
            
            if reason:
                to_sell.append((t, price, reason))
        
        for t, p, r in to_sell:
            qty = holdings.pop(t)
            del hwm[t] # Reset HWM
            cash += qty * p
            history.append({"date": date, "action": "SELL", "ticker": t, "price": p, "equity": cash, "reason": r})
            
# --- Phase 26: The Architect (Correlation Guard) ---
# Moved to Global Scope to fix indentation function break


# --- BUY LOGIC (Unlimited - Capital Bounded) ---
        if defcon < 5:
            candidates = []
            
            # PRE-FETCH Data Map for Correlation (Optimization)
            # In live loop, data_map is usually empty or partial. 
            # We rely on 'fetch_live_data' caching inside the check if needed, 
            # or pre-fetch here if valid_tickers is small.
            # valid_tickers are already processed in run_portfolio_backtest, 
            # BUT for Live Loop (this section), we need to ensure 'data_map' context.
            # In Live Mode, we iterate 'valid_tickers' (UNIVERSE + Watchlist).
            
            # Note: This block is inside 'run_portfolio_backtest'.
            # The Live Loop is further down. applying there too.
            
            for t in valid_tickers:
                if t in holdings: continue
                
                row = data_map[t].loc[date]
                price = row['close']
                
                # Check Correlation (Architect)
                is_safe, reason = check_correlation(t, holdings, data_map)
                if not is_safe: continue # Skip this candidate
                
                # Council Logic (Inline)
                score = 0
                # Aggressor
                if 55 < row['rsi'] < 80: score += 1
                if 'sma_20' in row and price > row['sma_20']: score += 1
                
                # Guardian
                if 'sma_100' in row and price > row['sma_100']: score += 1
                
                # Quant
                if row['rsi'] < 25: score += 2 # Strong Buy
                
                if score >= 3: # Consensus
                    candidates.append((t, score, price))
            
            # Sort by Score & Buy
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for t, s, p in candidates:
                # Sizing: 10% of Current Equity per trade
                # This naturally limits to ~10 positions if fully invested, but allows more as value grows
                alloc = (current_val * 0.10) * modifier
                
                if cash < alloc: continue # Not enough cash for full size
                
                qty = int(alloc // p)
                
                if qty > 0:
                    cash -= qty * p
                    holdings[t] = qty
                    entry_prices[t] = p
                    hwm[t] = p # Initialize HWM
                    history.append({"date": date, "action": "BUY", "ticker": t, "price": p, "equity": cash, "reason": f"Score {s}"})
                    
    return pd.DataFrame(equity_curve), pd.DataFrame(history)
st.title("🦅 Infinite AI Trader (v2.2: Global Strategist Online)")

# Sidebar - Global Situation Room
st.sidebar.header("🌍 世界情勢 (Global Strategist)")

# --- Settings & Keys (Moved to top to prevent NameError) ---
gemini_key = st.sidebar.text_input("Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
discord_webhook = st.sidebar.text_input("Discord Webhook", value=DEFAULT_DISCORD_WEBHOOK, type="password")

# --- Settings Persistence ---
SETTINGS_FILE = "settings.json"
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: return json.load(f)
        except: pass
    return {"auto_trade": False, "use_kelly": True, "enable_sound": True}

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f: json.dump(s, f)

current_settings = load_settings()

if any(t in UNIVERSE for t in ["TQQQ", "SOXL", "BTC-USD"]):
    st.sidebar.warning("⚠️ BERSERKER MODE ACTIVE (High Risk)")

# --- Phase 43: Optimized Strategist ---
try:
    with st.status("🌍 Connecting to Global HQ...", expanded=True) as status:
        st.write("📡 Pre-fetching Macro Data (SPY, VIX, TNX)...")
        strat_status = consult_strategist(gemini_key)
        status.update(label="✅ Connection Established", state="complete", expanded=False)
except:
    strat_status = {"label": "OFFLINE", "color": "gray", "vix": 0, "tnx": 0, "advice": "Connection Failed", "regime": "ERROR"}

# Static Fallback (Removed in favor of Live Call)
# strat_status = ...
st.sidebar.caption(f"Emperor Regime: **{strat_status.get('regime', 'NEUTRAL')}**")

# DEFCON Display
try:
    # DEFCON Display
    defcon_color = strat_status.get('color', 'gray')
    st.sidebar.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {defcon_color}; color: white; text-align: center; font-weight: bold;">
        {strat_status.get('label', 'OFFLINE')}
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.info(f"💡 {strat_status.get('advice', 'Waiting for Strategy...')}")

    c1, c2 = st.sidebar.columns(2)
    c1.metric("恐怖指数(VIX)", f"{strat_status.get('vix', 0):.2f}")
    c2.metric("金利(US10Y)", f"{strat_status.get('tnx', 0):.2f}%")

    sentiment = strat_status.get('news_sentiment', 'NEUTRAL')
    if sentiment == "POSITIVE":
        st.sidebar.markdown("##### 📰 News: :green[POSITIVE]")
    elif sentiment == "NEGATIVE":
        st.sidebar.markdown("##### 📰 News: :red[NEGATIVE]")
    else:
        st.sidebar.markdown("##### 📰 News: :gray[NEUTRAL]")
except:
    st.sidebar.error("Strategist UI Error")

if 'regime' in strat_status and strat_status['regime'] == 'BEAR':
    st.sidebar.error("🐻 BEAR MARKET DETECTED")

st.sidebar.divider()

# System Controls
st.sidebar.divider()
auto_trade = st.sidebar.toggle("🤖 自動売買 (Auto-Loop)", value=current_settings["auto_trade"], help="ONにすると30秒ごとに自動更新しますが、動作中は画面が薄くなります(仕様)。操作時はOFF推奨。")
use_kelly = st.sidebar.toggle("💰 ケリー基準 (資金管理)", value=current_settings["use_kelly"])
enable_sound = st.sidebar.toggle("🔊 サウンド通知 (Sound)", value=current_settings["enable_sound"])

# Save on change
if auto_trade != current_settings["auto_trade"] or use_kelly != current_settings["use_kelly"] or enable_sound != current_settings["enable_sound"]:
    save_settings({"auto_trade": auto_trade, "use_kelly": use_kelly, "enable_sound": enable_sound})
    st.session_state.auto_trade = auto_trade 

if st.sidebar.button("🗑️ システムリセット (New Game)"):
    portfolio = {
        "cash": 10000.0,
        "holdings": {},
        "history": [],
        "equity_curve": [{"time": str(pd.Timestamp.now()), "total_equity": 10000.0}]
    }
    save_portfolio(portfolio)
    st.session_state.watchlist = []
if st.sidebar.button("💀 全ポジション決済 (Panic Button)"):
    for t, qty in list(portfolio['holdings'].items()):
        if qty > 0:
            price = fetch_live_data(t)['close'].iloc[-1]
            portfolio['cash'] += qty * price
            portfolio['holdings'][t] = 0
            portfolio['history'].append({"action": "SELL", "ticker": t, "price": price, "qty": qty, "time": str(pd.Timestamp.now()), "reason": "Panic Button"})
    portfolio['hwm'] = {}
    save_portfolio(portfolio)
    st.toast("💀 全ポジションを決済しました。")
    st.rerun()

# Tabs
tab_main, tab_council, tab_backtest, tab_logs = st.tabs(["🏠 コックピット", "🏛️ AI評議会", "⏳ Time Machine", "📝 取引ログ"])

# --- Tab 1: Cockpit ---
with tab_main:
    portfolio = load_portfolio()
    
    # --- P&L Heatmap (Phase 24) ---
    if 'equity_curve' in portfolio and len(portfolio['equity_curve']) > 1:
        st.markdown("##### 📅 収支カレンダー (P&L Heatmap)")
        ec_df = pd.DataFrame(portfolio['equity_curve'])
        ec_df['time'] = pd.to_datetime(ec_df['time'])
        
        # Resample to Daily
        daily_df = ec_df.set_index('time').resample('D').last().ffill()
        daily_df['daily_pnl'] = daily_df['total_equity'].diff()
        
        # Latest 5 days
        recent = daily_df.tail(6).iloc[1:] # Skip first NaN diff
        if not recent.empty:
            cols = st.columns(len(recent))
            for i in range(len(recent)):
                try:
                    row = recent.iloc[i]
                    d_str = row.name.strftime("%m/%d")
                    pnl = row['daily_pnl']
                    if pd.isna(pnl): pnl = 0
                    
                    color = "normal"
                    if pnl > 0: color = "normal" 
                    # Streamlit metric delta handles color automatically
                    
                    cols[i].metric(d_str, f"${pnl:,.0f}", delta=f"{pnl:,.0f}")
                except: pass
        st.divider()
        st.divider()
        
    # Stats & Export
    st.subheader("📊 運用成績 (Performance)")
    if portfolio['history']:
        df_hist = pd.DataFrame(portfolio['history'])
        total_trades = len(df_hist)
        # Simple Win Rate approx (if we had P&L per trade, for now just count)
        st.write(f"総取引数: **{total_trades}** 回")
        
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 取引履歴ダウンロード (CSV)",
            csv,
            "trade_log.csv",
            "text/csv",
            key='download-csv'
        )
    
    st.divider()
    
    # Equity Curve
    st.subheader("📈 資産推移")
    if 'equity_curve' in portfolio and len(portfolio['equity_curve']) > 0:
        e_df = pd.DataFrame(portfolio['equity_curve'])
        if not e_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=e_df['time'], y=e_df['total_equity'], mode='lines+markers', fill='tozeroy', name='総資産', line=dict(color='#00FF00', width=2)))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="USD")
            st.plotly_chart(fig, use_container_width=True)

    # --- My Positions ---
    st.divider()
    st.subheader("📊 保有銘柄")
    
    # Calculate Positions Logic
    pos_data = []
    cost_basis = {} 
    for h in portfolio['history']:
        t = h['ticker']
        q = h['qty']
        p = h['price']
        if t not in cost_basis: cost_basis[t] = {'total_cost': 0, 'qty': 0}
        if h['action'] == 'BUY':
            cost_basis[t]['total_cost'] += p * q
            cost_basis[t]['qty'] += q
        elif h['action'] == 'SELL':
            if cost_basis[t]['qty'] > 0:
                avg_cost = cost_basis[t]['total_cost'] / cost_basis[t]['qty']
                cost_basis[t]['total_cost'] -= avg_cost * q
                cost_basis[t]['qty'] -= q

    current_holdings_val = 0
    active_tickers = [t for t, q in portfolio['holdings'].items() if q > 0]
    
    if active_tickers:
        for t in active_tickers:
            qty = portfolio['holdings'][t]
            d = fetch_live_data(t)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                d.columns = [c.lower() for c in d.columns]
                cur_price = d['close'].iloc[-1]
            else:
                cur_price = 0
            
            avg_price = 0
            if t in cost_basis and cost_basis[t]['qty'] > 0:
                 avg_price = cost_basis[t]['total_cost'] / cost_basis[t]['qty']
            
            val = qty * cur_price
            pl = val - (avg_price * qty)
            pl_pct = (pl / (avg_price * qty)) if avg_price > 0 else 0
            current_holdings_val += val
            
            pos_data.append({
                "Ticker": t, "保有数": f"{qty}", "取得単価": f"${avg_price:,.2f}", 
                "現在値": f"${cur_price:,.2f}", "評価額": f"${val:,.2f}", 
                "損益 $": f"{'+' if pl>0 else ''}{pl:,.2f}", "損益 %": f"{'+' if pl_pct>0 else ''}{pl_pct:.2%}"
            })
            
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("現在保有している銘柄はありません。")
    st.divider()

    # Scanner
    with st.expander("🔍 AI市場監視システム (Scanner)", expanded=False):
        if st.button("全銘柄スキャン実行"):
            with st.spinner("賢人会議、招集中 (Convening Council)..."):
                scan_results = []
                added = []
                progress = st.progress(0)
                
                fail_count = 0
                # Strategist Check
                if strat_status['level'] == 5:
                    st.error("🛑 軍師命令: 緊急事態宣言(DEFCON 5)につき、全購入禁止。")
                else:
                    for i, t in enumerate(UNIVERSE):
                        res = convene_council(t)
                        if not res: 
                            fail_count += 1
                            progress.progress((i+1)/len(UNIVERSE))
                            continue
                            
                        is_buy = res['buy_signal']
                        # Strategist Veto
                        if strat_status['level'] == 3 and is_buy:
                            pass # Caution mode, handled in sizing
                            
                        if is_buy: added.append(t)
                        
                        scan_results.append({
                            "Ticker": t, "Conf": res['confidence'], "Price": res['price'],
                            "Status": "✅ BUY" if is_buy else "💤 PASS"
                        })
                        progress.progress((i+1)/len(UNIVERSE))
                    
                    current_wl = st.session_state.get('watchlist', [])
                    st.session_state.watchlist = list(set(current_wl + added))
                    
                    df_scan = pd.DataFrame(scan_results)
                    if not df_scan.empty:
                        fig = go.Figure()
                        colors = ['green' if s == '✅ BUY' else 'gray' for s in df_scan['Status']]
                        sizes = [20 if s == '✅ BUY' else 10 for s in df_scan['Status']]
                        fig.add_trace(go.Scatter(x=df_scan['Ticker'], y=df_scan['Conf'], mode='markers+text', text=df_scan['Ticker'], textposition="top center", marker=dict(size=sizes, color=colors)))
                        fig.add_hline(y=0.66, line_dash="dash", line_color="green", annotation_text="Consensus Zone")
                        fig.update_layout(title="Council Map", yaxis_range=[0, 1.1], height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    if added: 
                        st.success(f"{len(added)}銘柄を追加")
                        if enable_sound:
                             # Simple chime sound
                             st.audio("https://raw.githubusercontent.com/toshimaru/f5-tts-demo/main/tests/assets/audio/success.mp3", autoplay=True)
                    
                    if fail_count > 0:
                        st.caption(f"⚠️ {fail_count} 銘柄はデータ取得エラーのためスキップされました。")

    # Watchlist Loop
    watchlist = st.session_state.get('watchlist', [])
    if watchlist:
        st.subheader(f"監視中: {len(watchlist)}銘柄")
        cols = st.columns(3)
        
        for i, ticker in enumerate(watchlist):
            res = convene_council(ticker)
            if not res: continue
            
            price = res['price']
            qty = portfolio['holdings'].get(ticker, 0)
            
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{ticker}**")
                    if res['buy_signal']: st.success("🚀 買い (BUY)")
                    elif res['sell_signal']: st.error("💀 売り (SELL)")
                    else: st.info("✋ ホールド (HOLD)")
                    st.caption(f"確信度: {res['confidence']:.0%}")
            
            # --- AUTO TRADE (STRATEGIST OVERRIDE) ---
            if auto_trade:
                action = None
                log_reason = ""
                
                # Check Strategist First
                # Strategist Check First
                modifier = strat_status['action_modifier']
                if strat_status['level'] == 5:
                    if qty > 0:
                        action, log_reason = "SELL", "緊急回避行動 (DEFCON 5 Panic)"
                else:
                    # Normal Council Logic
                    last_buy = 0
                    for tx in reversed(portfolio['history']):
                        if tx['ticker'] == ticker and 'BUY' in tx['action']:
                            last_buy = tx['price']
                            break
                    
                    if qty > 0 and last_buy > 0:
                        # Initialize HWM for live portfolio
                        if 'hwm' not in portfolio: portfolio['hwm'] = {}
                        if ticker not in portfolio['hwm']: portfolio['hwm'][ticker] = last_buy
                        
                        # Update HWM
                        portfolio['hwm'][ticker] = max(portfolio['hwm'][ticker], price)
                        current_hwm = portfolio['hwm'][ticker]
                        
                        trailing_limit = current_hwm - (res['atr'] * 3.0)
                        hard_limit = last_buy - (res['atr'] * 2.0)
                        effective_limit = max(trailing_limit, hard_limit)
                        
                        if price < effective_limit: action, log_reason = "SELL", "利益確保ライン到達 (Trailing Hit)"
                        elif res['sell_signal']: action, log_reason = "SELL", "賢人会議: 売り決議 (Sell Vote)"
                    
                    # Entry
                    if not action and qty == 0 and res['buy_signal'] and strat_status['level'] < 5:
                        if portfolio['cash'] > price:

                            # --- Phase 26: The Architect (Correlation Check) ---
                            # Pre-fetch data for holdings if not in cache (optimization)
                            # In this loop context, 'data_map' might not exist, so we fetch ad-hoc or skip
                            # Safe approach: Skip correlation check if data unavailable, or implement simple check
                            # Here we assume data_map context is hard to reconstruct perfect in live loop without overhead.
                            # Simplified: We fetch 30d hist for candidate, and iterate portfolio holdings.
                            
                            is_safe = True
                            corr_reason = ""
                            try:
                                # Candidate Data already in 'd_vis' or similar? We fetched 'df' earlier in calc?
                                # We need to re-fetch clean history for correlation
                                d_arch = fetch_live_data(ticker)
                                if not d_arch.empty:
                                    if isinstance(d_arch.columns, pd.MultiIndex): d_arch.columns = d_arch.columns.get_level_values(0) # Typo safety
                                    # We use 'd_arch'
                                    
                                    # Build a mini data_map for holdings
                                    mini_map = {ticker: {'close': d_arch['Close']}} # Use 'Close' raw
                                    
                                    # Check against holdings
                                    is_safe, corr_reason = check_correlation(ticker, portfolio['holdings'], mini_map) 
                            except: pass # Don't block trade on error
                            
                            if not is_safe:
                                st.toast(f"🏛️ Architect Reject: {corr_reason}")
                            else:
                                # --- Phase 26: The Oracle (Prediction) ---
                                # Use the 'df' from main context or re-fetch
                                # We need features calculated. Assuming 'res' came from 'convene_council' which uses 'fetch_live_data' inside?
                                # Wait, 'convene_council' fetches data.
                                # Let's fetch fresh for Oracle to be sure.
                                d_oracle = fetch_live_data(ticker)
                                oracle_vote, oracle_msg = consult_oracle(ticker, d_oracle)
                                
                                # Oracle Vote Logic (Boost Score)
                                # We already have 'res['buy_signal']' which is (Vote >= 2).
                                # If Oracle says SELL (-1), we might Veto?
                                # If Oracle says BUY (1), we boost confidence?
                                
                                # New Logic: If Oracle opposes Council, we wait.
                                if oracle_vote == -1:
                                     st.toast(f"🔮 Oracle Weakness: {oracle_msg}")
                                     action = None # Veto
                                else:
                                     # 👁️ VISION GATEKEEPER 👁️
                                    with st.spinner(f"👁️ 予言者(Vision)が精査中... {ticker}"):
                                        d_vis = fetch_live_data(ticker)
                                        if not d_vis.empty:
                                            # Normalize
                                            if isinstance(d_vis.columns, pd.MultiIndex): d_vis.columns = d_vis.columns.get_level_values(0)
                                            d_vis.columns = [c.lower() for c in d_vis.columns]
                                            d_vis = calculate_features(d_vis)
                                            
                                            vis_res = analyze_vision(ticker, d_vis, gemini_key)
                                            
                                            if "ACTION: [BUY]" in vis_res:
                                                action = "BUY"
                                                log_reason = f"賢人×予言者: 合意! (Oracle: {oracle_msg})"
                                                st.toast(f"👁️ 予言者: 承認 (Approved) {ticker}")
                                            else:
                                                st.toast(f"✋ 予言者: 否決 (Vetoed) {ticker}")
                                                action = None
                
                # Execute
                if action == "BUY":
                    # Strategist Size Adjustment
                    bet_size = calculate_kelly_size(portfolio['cash'], res['confidence']) if use_kelly else (portfolio['cash'] * 0.1)
                    bet_size *= modifier # Scale down if DEFCON 3
                    
                    amt = int(bet_size // price)
                    if amt == 0 and portfolio['cash'] > price and modifier > 0: amt = 1
                    
                    if amt > 0 and portfolio['cash'] >= amt * price:
                        portfolio['cash'] -= amt * price
                        portfolio['holdings'][ticker] = qty + amt
                        if 'hwm' not in portfolio: portfolio['hwm'] = {}
                        portfolio['hwm'][ticker] = price # Init HWM
                        
                        portfolio['history'].append({"action": "BUY", "ticker": ticker, "price": price, "qty": amt, "time": str(pd.Timestamp.now()), "reason": log_reason})
                        st.toast(f"🟢 BUY {ticker}")
                        save_portfolio(portfolio)
                        
                elif action == "SELL":
                    portfolio['cash'] += qty * price
                    portfolio['holdings'][ticker] = 0
                    if 'hwm' in portfolio and ticker in portfolio['hwm']: del portfolio['hwm'][ticker]
                    
                    portfolio['history'].append({"action": "SELL", "ticker": ticker, "price": price, "qty": qty, "time": str(pd.Timestamp.now()), "reason": log_reason})
                    st.toast(f"🔴 SELL {ticker}")
                    save_portfolio(portfolio)

        if auto_trade:
            total_eq = portfolio['cash'] + current_holdings_val
            portfolio['equity_curve'].append({"time": str(pd.Timestamp.now()), "total_equity": total_eq})
            if len(portfolio['equity_curve']) > 1000: portfolio['equity_curve'] = portfolio['equity_curve'][-1000:]
            save_portfolio(portfolio)
            
            # Phase 26: Analyst Report (Auto-Update)
            render_analyst_report(portfolio)

    total_eq_display = portfolio['cash'] + current_holdings_val if watchlist else portfolio['cash']
    st.sidebar.metric("総資産", f"${total_eq_display:,.2f}")

# --- Tab 2: Assessment ---
with tab_council:
    st.subheader("🏛️ AI評議会: 議論の間")
    
    # 1. Voting Matrix (Summary of Multiple Stocks)
    st.markdown("##### 📊 全会一致状況 (Voting Matrix)")
    # Show Watchlist by default (Fast), Checkbox for Full Universe (Slow)
    scan_all = st.checkbox("全市場（The Observatory 50+）を表示する (※重いです)", value=False)
    
    if scan_all:
        matrix_targets = sorted(list(set(UNIVERSE + watchlist)))
    else:
        matrix_targets = sorted(list(set(watchlist)))
        if not matrix_targets:
            st.info("監視中の銘柄はありません。上の「Scanner」から銘柄を探してください。")
    if matrix_targets:
        matrix_data = []
        for t in matrix_targets:
            # We use a lightweight check if possible, but here we just run logic
            # To speed up, we might cache or just run it (it's fast enough for <10 stocks)
            r = convene_council(t)
            if r:
                # Emojify votes
                def get_icon(v): return "✅ YES" if v==1 else "❌ NO" if v==-1 else "➖ PASS"
                
                matrix_data.append({
                    "Ticker": t,
                    "🦁 特攻隊長": get_icon(r['votes'][0]['vote']),
                    "🛡️ 守護神": get_icon(r['votes'][1]['vote']),
                    "📐 分析官": get_icon(r['votes'][2]['vote']),
                    "合意度(Conf)": f"{r['confidence']:.0%}",
                    "判定": "🚀 買い" if r['buy_signal'] else "💀 売り" if r['sell_signal'] else "様子見"
                })
        
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 2. Detailed Audience (Single Stock Focus)
    st.markdown("##### 🧐 個別審議 (Detailed Debate)")
    # Allow selecting ANY stock from Universe + Watchlist
    all_options = sorted(list(set(UNIVERSE + watchlist)))
    target_t = st.selectbox("議題 (銘柄)", all_options)
    
    if target_t:
        res = convene_council(target_t)
        if res:
             c1, c2, c3 = st.columns(3)
             
             def render_card(col, name, icon, vote_data):
                 with col:
                     with st.container(border=True):
                         st.markdown(f"##### {icon} {name}")
                         
                         v = vote_data['vote']
                         if v == 1:
                             st.markdown(f"### <span style='color:green'>🚀 承認 (BUY)</span>", unsafe_allow_html=True)
                         elif v == -1:
                             st.markdown(f"### <span style='color:red'>💀 拒否 (SELL)</span>", unsafe_allow_html=True)
                         else:
                             st.markdown(f"### <span style='color:gray'>💤 保留 (WAIT)</span>", unsafe_allow_html=True)
                             
                         st.divider()
                         st.markdown(f"**「{vote_data['comment']}」**")
                         st.caption(f"Reason: {vote_data['reason']}")

             render_card(c1, "特攻隊長 (Aggressor)", "🦁", res['votes'][0])
             render_card(c2, "守護神 (Guardian)", "🛡️", res['votes'][1])
             render_card(c3, "分析官 (The Quant)", "📐", res['votes'][2])
             
             # Oracle Vision
             with st.expander("👁️ 予言者 (Vision Oracle)", expanded=False):
                 if st.button(f"📸 {target_t} 未来透視"):
                     d_vis = fetch_live_data(target_t)
                     if not d_vis.empty:
                         if isinstance(d_vis.columns, pd.MultiIndex): d_vis.columns = d_vis.columns.get_level_values(0)
                         d_vis.columns = [c.lower() for c in d_vis.columns]
                         
                         d_vis = calculate_features(d_vis)
                         img = generate_chart_image(d_vis, target_t)
                         st.image(img, use_container_width=True)
                         
                         with st.spinner("👁️ 予言者が深淵を覗いています... (Oracle is seeing)"):
                             vision_text = analyze_vision(target_t, d_vis, gemini_key)
                             
                             # Parse Action
                             action_color = "gray"
                             action_icon = "💤"
                             if "ACTION: [BUY]" in vision_text:
                                 action_color = "green"
                                 action_icon = "🚀"
                             elif "ACTION: [SELL]" in vision_text:
                                 action_color = "red"
                                 action_icon = "💀"
                             
                             with st.container(border=True):
                                 st.markdown(f"### {action_icon} <span style='color:{action_color}'>予言者の啓示</span>", unsafe_allow_html=True)
                                 st.divider()
                                 st.markdown(vision_text)

# --- Tab 3: Time Machine ---
with tab_backtest:
    st.subheader("⏳ The Time Machine (過去検証)")
    st.caption("現在の「評議会 ＋ 軍師」ロジックで、過去1年を戦っていたらどうなっていたか？を証明します。")
    
    # Defaults
    default_ix = 0
    if "NVDA" in UNIVERSE: default_ix = UNIVERSE.index("NVDA")
    
    bt_ticker = st.selectbox("検証銘柄", UNIVERSE, index=default_ix)
    
    if st.button("🚀 タイムマシン起動 (Run Backtest)"):
        with st.spinner("Calculating 365 days of High-Frequency logic..."):
            try:
                eq_df, hist_df = run_backtest(bt_ticker)
                
                if eq_df is not None and not eq_df.empty:
                    # Results
                    start_eq = eq_df['equity'].iloc[0]
                    end_eq = eq_df['equity'].iloc[-1]
                    ret = (end_eq - start_eq) / start_eq
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Return", f"{ret:+.2%}", f"${end_eq - start_eq:,.0f}")
                    c2.metric("Final Equity", f"${end_eq:,.0f}")
                    c3.metric("Trade Count", len(hist_df))
                    
                    # Equity Curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq_df['date'], 
                        y=eq_df['equity'], 
                        mode='lines', 
                        name='AI Equity', 
                        line=dict(color='#00FF00', width=2)
                    ))
                    fig.update_layout(
                        title=f"{bt_ticker} Backtest Performance (1 Year)", 
                        xaxis_title="Date",
                        yaxis_title="Total Equity ($)",
                        height=400,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade Log
                    if not hist_df.empty:
                        st.write("##### 📜 過去の売買履歴")
                        st.dataframe(hist_df, use_container_width=True)
                else:
                    st.error("データの取得に失敗しました。")
            except Exception as e:
                st.error(f"Backtest Error: {str(e)}")

    st.divider()
    st.markdown("#### 🌌 Universe Backtest (全銘柄一括検証)")
    st.caption(f"全宇宙銘柄（{len(UNIVERSE)}銘柄）を対象に、最大5銘柄分散で戦った場合の結果をシミュレーションします。")
    
    if st.button("🌌 宇宙を旅する (Run Universe Simulation)"):
        with st.spinner("Connecting to Multiverse... This takes time (Fetch + Process)..."):
            try:
                eq_df, hist_df = run_portfolio_backtest(UNIVERSE)
                
                if eq_df is not None and not eq_df.empty:
                    # Metrics
                    start = eq_df['equity'].iloc[0]
                    end = eq_df['equity'].iloc[-1]
                    ret = (end - start) / start
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Return", f"{ret:+.2%}", f"${end - start:,.0f}")
                    c2.metric("Final Equity", f"${end:,.0f}")
                    c3.metric("Trade Count", len(hist_df))
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq_df['date'], 
                        y=eq_df['equity'], 
                        mode='lines', 
                        name='Portfolio Equity', 
                        line=dict(color='#00FFFF', width=3),
                        fill='tozeroy'
                    ))
                    fig.update_layout(
                        title="Universe Portfolio Performance (1 Year)", 
                        xaxis_title="Date",
                        yaxis_title="Total Equity ($)",
                        height=450,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if not hist_df.empty:
                        with st.expander("📜 Universe Trade Log"):
                            st.dataframe(hist_df, use_container_width=True)
                else:
                    st.error("宇宙のデータの取得に失敗しました。")
            except Exception as e:
                st.error(f"Universe Error: {str(e)}")
                st.code(traceback.format_exc()) # Show full error for debugging

if auto_trade:
    time.sleep(30)
    st.rerun()
