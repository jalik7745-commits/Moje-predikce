import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Inteligentní Skener Trhu", layout="wide")

st.title("🛡️ Automatický XGBoost Skener (TOP 30 Akcií)")
st.write("Live přehled 30 nejvhodnějších amerických mega-caps akcií na jedno kliknutí.")

# --- DIGITÁLNÍ WATCHLIST: 30 NEJVHODNĚJŠÍCH AMERICKÝCH AKCIÍ ---
watchlist = {
    "Abbott Laboratories": "ABT",
    "Advanced Micro Devices": "AMD",
    "Adobe Inc.": "ADBE",
    "Alphabet / Google": "GOOGL",
    "Amazon.com Inc.": "AMZN",
    "Apple Inc.": "AAPL",
    "Berkshire Hathaway": "BRK-B",
    "Broadcom Inc.": "AVGO",
    "Caterpillar Inc.": "CAT",
    "Costco Wholesale": "COST",
    "Eli Lilly & Co.": "LLY",
    "Exxon Mobil Corp.": "XOM",
    "Home Depot Inc.": "HD",
    "Chevron Corp.": "CVX",
    "Intel Corp.": "INTC",
    "JPMorgan Chase & Co.": "JPM",
    "Mastercard Inc.": "MA",
    "Meta Platforms": "META",
    "Microsoft Corp.": "MSFT",
    "Netflix Inc.": "NFLX",
    "NVIDIA Corp.": "NVDA",
    "Oracle Corp.": "ORCL",
    "PepsiCo Inc.": "PEP",
    "Pfizer Inc.": "PFE",
    "Procter & Gamble": "PG",
    "Qualcomm Inc.": "QCOM",
    "Salesforce Inc.": "CRM",
    "Tesla Inc.": "TSLA",
    "Visa Inc.": "V",
    "Walmart Inc.": "WMT"
}

# --- SIDEBAR CONFIG ---
st.sidebar.header("💰 Nastavení XTB Účtu")
kapital = st.sidebar.number_input("Zůstatek na účtu ($):", min_value=100.0, value=2000.0, step=100.0)
max_risk_procenta = st.sidebar.slider("Maximální risk na obchod (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

def ziskej_sentiment_zprav(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        zpravy = ticker_obj.news
        if not zpravy: return 0.0
        analyzer = SentimentIntensityAnalyzer()
        skore = [analyzer.polarity_scores(z.get('title', ''))['compound'] for z in zpravy[:3]]
        return float(np.mean(skore)) if skore else 0.0
    except Exception: return 0.0

@st.cache_data(ttl=3600) # Data se uloží na 1 hodinu pro bleskovou rychlost
def analyzuj_akcii(ticker):
    try:
        df = yf.download(ticker, period="3y", multi_level_index=False, progress=False)
        spy = yf.download("^GSPC", period="3y", multi_level_index=False, progress=False)
        if df.empty or spy.empty: return None
        
        df.columns = [str(col).strip() for col in df.columns]
        spy.columns = [str(col).strip() for col in spy.columns]
        
        df['SPY_Close'] = spy['Close']
        df['SPY_Rocni_Trend'] = spy['Close'].rolling(window=200).mean()
        df['Dist_SPY_Trend'] = (df['SPY_Close'] - df['SPY_Rocni_Trend']) / df['SPY_Close']
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
        
        okno_fib = 50
        df['High_50'] = df['High'].rolling(window=okno_fib).max()
        df['Low_50'] = df['Low'].rolling(window=okno_fib).min()
        df['Fib_Range'] = df['High_50'] - df['Low_50']
        df['Fib_618'] = df['High_50'] - (df['Fib_Range'] * 0.618)
        df['Fib_382'] = df['High_50'] - (df['Fib_Range'] * 0.382)
        
        df['Dist_Fib_618'] = (df['Close'] - df['Fib_618']) / df['Close']
        df['Dist_Fib_382'] = (df['Close'] - df['Fib_382']) / df['Close']
        df['ATR_Percent'] = df['ATR'] / df['Close']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        z_sent = ziskej_sentiment_zprav(ticker)
        df['Sentiment'] = np.where(df['Close'] > df['Close'].shift(1), z_sent * 0.8 + 0.1, z_sent * 0.8 - 0.1)
        df.loc[df.index[-1], 'Sentiment'] = z_sent
        
        budouci_max = df['High'].shift(-5).rolling(window=5).max()
        df['Target'] = (budouci_max > (df['Close'] + (df['ATR'] * 1.5))).astype(int)
        df.dropna(inplace=True)
        
        features = ['Dist_Fib_382', 'Dist_Fib_618', 'RSI', 'ATR_Percent', 'Dist_SPY_Trend', 'Sentiment']
        X = df[features]
        y = df['Target']
        
        X_dnes = X.tail(1)
        X_hist = X.iloc[:-1]
        y_hist = y.iloc[:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X_hist, y_hist, test_size=0.2, shuffle=False)
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        prob = float(pd.DataFrame(model.predict_proba(X_dnes)).iloc[0, 1]) * 100
        
        return {
            'Cena': float(df['Close'].iloc[-1]), 'ATR': float(df['ATR'].iloc[-1]),
            'Fib_618': float(df['Fib_618'].iloc[-1]), 'Fib_382': float(df['Fib_382'].iloc[-1]),
            'High_50': float(df['High_50'].iloc[-1]), 'Low_50': float(df['Low_50'].iloc[-1]),
            'Presnost': acc, 'Pravdepodobnost': prob, 'Data完整': df
        }
    except Exception: return None

# --- HLAVNÍ BĚH SCANNERU ---
st.subheader("📊 Živý přehled celého trhu (Skenování 30 firem)")
vysledky_list = []

with st.spinner("🔄 Algoritmus skenuje 30 firem, index S&P 500 a živé zprávy..."):
    for jmeno, tkr in watchlist.items():
        res = analyzuj_akcii(tkr)
        if res:
            stav = "🟩 NÁKUP" if res['Pravdepodobnost'] >= 60.0 else "⬜ ČEKEJ"
            vysledky_list.append({
                "Akcie": jmeno, "Ticker": tkr, "Aktuální cena ($)": f"{res['Cena']:.2f}",
                "Pravděpodobnost růstu": res['Pravdepodobnost'], "Historická přesnost": res['Presnost'], "Doporučený příkaz": stav
            })

df_watchlist = pd.DataFrame(vysledky_list)
df_watchlist = df_watchlist.sort_values(by=["Doporučený příkaz", "Pravděpodobnost růstu"], ascending=[False, False])

df_watchlist["Pravděpodobnost růstu"] = df_watchlist["Pravděpodobnost růstu"].map("{:.1f} %".format)
df_watchlist["Historická přesnost"] = df_watchlist["Historická přesnost"].map("{:.1f} %".format)

st.dataframe(df_watchlist, use_container_width=True, hide_index=True)

st.markdown("---")
# --- DETAILNÍ POKYNY POD TABULKOU ---
st.subheader("🔍 Detailní instrukce a graf vybrané akcie")
vybrana_akcie_box = st.selectbox("Zvol akcii pro zobrazení grafu a přesných pokynů pro XTB:", list(watchlist.keys()))
ticker_vybrane = watchlist[vybrana_akcie_box]

res_detail = analyzuj_akcii(ticker_vybrane)
if res_detail:
    cena_d = res_detail['Cena']
    atr_d = res_detail['ATR']
    
    sl_d = cena_d - (atr_d * 1.0)
    tp_d = cena_d + (atr_d * 2.0)
    if res_detail['Pravdepodobnost'] < 60.0:
        sl_d = res_detail['Fib_618'] - atr_d
        tp_d = res_detail['High_50'] - (atr_d * 0.5)
        if sl_d >= cena_d: sl_d = cena_d - (atr_d * 1.5)

    risk_d = kapital * (max_risk_procenta / 100.0)
    loty_d = risk_d / abs(cena_d - sl_d)

    if res_detail['Pravdepodobnost'] >= 60.0:
        st.success(f"🔥 **SIGNÁL AKTIVNÍ:** V XTB otevři **BUY** příkaz na **{vybrana_akcie_box}**.")
        st.warning(f"👉 **NASTAVENÍ PRO XTB:** Objem: **{loty_d:.2f} Lotů** | STOP-LOSS: **{sl_d:.2f} USD** | TAKE-PROFIT: **{tp_d:.2f} USD**")
    else:
        st.info("⏳ Pro tuto akcii dnes platí pokyn **ČEKEJ**. Neriskuj peníze a vyčkej na zelený signál v tabulce výše.")

    # Graf
    fig, ax = plt.subplots(figsize=(10, 3.5))
    p_dny = res_detail['Data完整'].tail(60)
    ax.plot(p_dny.index, p_dny['Close'], color='black', label='Cena', linewidth=2)
    ax.plot(p_dny.index, p_dny['Fib_618'], color='purple', linestyle='--', alpha=0.4, label='Fib 61.8%')
    ax.plot(p_dny.index, p_dny['Fib_382'], color='orange', linestyle='--', alpha=0.4, label='Fib 38.2%')
    ax.axhline(y=sl_d, color='red', linestyle=':', label='SL')
    ax.axhline(y=tp_d, color='green', linestyle=':', label='TP')
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
else:
    st.error("Nepodařilo se načíst data.")
