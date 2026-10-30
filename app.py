import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Inteligentní Trading Asistent", layout="wide")

st.title("🛡️ Risk-Aware XGBoost Trading Asistent")
st.write("Model zaměřený na zachycení silných trendů s vysokým poměrem zisku k riziku (RRR 1:2).")

# --- SIDEBAR: WATCHLIST & RIZIKO ---
st.sidebar.header("📋 Tvůj digitální Watchlist")

watchlist = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Alphabet / Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta Platforms (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Broadcom (AVGO)": "AVGO",
    "Advanced Micro Devices (AMD)": "AMD",
    "Netflix (NFLX)": "NFLX",
    "Walmart (WMT)": "WMT",
    "JPMorgan Chase (JPM)": "JPM",
    "Visa (V)": "V",
    "Eli Lilly (LLY)": "LLY",
    "Caterpillar (CAT)": "CAT"
}

vybrany_nazev = st.sidebar.selectbox("Vyber akcii k analýze:", list(watchlist.keys()))
ticker_akcie = watchlist[vybrany_nazev]

st.sidebar.markdown("---")
st.sidebar.header("💰 Nastavení XTB Účtu")
kapital = st.sidebar.number_input("Zůstatek na tvém obchodním účtu ($):", min_value=100.0, value=2000.0, step=100.0)
max_risk_procenta = st.sidebar.slider("Maximální povolená ztráta na jeden obchod (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

st.header(f"Aktuální analýza: {vybrany_nazev}")

def ziskej_sentiment_zprav(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        zpravy = ticker_obj.news
        if not zpravy:
            return 0.0
        analyzer = SentimentIntensityAnalyzer()
        skore_sentimentu = []
        for zprava in zpravy[:5]:
            titulek = zprava.get('title', '')
            vysledek = analyzer.polarity_scores(titulek)
            skore_sentimentu.append(vysledek['compound'])
        return float(np.mean(skore_sentimentu)) if skore_sentimentu else 0.0
    except Exception:
        return 0.0

def priprav_data_trend(ticker, perioda="8y"):
    try:
        df = yf.download(ticker, period=perioda, multi_level_index=False, progress=False)
        spy = yf.download("^GSPC", period=perioda, multi_level_index=False, progress=False)
        
        if df.empty or spy.empty:
            return None, 0.0
        
        df.columns = [str(col).strip() for col in df.columns]
        spy.columns = [str(col).strip() for col in spy.columns]
        
        # S&P 500 Kontext
        df['SPY_Close'] = spy['Close']
        df['SPY_Rocni_Trend'] = spy['Close'].rolling(window=200).mean()
        df['Dist_SPY_Trend'] = (df['SPY_Close'] - df['SPY_Rocni_Trend']) / df['SPY_Close']
        
        # Volatilita (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
        
        # Fibonacci (50 dní)
        okno_fib = 50
        df['High_50'] = df['High'].rolling(window=okno_fib).max()
        df['Low_50'] = df['Low'].rolling(window=okno_fib).min()
        df['Fib_Range'] = df['High_50'] - df['Low_50']
        df['Fib_618'] = df['High_50'] - (df['Fib_Range'] * 0.618)
        df['Fib_382'] = df['High_50'] - (df['Fib_Range'] * 0.382)
        
        df['Dist_Fib_618'] = (df['Close'] - df['Fib_618']) / df['Close']
        df['Dist_Fib_382'] = (df['Close'] - df['Fib_382']) / df['Close']
        df['ATR_Percent'] = df['ATR'] / df['Close']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # Sentiment
        zivy_sentiment = ziskej_sentiment_zprav(ticker)
        df['Sentiment'] = np.where(df['Close'] > df['Close'].shift(1), zivy_sentiment * 0.8 + 0.1, zivy_sentiment * 0.8 - 0.1)
        df.loc[df.index[-1], 'Sentiment'] = zivy_sentiment
        
        # --- NOVÝ CÍL (TARGET): Hledáme silný růstový impuls v příštích 5 dnech ---
        # 1 = cena během 5 dnů vzroste o více než 1.5x násobek ATR (silný trend)
        budouci_max = df['High'].shift(-5).rolling(window=5).max()
        df['Target'] = (budouci_max > (df['Close'] + (df['ATR'] * 1.5))).astype(int)
        
        df.dropna(inplace=True)
        return df, zivy_sentiment
    except Exception:
        return None, 0.0

data, aktualni_sentiment = priprav_data_trend(ticker_akcie, perioda="8y")

if data is not None and not data.empty:
    features = ['Dist_Fib_382', 'Dist_Fib_618', 'RSI', 'ATR_Percent', 'Dist_SPY_Trend', 'Sentiment']
    X = data[features]
    y = data['Target']

    X_dnes = X.tail(1)
    X_historie = X.iloc[:-1]
    y_historie = y.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X_historie, y_historie, test_size=0.2, shuffle=False)

    # Optimalizace pro vyhledávání vzácnějších silných trendů
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    historicka_presnost = accuracy_score(y_test, y_pred_test) * 100

    raw_pravdepodobnost = model.predict_proba(X_dnes)
    pravdepodobnost_rustu = float(pd.DataFrame(raw_pravdepodobnost).iloc[0, 1]) * 100

    aktualni_cena = float(data['Close'].iloc[-1])
    atr = float(data['ATR'].iloc[-1])
    
    # --- ASYMETRICKÉ ŘÍZENÍ PŘÍKAZU (RRR 1:2) ---
    stop_loss = aktualni_cena - (atr * 1.0)    # Riskujeme pouze 1x běžný denní pohyb
    take_profit = aktualni_cena + (atr * 2.0)  # Cílíme na 2x větší zisk
    
    max_dolarovy_risk = kapital * (max_risk_procenta / 100.0)
    vzdalenost_sl = abs(aktualni_cena - stop_loss)
    doporuceny_objem_loty = max_dolarovy_risk / vzdalenost_sl

    # Slovní vyhodnocení filtru
    if pravdepodobnost_rustu >= 60.0:
        smer_text = "NÁKUP (BUY) 📈"
        barva_boxu = st.success
        pokyn_text = f"👉 **CO ZADAT V XTB:** Otevři BUY příkaz, nastav objem na **{doporuceny_objem_loty:.2f} Lotů**, zadej **SL: {stop_loss:.2f}** a **TP: {take_profit:.2f}**."
    else:
        smer_text = "ČEKEJ (ŽÁDNÝ SIGNÁL) ⏳"
        barva_boxu = st.info
        pokyn_text = "❌ **Doporučení:** Dnes chybí silná shoda indikátorů. Do obchodu nevstupuj a počkej na jasnější trend."

    st.subheader(f"📋 Pokyny pro ruční zadání do XTB ({ticker_akcie} CFD)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Doporučený příkaz", value=smer_text)
        st.metric(label="Aktuální cena", value=f"{aktualni_cena:.2f} USD")
    with col2:
        st.metric(label="Pravděpodobnost silného růstu", value=f"{pravdepodobnost_rustu:.1f} %")
        st.metric(label="Denní volatilita (ATR)", value=f"{atr:.2f} USD")
    with col3:
        st.metric(label="Přesnost záchytu trendů", value=f"{historicka_presnost:.1f} %", 
                  help="Úspěšnost modelu při správném označení začátku silného pohybu v minulosti.")

    barva_boxu(f"🛡️ **MANAGEMENT RIZIKA:** Pokud systém vygeneruje signál, riskuješ maximálně **{max_dolarovy_risk:.2f} USD**.")
    st.markdown(f"<div style='background-color:#f0f2f6;padding:15px;border-radius:5px;'>{pokyn_text}</div>", unsafe_value=True)

    # --- GRAF ---
    fig, ax = plt.subplots(figsize=(10, 3.5))
    posledni_dny = data.tail(60)
    ax.plot(posledni_dny.index, posledni_dny['Close'], color='black', label=f'Cena {ticker_akcie}', linewidth=2)
    ax.axhline(y=stop_loss, color='red', linestyle=':', linewidth=2, label=f'SL ({stop_loss:.2f})')
    ax.axhline(y=take_profit, color='green', linestyle=':', linewidth=2, label=f'TP ({take_profit:.2f})')
    ax.set_title(f"Technická mapa pro {ticker_akcie}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.error("Nepodařilo se načíst data.")
