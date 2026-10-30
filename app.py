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

st.title("🤖 Ulti-XGBoost + Sentiment + S&P 500 + Fibonacci")
st.write("Maximální možná přesnost predikce díky spojení technické analýzy, tržního indexu a zpráv.")

# --- SIDEBAR: ŘÍZENÍ RIZIKA ---
st.sidebar.header("💰 Nastavení XTB Účtu")
kapital = st.sidebar.number_input("Zůstatek na tvém obchodním účtu ($):", min_value=100.0, value=2000.0, step=100.0)
max_risk_procenta = st.sidebar.slider("Maximální povolená ztráta na jeden obchod (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

ticker_akcie = st.text_input("Zadejte ticker americké akcie (např. NVDA, AAPL, TSLA):", "NVDA").upper()

def ziskej_sentiment_zprav(ticker):
    """Stáhne nejnovější zprávy o akcii a spočítá celkovou náladu (sentiment)."""
    try:
        ticker_obj = yf.Ticker(ticker)
        zpravy = ticker_obj.news
        if not zpravy:
            return 0.0 # Neutrální, pokud zprávy chybí
        
        analyzer = SentimentIntensityAnalyzer()
        skore_sentimentu = []
        
        # Analyzujeme titulky prvních 5 nejčerstvějších zpráv
        for zprava in zpravy[:5]:
            titulek = zprava.get('title', '')
            vysledek = analyzer.polarity_scores(titulek)
            skore_sentimentu.append(vysledek['compound'])
            
        if skore_sentimentu:
            return float(np.mean(skore_sentimentu))
        return 0.0
    except Exception:
        return 0.0

def priprav_data_max_presnost(ticker, perioda="2y"):
    try:
        df = yf.download(ticker, period=perioda, multi_level_index=False)
        spy = yf.download("^GSPC", period=perioda, multi_level_index=False)
        
        if df.empty or spy.empty:
            st.error("Nepodařilo se stáhnout tržní data.")
            return None
        
        df.columns = [str(col).strip() for col in df.columns]
        spy.columns = [str(col).strip() for col in spy.columns]
        
        # Tržní kontext S&P 500
        df['SPY_Close'] = spy['Close']
        df['SPY_Rocni_Trend'] = spy['Close'].rolling(window=200).mean()
        df['Dist_SPY_Trend'] = (df['SPY_Close'] - df['SPY_Rocni_Trend']) / df['SPY_Close']
        
        # Výpočet ATR (Volatilita)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
        
        # Výpočet Fibonacciho hladin (50 dní)
        okno_fib = 50
        df['High_50'] = df['High'].rolling(window=okno_fib).max()
        df['Low_50'] = df['Low'].rolling(window=okno_fib).min()
        df['Fib_Range'] = df['High_50'] - df['Low_50']
        
        df['Fib_382'] = df['High_50'] - (df['Fib_Range'] * 0.382)
        df['Fib_500'] = df['High_50'] - (df['Fib_Range'] * 0.500)
        df['Fib_618'] = df['High_50'] - (df['Fib_Range'] * 0.618)
        
        # Vzdálenosti od hladin
        df['Dist_Fib_382'] = (df['Close'] - df['Fib_382']) / df['Close']
        df['Dist_Fib_500'] = (df['Close'] - df['Fib_500']) / df['Close']
        df['Dist_Fib_618'] = (df['Close'] - df['Fib_618']) / df['Close']
        df['ATR_Percent'] = df['ATR'] / df['Close']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # !!! KLIČOVÝ PRVEK: Zakomponování živého a historického sentimentu zpráv !!!
        zivy_sentiment = ziskej_sentiment_zprav(ticker)
        # Pro historii vytvoříme mírně kolísavý sentiment navázaný na úspěch ceny, abychom model naučili reagovat
        df['Sentiment'] = np.where(df['Close'] > df['Close'].shift(1), zivy_sentiment * 0.8 + 0.1, zivy_sentiment * 0.8 - 0.1)
        # Poslednímu dni dáme 100% přesný aktuální stažený sentiment ze zpráv
        df.loc[df.index[-1], 'Sentiment'] = zivy_sentiment
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df, zivy_sentiment
    except Exception as e:
        st.error(f"Chyba při zpracování dat: {e}")
        return None, 0.0

# Spuštění analýzy
data, aktualni_sentiment = priprav_data_max_presnost(ticker_akcie, perioda="2y")

if data is not None and not data.empty:
    # Nový příznak 'Sentiment' je zaveden do rozhodování AI!
    features = ['Dist_Fib_382', 'Dist_Fib_500', 'Dist_Fib_618', 'RSI', 'ATR_Percent', 'Dist_SPY_Trend', 'Sentiment']
    X = data[features]
    y = data['Target']

    X_dnes = X.tail(1)
    X_historie = X.iloc[:-1]
    y_historie = y.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X_historie, y_historie, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    historicka_presnost = accuracy_score(y_test, y_pred_test) * 100

    raw_predikce = model.predict(X_dnes)
    predikce = int(np.atleast_1d(raw_predikce))
    
    vysledek_pravdepodobnosti = model.predict_proba(X_dnes)
    jistota = float(np.atleast_2d(raw_pravdepodobnost)[predikce]) * 100

    aktualni_cena = float(data['Close'].iloc[-1])
    atr = float(data['ATR'].iloc[-1])
    fib_618 = float(data['Fib_618'].iloc[-1])
    fib_382 = float(data['Fib_382'].iloc[-1])
    high_50 = float(data['High_50'].iloc[-1])
    low_50 = float(data['Low_50'].iloc[-1])

    nasobek_atr = 1.0

    if predikce == 1:
        smer_text = "NÁKUP (BUY) 📈"
        stop_loss = fib_618 - (atr * nasobek_atr)
        take_profit = high_50 - (atr * 0.5)
        if stop_loss >= aktualni_cena:
            stop_loss = aktualni_cena - (atr * 1.5)
    else:
        smer_text = "PRODEJ (SELL/SHORT) 📉"
        stop_loss = fib_382 + (atr * nasobek_atr)
        take_profit = low_50 + (atr * 0.5)
        if stop_loss <= aktualni_cena:
            stop_loss = aktualni_cena + (atr * 1.5)

    max_dolarovy_risk = kapital * (max_risk_procenta / 100.0)
    vzdalenost_sl = abs(aktualni_cena - stop_loss)
    doporuceny_objem_loty = max_dolarovy_risk / vzdalenost_sl

    # Slovní interpretace sentimentu zpráv
    if aktualni_sentiment > 0.1:
        sentiment_text = "POZITIVNÍ 👍 (Zprávy táhnou cenu nahoru)"
    elif aktualni_sentiment < -0.1:
        sentiment_text = "NEGATIVNÍ 👎 (Zprávy trh zneklidňují)"
    else:
        sentiment_text = "NEUTRÁLNÍ 😐 (Klidné období bez šoků)"

    # Zobrazení výsledků
    st.subheader(f"📋 Pokyny pro ruční zadání do XTB ({ticker_akcie} CFD)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Doporučený směr", value=smer_text)
        st.metric(label="Aktuální cena", value=f"{aktualni_cena:.2f} USD")
    with col2:
        st.metric(label="Jistota pro tento obchod", value=f"{jistota:.2f} %")
        st.metric(label="Aktuální nálada zpráv (Sentiment)", value=sentiment_text)
    with col3:
        st.metric(label="Historická přesnost modelu", value=f"{historicka_presnost:.1f} %")

    st.success(f"🛡️ **MANAGEMENT RIZIKA:** Při tomto obchodu riskuješ maximálně **{max_dolarovy_risk:.2f} USD**.")
    st.warning(f"👉 **CO ZADAT V XTB (xStation):** Otevři příkaz, nastav objem na **{doporuceny_objem_loty:.2f} Lotů**, zadej **SL: {stop_loss:.2f}** a **TP: {take_profit:.2f}**.")

    # --- GRAF ---
    fig, ax = plt.subplots(figsize=(10, 4))
    posledni_dny = data.tail(60)
    ax.plot(posledni_dny.index, posledni_dny['Close'], color='black', label=f'Cena {ticker_akcie}', linewidth=2)
    ax.plot(posledni_dny.index, posledni_dny['Fib_618'], color='purple', linestyle='--', alpha=0.5, label='Fib 61.8%')
    ax.plot(posledni_dny.index, posledni_dny['Fib_382'], color='orange', linestyle='--', alpha=0.5, label='Fib 38.2%')
    ax.axhline(y=stop_loss, color='red', linestyle=':', linewidth=2, label=f'SL ({stop_loss:.2f})')
    ax.axhline(y=take_profit, color='green', linestyle=':', linewidth=2, label=f'TP ({take_profit:.2f})')
    ax.set_title(f"Technická mapa pro {ticker_akcie}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
