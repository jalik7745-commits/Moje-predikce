import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Inteligentní Trading Asistent", layout="wide")

st.title("🤖 Pokročilý XGBoost + S&P 500 + Fibonacci Trading Asistent")
st.write("Maximalizace zisku a automatické řízení rizika pro americké CFD kontrakty.")

# --- SIDEBAR: ŘÍZENÍ RIZIKA (PENÍZE A KAPITÁL) ---
st.sidebar.header("💰 Nastavení XTB Účtu")
kapital = st.sidebar.number_input("Zůstatek na tvém obchodním účtu ($):", min_value=100.0, value=2000.0, step=100.0)
max_risk_procenta = st.sidebar.slider("Maximální povolená ztráta na jeden obchod (%):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# Výběr akcie přímo ve Streamlitu
ticker_akcie = st.text_input("Zadejte ticker americké akcie (např. NVDA, AAPL, TSLA):", "NVDA").upper()

def priprav_data_s_indexem(ticker, perioda="2y"):
    try:
        # 1. Stažení dat pro samotnou akcii i pro tržní index S&P 500 (^GSPC)
        df = yf.download(ticker, period=perioda, multi_level_index=False)
        spy = yf.download("^GSPC", period=perioda, multi_level_index=False)
        
        if df.empty or spy.empty:
            st.error("Nepodařilo se stáhnout tržní data.")
            return None
        
        # Očištění sloupců
        df.columns = [str(col).strip() for col in df.columns]
        spy.columns = [str(col).strip() for col in spy.columns]
        
        # Přidání tržního kontextu (S&P 500) do tabulky naší akcie
        df['SPY_Close'] = spy['Close']
        df['SPY_Rocni_Trend'] = spy['Close'].rolling(window=200).mean() # 200denní klouzavý průměr trhu
        df['Dist_SPY_Trend'] = (df['SPY_Close'] - df['SPY_Rocni_Trend']) / df['SPY_Close']
        
        # 2. Výpočet ATR (Average True Range) pro filtr volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
        
        # 3. Výpočet dynamických Fibonacciho hladin (50 dní)
        okno_fib = 50
        df['High_50'] = df['High'].rolling(window=okno_fib).max()
        df['Low_50'] = df['Low'].rolling(window=okno_fib).min()
        df['Fib_Range'] = df['High_50'] - df['Low_50']
        
        df['Fib_382'] = df['High_50'] - (df['Fib_Range'] * 0.382)
        df['Fib_500'] = df['High_50'] - (df['Fib_Range'] * 0.500)
        df['Fib_618'] = df['High_50'] - (df['Fib_Range'] * 0.618)
        
        # 4. Feature Engineering - Příznaky pro umělou inteligenci
        df['Dist_Fib_382'] = (df['Close'] - df['Fib_382']) / df['Close']
        df['Dist_Fib_500'] = (df['Close'] - df['Fib_500']) / df['Close']
        df['Dist_Fib_618'] = (df['Close'] - df['Fib_618']) / df['Close']
        df['ATR_Percent'] = df['ATR'] / df['Close']
        
        # RSI indikátor
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # Target: Směr pro příští den
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Chyba při zpracování dat: {e}")
        return None

# Spuštění analýzy
data = priprav_data_s_indexem(ticker_akcie, perioda="2y")

if data is not None and not data.empty:
    features = ['Dist_Fib_382', 'Dist_Fib_500', 'Dist_Fib_618', 'RSI', 'ATR_Percent', 'Dist_SPY_Trend']
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

    # === NEPRŮSTŘELNÉ VYTAŽENÍ PREDIKCE A JISTOTY ===
    raw_predikce = model.predict(X_dnes)
    # Převedeme jakýkoliv formát (NumPy array, list, Pandas Series) na jednorozměrné pole a vezmeme první prvek
    predikce = int(np.atleast_1d(raw_predikce)[0])
    
    raw_pravdepodobnost = model.predict_proba(X_dnes)
    # Zde bezpečně získáme pravděpodobnost pro vybranou třídu (0 nebo 1)
    jistota = float(np.atleast_2d(raw_pravdepodobnost)[0][predikce]) * 100
    # ===============================================

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

    # --- CHYTRÝ VÝPOČET RIZIKA A LOTŮ PRO CFD U XTB ---
    max_dolarovy_risk = kapital * (max_risk_procenta / 100.0)
    vzdalenost_sl = abs(aktualni_cena - stop_loss)
    
    # Výpočet doporučeného objemu v lotech
    doporuceny_objem_loty = max_dolarovy_risk / vzdalenost_sl

    # Zobrazení výsledků
    st.subheader(f"📋 Pokyny pro ruční zadání do XTB ({ticker_akcie} CFD)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Doporučený směr", value=smer_text)
        st.metric(label="Aktuální cena", value=f"{aktualni_cena:.2f} USD")
    with col2:
        st.metric(label="Jistota pro tento obchod", value=f"{jistota:.2f} %")
        st.metric(label="Denní volatilita (ATR)", value=f"{atr:.2f} USD")
    with col3:
        st.metric(label="Historická přesnost modelu", value=f"{historicka_presnost:.1f} %")

    # Panel s řízením rizika
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
