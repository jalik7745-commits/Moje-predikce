import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Trading Asistent", layout="wide")

st.title("🤖 XGBoost + Fibonacci + ATR Trading Asistent")
st.write("Predikce a výpočet hladin pro americké CFD kontrakty.")

# Výběr akcie přímo ve Streamlitu
ticker_akcie = st.text_input("Zadejte ticker americké akcie (např. NVDA, AAPL, TSLA):", "NVDA").upper()

def priprav_data_s_atr(ticker, perioda="2y"):
    try:
        # multi_level_index=False zamezí vzniku MultiIndex chyb v novém yfinance
        df = yf.download(ticker, period=perioda, multi_level_index=False)
        if df.empty:
            st.error(f"Nepodařilo se stáhnout žádná data pro ticker {ticker}. Ověřte, zda je správný.")
            return None
        
        # Sjednocení názvů sloupců
        df.columns = [str(col).strip() for col in df.columns]
        
        # Výpočet ATR (Average True Range) - perioda 14
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Výpočet dynamických Fibonacciho hladin (50 dní)
        okno_fib = 50
        df['High_50'] = df['High'].rolling(window=okno_fib).max()
        df['Low_50'] = df['Low'].rolling(window=okno_fib).min()
        df['Fib_Range'] = df['High_50'] - df['Low_50']
        
        df['Fib_382'] = df['High_50'] - (df['Fib_Range'] * 0.382)
        df['Fib_500'] = df['High_50'] - (df['Fib_Range'] * 0.500)
        df['Fib_618'] = df['High_50'] - (df['Fib_Range'] * 0.618)
        
        # Příznaky (Features)
        df['Dist_Fib_382'] = (df['Close'] - df['Fib_382']) / df['Close']
        df['Dist_Fib_500'] = (df['Close'] - df['Fib_500']) / df['Close']
        df['Dist_Fib_618'] = (df['Close'] - df['Fib_618']) / df['Close']
        df['ATR_Percent'] = df['ATR'] / df['Close']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Chyba při zpracování dat: {e}")
        return None

# Spuštění analýzy
data = priprav_data_s_atr(ticker_akcie, perioda="2y")

if data is not None and not data.empty:
    features = ['Dist_Fib_382', 'Dist_Fib_500', 'Dist_Fib_618', 'RSI', 'ATR_Percent']
    X = data[features]
    y = data['Target']

    X_dnes = X.tail(1)
    X_historie = X.iloc[:-1]
    y_historie = y.iloc[:-1]

    # Rozdělení na trénovací a testovací sadu
    X_train, X_test, y_train, y_test = train_test_split(X_historie, y_historie, test_size=0.2, shuffle=False)

    # Trénování modelu
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Zjištění historické úspěšnosti na testovacích datech
    y_pred_test = model.predict(X_test)
    historicka_presnost = accuracy_score(y_test, y_pred_test) * 100

    # Predikce pro dnešní den
    vysledek_predikce = model.predict(X_dnes)
    predikce = int(vysledek_predikce[0])
    
    vysledek_pravdepodobnosti = model.predict_proba(X_dnes)
    jistota = float(vysledek_pravdepodobnosti[0][predikce]) * 100

    # Bezpečné stažení skalárních hodnot
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
    else:
        smer_text = "PRODEJ (SELL/SHORT) 📉"
        stop_loss = fib_382 + (atr * nasobek_atr)
        take_profit = low_50 + (atr * 0.5)

    # Zobrazení výsledků ve Streamlitu
    st.subheader(f"📋 Pokyny pro ruční zadání do XTB ({ticker_akcie})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Doporučený směr", value=smer_text)
        st.metric(label="Aktuální cena", value=f"{aktualni_cena:.2f} USD")
    with col2:
        st.metric(label="Jistota pro tento obchod", value=f"{jistota:.2f} %")
        st.metric(label="Denní volatilita (ATR)", value=f"{atr:.2f} USD")
    with col3:
        st.metric(label="Historická přesnost modelu", value=f"{historicka_presnost:.1f} %", 
                  help="Úspěšnost modelu na historických datech této konkrétní akcie, která nebyla použita k trénování.")

    st.info(f"**STOP-LOSS (SL):** {stop_loss:.2f} USD | **TAKE-PROFIT (TP):** {take_profit:.2f} USD")

    # --- GRAF PRO STREAMLIT ---
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
