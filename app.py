import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Nastavení vzhledu stránky ve Streamlitu
st.set_page_config(page_title="AI Predikce Akcií", layout="wide")
st.title("📊 AI Analýza a Predikce Pohybu Akcií")
st.write("Aplikace využívá strojové učení k odhadu budoucího vývoje cen na příštích 5 dní.")

# Uživatelské rozhraní
ticker = st.text_input("Zadejte ticker akcie (např. AAPL, TSLA, NVDA, BTC-USD):", "AAPL").upper().strip()
tlacitko = st.button("Spustit analýzu")

if tlacitko:
    with st.spinner("Stahuji data z burzy Yahoo Finance a trénuji AI model..."):
        try:
            # 1. Bezpečné stažení dat s vypnutým MultiIndexem
            data = yf.download(ticker, period="5y", interval="1d", multi_level_index=False)
            
            if data.empty or len(data) < 100:
                st.error(f"Nepodařilo se stáhnout platná data pro ticker '{ticker}'. Zkontrolujte, zda je správný.")
                st.stop()
            
            # Bezpečné vytvoření kopie a vyčištění sloupců
            data = data.copy()
            close_prices = pd.Series(data['Close'].values.flatten(), index=data.index)

            # 2. Matematické výpočty indikátorů přímo přes čistý Pandas
            data['SMA20'] = close_prices.rolling(window=20).mean()
            data['SMA50'] = close_prices.rolling(window=50).mean()
            
            # Výpočet RSI
            delta = close_prices.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Výpočet MACD
            data['MACD'] = close_prices.rolling(window=12).mean() - close_prices.rolling(window=26).mean()
            
            # Odstranění řádků s chybějícími hodnotami
            data = data.dropna()

            # 3. Příprava dat pro strojové učení
            predikce_na_dni = 5
            target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
            data['Target'] = target_values[:len(data)]
            
            X = data[['SMA20', 'SMA50', 'RSI', 'MACD']].values
            y = data['Target'].values
            
            # Zajištění správného 2D tvaru pro predikci z posledního řádku
            X_aktualni = X[-1].reshape(1, -1)
            
            X_model = X[:-predikce_na_dni]
            y_model = y[:-predikce_na_dni]
            
            X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
            
            # Trénování modelu Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            uprocenta = model.score(X_test, y_test) * 100

            # 4. Vykreslení výsledků na webové stránce
            st.subheader(f"Výsledky analýzy pro {ticker}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Úspěšnost modelu na historii (Accuracy)", value=f"{uprocenta:.2f} %")
            
            # BEZPEČNÝ VÝPOČET FINÁLNÍ PREDIKCE S INDEXEM [0]
            vysledek = int(model.predict(X_aktualni)[0])
            pravdepodobnosti = model.predict_proba(X_aktualni)[0]
            pravdepodobnost_vysledku = pravdepodobnosti[vysledek] * 100
            
            with col2:
                if vysledek == 1:
                    st.success(f"🤖 AI predikuje pro příštích 5 dní: RŮST (Pravděpodobnost: {pravdepodobnost_vysledku:.1f} %)")
                else:
                    st.warning(f"🤖 AI predikuje pro příštích 5 dní: POKLES (Pravděpodobnost: {pravdepodobnost_vysledku:.1f} %)")

            # Interaktivní graf Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena akcie', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20 (Krátkodobý trend)', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50 (Dlouhodobý trend)', line=dict(color='green', dash='dot')))
            fig.update_layout(title=f"Graf vývoje ceny {ticker}", xaxis_title="Datum", yaxis_title="Cena ($)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Poslední dostupná data z burzy:")
            st.dataframe(data.tail(5))
            
        except Exception as e:
            st.error(f"Došlo k chybě při zpracování dat: {e}")
