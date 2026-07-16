import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Jednorázové stažení slovníku pro analýzu sentimentu textu
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Pokročilá AI Predikce Akcií", layout="wide")
st.title("📊 Pokročilá AI Analýza s více zdroji dat")
st.write("Tato verze kombinuje technickou analýzu, finanční zdraví firmy, index strachu VIX a sentiment zpráv.")

ticker = st.text_input("Zadejte ticker akcie (např. AAPL, TSLA, NVDA):", "AAPL").upper().strip()
tlacitko = st.button("Spustit komplexní analýzu")

if tlacitko:
    with st.spinner("Stahuji data z trhů, analýzu zpráv a trénuji model..."):
        try:
            # 1. NAČTENÍ DAT Z BURZY (Zajištění ploché struktury pomocí group_by='ticker')
            data = yf.download(ticker, period="5y", interval="1d", multi_level_index=False, group_by='ticker')
            sp500 = yf.download("^GSPC", period="5y", interval="1d", multi_level_index=False, group_by='ticker')
            vix = yf.download("^VIX", period="5y", interval="1d", multi_level_index=False, group_by='ticker')
            
            if data.empty or sp500.empty or vix.empty:
                st.error("Nepodařilo se stáhnout kompletní tržní data.")
                st.stop()
                
            data = data.copy()
            
            # Bezpečné vytvoření jednorozměrné Series pro Close ceny
            close_prices = pd.Series(data['Close'].to_numpy().ravel(), index=data.index)
            
            # Propojení dat s tržními indexy podle data
            data['SP500_Close'] = pd.Series(sp500['Close'].to_numpy().ravel(), index=sp500.index)
            data['VIX_Close'] = pd.Series(vix['Close'].to_numpy().ravel(), index=vix.index)

            # 2. TECHNICKÁ ANALÝZA (Základní indikátory)
            data['SMA20'] = close_prices.rolling(window=20).mean()
            data['SMA50'] = close_prices.rolling(window=50).mean()
            
            delta = close_prices.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            data['MACD'] = close_prices.rolling(window=12).mean() - close_prices.rolling(window=26).mean()

            # 3. FUNDAMENTÁLNÍ ANALÝZA (Účetnictví firmy)
            st.sidebar.subheader("📋 Finanční zdraví firmy")
            try:
                info = yf.Ticker(ticker).info
                pe_ratio = info.get('trailingPE', 0) if info.get('trailingPE') is not None else 0
                profit_margin = info.get('profitMargins', 0) if info.get('profitMargins') is not None else 0
                rev_growth = info.get('revenueGrowth', 0) if info.get('revenueGrowth') is not None else 0
                
                st.sidebar.write(f"**P/E Ratio:** {pe_ratio:.2f}" if pe_ratio else "**P/E Ratio:** N/A")
                st.sidebar.write(f"**Zisková marže:** {profit_margin*100:.1f} %")
                st.sidebar.write(f"**Růst tržeb:** {rev_growth*100:.1f} %")
            except:
                st.sidebar.write("Fundamentální data momentálně nedostupná.")
                pe_ratio, profit_margin, rev_growth = 0, 0, 0

            data['PE'] = pe_ratio
            data['Margin'] = profit_margin

            # 4. ANALÝZA SENTIMENTU ZPRÁV (News Sentiment)
            st.sidebar.subheader("📰 Nejnovější titulky zpráv")
            sia = SentimentIntensityAnalyzer()
            vysledny_sentiment = 0.0
            pocet_zprav = 0
            
            try:
                zpravy = yf.Ticker(ticker).news
                if zpravy:
                    for zprava in zpravy[:5]:
                        titulek = zprava.get('title', '')
                        score = sia.polarity_scores(titulek)['compound']
                        vysledny_sentiment += score
                        pocet_zprav += 1
                        
                        smajlík = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                        st.sidebar.write(f"{smajlík} {titulek[:60]}...")
                
                if pocet_zprav > 0:
                    vysledny_sentiment /= pocet_zprav
            except:
                pass
            
            data['Sentiment'] = vysledny_sentiment
            data = data.dropna()

            # 5. STROJOVÉ UČENÍ (Machine Learning)
            predikce_na_dni = 5
            target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
            
            # Oprava zarovnání délky cílového sloupce
            data['Target'] = target_values[:len(data)]
            
            vlastnosti = ['SMA20', 'SMA50', 'RSI', 'MACD', 'SP500_Close', 'VIX_Close', 'PE', 'Margin', 'Sentiment']
            X = data[vlastnosti].values
            y = data['Target'].values
            
            X_aktualni = X[-1].reshape(1, -1)
            X_model = X[:-predikce_na_dni]
            y_model = y[:-predikce_na_dni]
            
            X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            uprocenta = model.score(X_test, y_test) * 100

            # 6. ZOBRAZENÍ VÝSLEDKŮ
            st.subheader(f"Komplexní AI analýza pro {ticker}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Úspěšnost modelu (Accuracy)", value=f"{uprocenta:.2f} %")
            with col2:
                st.metric(label="Aktuální sentiment zpráv", value=f"{vysledny_sentiment:.2f}", help="Rozsah od -1 (nejhorší) do +1 (nejlepší)")
            
            vysledek = int(model.predict(X_aktualni)[0]) # Opraveno na skalární hodnotu z pole
            pravdepodobnosti = model.predict_proba(X_aktualni)[0] # Opraveno na indexaci nultého prvku
            pravdepodobnost_vysledku = pravdepodobnosti[vysledek] * 100
            
            with col3:
                if vysledek == 1:
                    st.success(f"🤖 AI PREDPOVÍDÁ: RŮST (Pravděpodobnost: {pravdepodobnost_vysledku:.1f} %)")
                else:
                    st.warning(f"🤖 AI PREDPOVÍDÁ: POKLES (Pravděpodobnost: {pravdepodobnost_vysledku:.1f} %)")

            # Graf
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena akcie', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='green', dash='dot')))
            fig.update_layout(title=f"Vývoj ceny {ticker}", xaxis_title="Datum", yaxis_title="Cena ($)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Došlo k chybě při komplexním zpracování: {e}")
