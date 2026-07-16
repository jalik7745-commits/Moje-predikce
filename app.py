import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Jednorázové stažení slovníku pro analýzu sentimentu textu
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="PRO AI Akciový Prediktor", layout="wide")
st.title("🚀 Profesionální AI Analytická Platforma")
st.write("Kombinace tržních indexů, technických indikátorů, finančního zdraví a stabilního RSS sentimentu zpráv.")

# -----------------------------------------------------------------------------
# POMOCNÉ FUNKCE S UKLÁDÁNÍM DO MEZIPAMĚTI (CACHING PRO VYSOKÝ VÝKON)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Data se udrží v paměti 1 hodinu, aplikace reaguje okamžitě
def stahni_trzni_data(ticker):
    data = yf.download(ticker, period="5y", interval="1d", multi_level_index=False, group_by='ticker')
    sp500 = yf.download("^GSPC", period="5y", interval="1d", multi_level_index=False, group_by='ticker')
    vix = yf.download("^VIX", period="5y", interval="1d", multi_level_index=False, group_by='ticker')
    return data, sp500, vix

@st.cache_data(ttl=1800)  # Zprávy se obnovují každých 30 minut
def ziskej_zpravy_rss(ticker):
    """Stabilní metoda stahování zpráv přes oficiální RSS kanál Yahoo Finance"""
    url = f"https://yahoo.com{ticker}"
    hlavicky = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    titulky = []
    try:
        odpoved = requests.get(url, headers=hlavicky, timeout=5)
        if odpoved.status_code == 200:
            koren = ET.fromstring(odpoved.content)
            for polozka in koren.findall('.//item')[:6]:  # Načte posledních 6 zpráv
                titulek = polozka.find('title')
                if titulek is not None and titulek.text:
                    titulky.append(titulek.text)
    except Exception:
        pass
    return titulky

# -----------------------------------------------------------------------------
# UŽIVATELSKÉ ROZHRANÍ
# -----------------------------------------------------------------------------
ticker = st.text_input("Zadejte ticker akcie (např. AAPL, AMZN, MSFT, ČEZ přes CEZ.PR):", "AAPL").upper().strip()
tlacitko = st.button("Spustit komplexní PRO analýzu")

if tlacitko:
    with st.spinner("Provádím hloubkovou analýzu trhů a textových zpráv..."):
        try:
            # 1. NAČTENÍ TRŽNÍCH DAT
            data, sp500, vix = stahni_trzni_data(ticker)
            
            if data.empty or sp500.empty or vix.empty:
                st.error("Nepodařilo se stáhnout kompletní tržní data. Ověřte správnost tickeru.")
                st.stop()
                
            data = data.copy()
            close_prices = pd.Series(data['Close'].to_numpy().ravel(), index=data.index)
            
            # Synchronizace indexů indexů s akcií podle data
            data['SP500_Close'] = pd.Series(sp500['Close'].to_numpy().ravel(), index=sp500.index)
            data['VIX_Close'] = pd.Series(vix['Close'].to_numpy().ravel(), index=vix.index)

            # 2. VÝPOČET POKROČILÝCH INDIKÁTORŮ
            data['SMA20'] = close_prices.rolling(window=20).mean()
            data['SMA50'] = close_prices.rolling(window=50).mean()
            
            delta = close_prices.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            
            # Výpočet MACD (12-denní EMA - 26-denní EMA) a Signální linie
            ema12 = close_prices.ewm(span=12, adjust=False).mean()
            ema26 = close_prices.ewm(span=26, adjust=False).mean()
            data['MACD'] = ema12 - ema26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # 3. FUNDAMENTÁLNÍ ANALÝZA
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
                st.sidebar.write("Fundamenty staženy s omezením.")
                pe_ratio, profit_margin, rev_growth = 0, 0, 0

            data['PE'] = pe_ratio
            data['Margin'] = profit_margin

            # 4. NOVÁ STABILNÍ ANALÝZA SENTIMENTU (RSS)
            st.sidebar.subheader("📰 Nejnovější titulky zpráv")
            sia = SentimentIntensityAnalyzer()
            vysledny_sentiment = 0.0
            titulky_zprav = ziskej_zpravy_rss(ticker)
            
            if titulky_zprav:
                for titulek in titulky_zprav:
                    score = sia.polarity_scores(titulek)['compound']
                    vysledny_sentiment += score
                    smajlík = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                    st.sidebar.write(f"{smajlík} {titulek[:75]}...")
                vysledny_sentiment /= len(titulky_zprav)
            else:
                st.sidebar.write("Žádné aktuální zprávy nebyly nalezeny.")

            data['Sentiment'] = vysledny_sentiment
            data = data.dropna()

            # 5. STROJOVÉ UČENÍ (Machine Learning)
            predikce_na_dni = 5
            target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
            data['Target'] = target_values[:len(data)]
            
            vlastnosti = ['SMA20', 'SMA50', 'RSI', 'MACD', 'SP500_Close', 'VIX_Close', 'PE', 'Margin', 'Sentiment']
            X = data[vlastnosti].values
            y = data['Target'].values
            
            X_aktualni = X[-1].reshape(1, -1)
            X_model = X[:-predikce_na_dni]
            y_model = y[:-predikce_na_dni]
            
            X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            uprocenta = model.score(X_test, y_test) * 100

            # 6. ZOBRAZENÍ METRIK VÝSLEDKŮ
            st.subheader(f"Komplexní AI analýza pro {ticker}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Úspěšnost modelu (Accuracy)", value=f"{uprocenta:.2f} %")
            with col2:
                st.metric(label="Kombinovaný sentiment zpráv", value=f"{vysledny_sentiment:.2f}", help="Od -1 (negativní) do +1 (pozitivní)")
            
            vysledek = int(model.predict(X_aktualni))
            pravdepodobnosti = model.predict_proba(X_aktualni)
            pravdepodobnost_vysledku = pravdepodobnosti[0][vysledek] * 100
            
            with col3:
                if vysledek == 1:
                    st.success(f"🤖 AI PREDPOVÍDÁ: RŮST DO {predikce_na_dni} DNÍ ({pravdepodobnost_vysledku:.1f} %)")
                else:
                    st.warning(f"🤖 AI PREDPOVÍDÁ: POKLES DO {predikce_na_dni} DNÍ ({pravdepodobnost_vysledku:.1f} %)")

            # 7. POKROČILÝ MULTI-GRAF (Cena + RSI + MACD)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                subplot_titles=(f'Cena akcie a klouzavé průměry', 'RSI Indikátor', 'MACD (Hybnost trhu)'),
                                row_width=[0.25, 0.25, 0.5])

            # Graf 1: Hlavní cena
            fig.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena akcie', line=dict(color='#1f77b4', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20 (Krátkodobý)', line=dict(color='#ff7f0e', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50 (Dlouhodobý)', line=dict(color='#2ca02c', dash='dot')), row=1, col=1)

            # Graf 2: RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#9467bd')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Překoupeno", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Přeprodáno", row=2, col=1)

            # Graf 3: MACD
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#e377c2')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signální linie', line=dict(color='#7f7f7f', width=1)), row=3, col=1)

            fig.update_layout(height=800, template="plotly_white", showlegend=True,
                              xaxis3_title="Datum", yaxis_title="Cena ($)", yaxis2_title="RSI", yaxis3_title="MACD")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Došlo k neočekávané chybě: {e}")
