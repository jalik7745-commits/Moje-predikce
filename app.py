import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xml.etree.ElementTree as ET
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Inicializace NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="ULTIMATE AI Akciový Prediktor", layout="wide")
st.title("🦅 ULTIMATE AI Analytická Platforma (XGBoost Engine)")
st.write("Finanční predikce poháněná algoritmem XGBoost s automatickým laděním parametrů v reálném čase.")

# --- CACHED POMOCNÉ FUNKCE (IZOLOVANÉ) ---
@st.cache_data(ttl=3600)  
def stahni_data(ticker):
    d = yf.download(ticker, period="5y", interval="1d")
    s = yf.download("^GSPC", period="5y", interval="1d")
    v = yf.download("^VIX", period="5y", interval="1d")
    return d, s, v

@st.cache_data(ttl=1800)  
def stahni_rss(ticker):
    url = "https://marketwatch.com" if "." in ticker else f"https://marketwatch.com{ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    titles = []
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for item in root.findall('.//item')[:6]:  
                t = item.find('title')
                if t is not None and t.text:
                    titles.append(t.text.strip())
    except:
        pass
    return titles

# --- UŽIVATELSKÝ VSTUP ---
ticker = st.text_input("Zadejte ticker akcie:", "AAPL").upper().strip()
tlacitko = st.button("Spustit ULTIMATE AI analýzu")

if tlacitko:
    with st.spinner("Stahuji data a optimalizuji matematický model XGBoost..."):
        # 1. Načtení dat
        raw_data, sp500, vix = stahni_data(ticker)
        
        if raw_data.empty or sp500.empty or vix.empty:
            st.error("Chyba při stahování dat z trhu.")
            st.stop()
            
        data = raw_data.copy()
        
        # Srovnání indexů (MultiIndex fix)
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        close_prices = pd.Series(data['Close'].to_numpy().flatten(), index=data.index)
        
        data = data.loc[~data.index.duplicated(keep='first')]
        data['SP500_Close'] = pd.Series(sp500['Close'].to_numpy().flatten(), index=sp500.index)
        data['VIX_Close'] = pd.Series(vix['Close'].to_numpy().flatten(), index=vix.index)

        # 2. Technické indikátory
        data['SMA20'] = close_prices.rolling(window=20).mean()
        data['SMA50'] = close_prices.rolling(window=50).mean()
        
        delta = close_prices.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        data['MACD'] = close_prices.ewm(span=12, adjust=False).mean() - close_prices.ewm(span=26, adjust=False).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # 3. Fundamentální data (Sidebar)
        st.sidebar.subheader("📋 Finanční zdraví")
        pe_ratio, profit_margin = 0.0, 0.0
        try:
            info = yf.Ticker(ticker).info
            pe_ratio = info.get('trailingPE', 0) or 0
            profit_margin = info.get('profitMargins', 0) or 0
            st.sidebar.write(f"**P/E Ratio:** {pe_ratio:.2f}" if pe_ratio else "**P/E Ratio:** N/A")
            st.sidebar.write(f"**Zisková marže:** {profit_margin*100:.1f} %")
        except:
            st.sidebar.write("Fundamentální data nedostupná.")

        data['PE'] = pe_ratio
        data['Margin'] = profit_margin

        # 4. Sentiment analýza zpráv
        st.sidebar.subheader("📰 Titulky zpráv")
        sia = SentimentIntensityAnalyzer()
        vysledny_sentiment = 0.0
        titulky = stahni_rss(ticker)
        
        if titulky:
            for t in titulky:
                score = sia.polarity_scores(t)['compound']
                vysledny_sentiment += score
                ikona = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                st.sidebar.write(f"{ikona} {t[:70]}...")
            vysledny_sentiment /= len(titulky)
        else:
            st.sidebar.write("Žádné zprávy nenalezeny.")

        data['Sentiment'] = vysledny_sentiment
        data = data.dropna()

        if data.empty:
            st.error("Nedostatek dat pro analýzu.")
            st.stop()

        # 5. Pokročilá příprava dat pro XGBoost
        predikce_na_dni = 5
        target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
        data['Target'] = target_values[:len(data)]
        
        vlastnosti = ['SMA20', 'SMA50', 'RSI', 'MACD', 'SP500_Close', 'VIX_Close', 'PE', 'Margin', 'Sentiment']
        X = data[vlastnosti].to_numpy()
        y = data['Target'].to_numpy().flatten()
        
        X_aktualni = X[-1].reshape(1, -1)
        X_model = X[:-predikce_na_dni]
        y_model = y[:-predikce_na_dni]
        
        X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
        
        # 6. AUTOMATICKÉ LADĚNÍ PARAMETRŮ (GridSearchCV) - OPRAVENO ZDE
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 150]
        }
        
        base_model = XGBClassifier(eval_metric='logloss', random_state=42)
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Výběr nejlepšího modelu
        model = grid_search.best_estimator_
        uprocenta = model.score(X_test, y_test) * 100

        # 7. Zobrazení výsledků
        st.subheader(f"Komplexní AI analýza pro {ticker}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="XGBoost Úspěšnost (Accuracy)", value=f"{uprocenta:.2f} %")
        with col2:
            st.metric(label="Optimální hloubka modelu", value=f"{grid_search.best_params_['max_depth']} (z 7)", help="Dynamicky zvolená hloubka stromu pro nejvyšší stabilitu.")
        
        vysledek = int(model.predict(X_aktualni)[0])
        pravdepodobnost = model.predict_proba(X_aktualni)[0][vysledek] * 100
        
        with col3:
            if vysledek == 1:
                st.success(f"🤖 AI PREDPOVÍDÁ: RŮST DO {predikce_na_dni} DNÍ ({pravdepodobnost:.1f} %)")
            else:
                st.warning(f"🤖 AI PREDPOVÍDÁ: POKLES DO {predikce_na_dni} DNÍ ({pravdepodobnost:.1f} %)")

        # 8. Vykreslení grafů
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.25, 0.25, 0.5])
        
        fig.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena', line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='#ff7f0e', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='#2ca02c', dash='dot')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#9467bd')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#e377c2')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signál', line=dict(color='#7f7f7f', width=1)), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_white", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
