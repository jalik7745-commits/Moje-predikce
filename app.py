import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xml.etree.ElementTree as ET
from xgboost import XGBClassifier

st.set_page_config(page_title="ULTIMATE AI Akciový Prediktor", layout="wide")
st.title("🦅 ULTIMATE AI Analytická Platforma & Backtester")
st.write("Optimalizovaná a odlehčený verze s XGBoost modelem šetrným k operační paměti.")

# --- DATOVÉ FUNKCE (Omezeno na 3 roky pro úsporu RAM) ---
@st.cache_data(ttl=3600)  
def stahni_data(ticker):
    d = yf.download(ticker, period="3y", interval="1d", multi_level_index=False)
    s = yf.download("^GSPC", period="3y", interval="1d", multi_level_index=False)
    v = yf.download("^VIX", period="3y", interval="1d", multi_level_index=False)
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
            for item in root.findall('.//item')[:4]:  
                t = item.find('title')
                if t is not None and t.text:
                    titles.append(t.text.strip())
    except:
        pass
    return titles

# --- UŽIVATELSKÝ VSTUP ---
ticker = st.text_input("Zadejte ticker akcie (např. AAPL, NVDA, TSLA):", "AAPL").upper().strip()
tlacitko = st.button("Spustit ULTIMATE AI analýzu")

if tlacitko:
    with st.spinner("Provádím rychlou a úspornou AI analýzu..."):
        # 1. Načtení dat
        raw_data, sp500, vix = stahni_data(ticker)
        
        if raw_data.empty:
            raw_data = yf.download(ticker, period="3y", interval="1d", multi_level_index=False)
        
        if raw_data.empty:
            st.error(f"Nepodařilo se načíst data pro ticker '{ticker}'.")
            st.stop()
            
        data = raw_data.copy()
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        close_prices = pd.Series(data['Close'].to_numpy().flatten(), index=data.index)
        data = data.loc[~data.index.duplicated(keep='first')]
        
        if not sp500.empty:
            data['SP500_Close'] = pd.Series(sp500['Close'].to_numpy().flatten(), index=sp500.index)
        else:
            data['SP500_Close'] = data['Close']
            
        if not vix.empty:
            data['VIX_Close'] = pd.Series(vix['Close'].to_numpy().flatten(), index=vix.index)
        else:
            data['VIX_Close'] = 20.0

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

        # 4. Sentiment zpráv
        st.sidebar.subheader("📰 Titulky zpráv")
        vysledny_sentiment = 0.0
        titulky = stahni_rss(ticker)
        if titulky:
            for t in titulky:
                st.sidebar.write(f"⚪ {t[:70]}...")
        else:
            st.sidebar.write("Žádné zprávy nenalezeny.")

        data['Sentiment'] = vysledny_sentiment
        data = data.dropna()

        if data.empty:
            st.error("Nedostatek dat pro analýzu.")
            st.stop()

        # 5. Chronologické rozdělení dat (80% trénink, 20% test)
        predikce_na_dni = 5
        target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
        data['Target'] = target_values[:len(data)]
        
        vlastnosti = ['SMA20', 'SMA50', 'RSI', 'MACD', 'SP500_Close', 'VIX_Close', 'PE', 'Margin', 'Sentiment']
        
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        X_train = train_data[vlastnosti].to_numpy()
        y_train = train_data['Target'].to_numpy().flatten()
        
        X_test = test_data[vlastnosti].to_numpy()
        y_test = test_data['Target'].to_numpy().flatten()
        
        X_aktualni = data[vlastnosti].to_numpy()[-1].reshape(1, -1)

        # 6. FIXNÍ POHODLNÝ XGBOOST MODEL (Extrémně úsporný na RAM)
        model = XGBClassifier(
            max_depth=4, 
            learning_rate=0.05, 
            n_estimators=80, 
            eval_metric='logloss', 
            random_state=42,
            n_jobs=1  # Vynucení jednoho vlákna pro stabilitu paměti
        )
        model.fit(X_train, y_train)
        uprocenta = model.score(X_test, y_test) * 100

        # 7. BACKTESTING
        test_predictions = model.predict(X_test)
        kapital = 10000.0  
        pozice = 0.0       
        historie_kapitalu = []
        historie_buy_hold = []
        
        pocatecni_cena = test_data['Close'].iloc
        mnozstvi_buy_hold = kapital / pocatecni_cena
        
        for i in range(len(test_data)):
            aktualni_cena = test_data['Close'].iloc[i]
            signal = test_predictions[i]
            
            if signal == 1 and pozice == 0.0:
                pozice = kapital / aktualni_cena
                kapital = 0.0
            elif signal == 0 and pozice > 0.0:
                kapital = pozice * aktualni_cena
                pozice = 0.0
                
            hodnota_portfolia = kapital if pozice == 0.0 else pozice * aktualni_cena
            historie_kapitalu.append(hodnota_portfolia)
            historie_buy_hold.append(mnozstvi_buy_hold * aktualni_cena)
            
        test_data = test_data.copy()
        test_data['AI_Strategie'] = historie_kapitalu
        test_data['Buy_Hold_Strategie'] = historie_buy_hold
        
        final_ai_vybava = historie_kapitalu[-1]
        final_bh_vybava = historie_buy_hold[-1]
        ai_procenta = ((final_ai_vybava - 10000.0) / 10000.0) * 100

        # 8. Zobrazení výsledků a metrik
        st.subheader(f"Komplexní AI analýza pro {ticker}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Reálná úspěšnost na testu", value=f"{uprocenta:.2f} %")
        with col2:
            st.metric(label="Finální hodnota AI účtu", value=f"${final_ai_vybava:,.2f}", delta=f"{ai_procenta:.1f} % zisku")
        
        vysledek = int(model.predict(X_aktualni))
        pravdepodobnost = model.predict_proba(X_aktualni)[vysledek] * 100
        
        with col3:
            if vysledek == 1:
                st.success(f"🤖 AI PREDPOVÍDÁ: RŮST DO 5 DNÍ ({pravdepodobnost:.1f} %)")
            else:
                st.warning(f"🤖 AI PREDPOVÍDÁ: POKLES DO 5 DNÍ ({pravdepodobnost:.1f} %)")

        st.write(f"**Výsledek simulace (počáteční vklad $10,000):** AI dosáhla stavu **${final_ai_vybava:,.2f}**, pasivní strategie Kup a drž **${final_bh_vybava:,.2f}**.")

        # 9. VYCRESLENÍ GRAFŮ
        st.subheader("📈 Vývoj simulovaného kapitálu")
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(x=test_data.index, y=test_data['AI_Strategie'], name='AI Engine', line=dict(color='#2ca02c', width=3)))
        fig_cap.add_trace(go.Scatter(x=test_data.index, y=test_data['Buy_Hold_Strategie'], name='Kup a drž', line=dict(color='#7f7f7f', width=2, dash='dot')))
        fig_cap.update_layout(height=350, template="plotly_white", showlegend=True, xaxis_title="Datum")
        st.plotly_chart(fig_cap, use_container_width=True)

        st.subheader("📊 Analýza trhu a technické indikátory")
        fig_ind = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Cena akcie', 'RSI', 'MACD'))
        
        fig_ind.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena', line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='#ff7f0e', dash='dash')), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='#d62728', dash='dot')), row=1, col=1)
        
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#9467bd')), row=2, col=1)
        fig_ind.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#e377c2')), row=3, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signál', line=dict(color='#bcbd22', width=1)), row=3, col=1)
        
