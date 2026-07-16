import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBClassifier

st.set_page_config(page_title="HIGH-PRECISION AI Engine", layout="wide")
st.title("🦅 Nejpřesnější AI Prediktor (XGBoost High-Precision)")
st.write("Veškerý výkon je alokován do matematické optimalizace dnešního směrového signálu.")

# --- ODLEHČENÉ NAČTÍTÁNÍ DAT (3 roky pro ideální trénovací vzorek) ---
@st.cache_data(ttl=1800)  
def stahni_cista_data(ticker):
    d = yf.download(ticker, period="3y", interval="1d", multi_level_index=False)
    s = yf.download("^GSPC", period="3y", interval="1d", multi_level_index=False)
    v = yf.download("^VIX", period="3y", interval="1d", multi_level_index=False)
    return d, s, v

# --- UŽIVATELSKÝ VSTUP ---
ticker = st.text_input("Zadejte ticker akcie (např. AAPL, NVDA, TSLA):", "AAPL").upper().strip()
tlacitko = st.button("SPUSTIT MAXIMÁLNÍ PREDIKCI")

if tlacitko:
    with st.spinner("AI optimalizuje matematické vztahy indikátorů pro dnešní den..."):
        # 1. Načtení dat
        raw_data, sp500, vix = stahni_cista_data(ticker)
        if raw_data.empty:
            raw_data = yf.download(ticker, period="3y", interval="1d", multi_level_index=False)
        if raw_data.empty:
            st.error("Chyba při načítání tržních dat.")
            st.stop()
            
        data = raw_data.copy()
        
        # Srovnání sloupců
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        close_prices = pd.Series(data['Close'].to_numpy().flatten(), index=data.index)
        high_prices = pd.Series(data['High'].to_numpy().flatten(), index=data.index)
        low_prices = pd.Series(data['Low'].to_numpy().flatten(), index=data.index)
        
        data = data.loc[~data.index.duplicated(keep='first')]
        data['SP500_Close'] = pd.Series(sp500['Close'].to_numpy().flatten(), index=sp500.index)
        data['VIX_Close'] = pd.Series(vix['Close'].to_numpy().flatten(), index=vix.index)

        # 2. POKROČILÝ FEATURE ENGINEERING (Klíč k přesnosti AI)
        data['SMA20'] = close_prices.rolling(window=20).mean()
        data['SMA50'] = close_prices.rolling(window=50).mean()
        
        # RSI Výpočet
        delta = close_prices.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # MACD
        data['MACD'] = close_prices.ewm(span=12, adjust=False).mean() - close_prices.ewm(span=26, adjust=False).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # NOVÉ: ATR (Průměrný skutečný rozsah - volatilita)
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        
        # NOVÉ: Tržní Momentum (Hybnost ceny za 10 dní)
        data['Momentum'] = close_prices.diff(10)

        # 3. Sidebar fundamenty (Informativní)
        st.sidebar.subheader("📋 Finanční fundamenty")
        try:
            info = yf.Ticker(ticker).info
            st.sidebar.write(f"**P/E Ratio:** {info.get('trailingPE', 0):.2f}")
            st.sidebar.write(f"**Zisková marže:** {info.get('profitMargins', 0)*100:.1f} %")
        except:
            st.sidebar.write("Fundamentální data nedostupná.")

        data = data.dropna()

        # 4. TRÉNOVÁNÍ STRATEGIE WALK-FORWARD
        predikce_na_dni = 5
        target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
        data['Target'] = target_values[:len(data)]
        
        # Seznam všech vlastností zkonstruovaných pro maximální přesnost
        vlastnosti = ['SMA20', 'SMA50', 'RSI', 'MACD', 'SP500_Close', 'VIX_Close', 'ATR', 'Momentum']
        X = data[vlastnosti].to_numpy()
        y = data['Target'].to_numpy().flatten()
        
        # Poslední řádek je náš dnešní stav trhu k predikci
        X_aktualni = X[-1].reshape(1, -1)
        
        # Učení probíhá na historických datech bez posledních 5 dní (u kterých ještě neznáme výsledek)
        X_model = X[:-predikce_na_dni]
        y_model = y[:-predikce_na_dni]
        
        # Rozdělení na trénovací a validační vzorek (posledních 60 obchodních dní jako test stability)
        X_train, X_val = X_model[:-60], X_model[-60:]
        y_train, y_val = y_model[:-60], y_model[-60:]
        
        # Konfigurace XGBoost zaměřená proti přeučení (vysoká zobecňovací schopnost)
        model = XGBClassifier(
            max_depth=3,            # Mělčí stromy brání pamatování si šumu
            learning_rate=0.03,      # Pomalejší učení pro stabilnější konvergenci
            n_estimators=120,       # Vyvážený počet iterací
            subsample=0.8,          # Model vidí pokaždé jen 80% řádků (odolnost vůči anomáliím)
            colsample_bytree=0.8,   # Model vidí pokaždé jen 80% indikátorů
            eval_metric='logloss',
            random_state=42,
            n_jobs=1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        uprocenta = model.score(X_val, y_val) * 100

        # 5. ZOBRAZENÍ FINÁLNÍHO VERDIKTU AI
        st.subheader(f"🎯 Výsledek matematické optimalizace pro {ticker}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Ověřená přesnost směrového signálu (Accuracy)", value=f"{uprocenta:.2f} %", help="Úspěšnost modelu na posledních 60 dnech před dneškem.")
        
        vysledek = int(model.predict(X_aktualni))
        pravdepodobnost = model.predict_proba(X_aktualni)[vysledek] * 100
        
        with col2:
            if vysledek == 1:
                st.success(f"🤖 VERDIKT AI: OČEKÁVÁ SE RŮST (Pravděpodobnost: {pravdepodobnost:.1f} %)")
            else:
                st.warning(f"🤖 VERDIKT AI: OČEKÁVÁ SE POKLES (Pravděpodobnost: {pravdepodobnost:.1f} %)")
                
        st.info(f"Tato predikce udává směr trendu na následujících **{predikce_na_dni} obchodních dní** na základě dnešní tržní konstelace.")

        # 6. TECHNICKÝ GRAF PRO KONTROLU
        fig_ind = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Cena akcie', 'RSI', 'MACD'))
        fig_ind.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena', line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', line=dict(color='#ff7f0e', dash='dash')), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='#d62728', dash='dot')), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#9467bd')), row=2, col=1)
        fig_ind.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#e377c2')), row=3, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signál', line=dict(color='#bcbd22', width=1)), row=3, col=1)
        fig_ind.update_layout(height=650, template="plotly_white", showlegend=True, xaxis3_title="Datum")
        st.plotly_chart(fig_ind, use_container_width=True)
