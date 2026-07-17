import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBClassifier

st.set_page_config(page_title="HIGH-PRECISION AI Engine", layout="wide")
st.title("🦅 Nejpřesnější AI Prediktor (XGBoost High-Precision)")
st.write("Veškerý výkon je alokován do pokročilé matematické transformace indikátorů pro dnešní den.")

# --- ODLEHČENÉ NAČTÍTÁNÍ DAT ---
@st.cache_data(ttl=1800)  
def stahni_cista_data(ticker):
    d = yf.download(ticker, period="3y", interval="1d", multi_level_index=False)
    s = yf.download("^GSPC", period="3y", interval="1d", multi_level_index=False)
    v = yf.download("^VIX", period="3y", interval="1d", multi_level_index=False)
    return d, s, v

# --- UŽIVATELSKÝ VSTUP ---
ticker = st.text_input("Zadejte ticker akcie (např. F, DVN, KMI, SO, CL, NG=F):", "DVN").upper().strip()
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

        # 2. POKROČILÝ FEATURE ENGINEERING (Transformace pro zvýšení přesnosti)
        data['SMA20'] = close_prices.rolling(window=20).mean()
        data['SMA50'] = close_prices.rolling(window=50).mean()
        
        # Procentuální vzdálenost od průměrů
        data['Dist_SMA20'] = (close_prices - data['SMA20']) / data['SMA20']
        data['Dist_SMA50'] = (close_prices - data['SMA50']) / data['SMA50']
        
        # RSI Výpočet a jeho hybnost (Změna RSI za 3 dny)
        delta = close_prices.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        data['RSI_ROC'] = data['RSI'].diff(3)
        
        # MACD a jeho histogram
        data['MACD'] = close_prices.ewm(span=12, adjust=False).mean() - close_prices.ewm(span=26, adjust=False).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Bollingerova pásma
        std20 = close_prices.rolling(window=20).std()
        data['BB_High'] = data['SMA20'] + (std20 * 2)
        data['BB_Low'] = data['SMA20'] - (std20 * 2)
        data['BB_Position'] = (close_prices - data['BB_Low']) / (data['BB_High'] - data['BB_Low'] + 1e-9)

        # ATR (Volatilita)
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        
        # Tržní Momentum (Hybnost ceny za 10 dní)
        data['Momentum'] = close_prices.diff(10)

        # 3. Sidebar fundamenty
        st.sidebar.subheader("📋 Finanční fundamenty")
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
        data['Sentiment'] = 0.0

        data = data.dropna()

        # 4. TRÉNOVÁNÍ STRATEGIE WALK-FORWARD
        predikce_na_dni = 5
        target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
        data['Target'] = target_values[:len(data)]
        
        # Seznam vlastností
        vlastnosti = ['Dist_SMA20', 'Dist_SMA50', 'RSI', 'RSI_ROC', 'MACD_Hist', 'BB_Position', 'SP500_Close', 'VIX_Close', 'ATR', 'Momentum', 'PE', 'Margin', 'Sentiment']
        X = data[vlastnosti].to_numpy()
        y = data['Target'].to_numpy().flatten()
        
        X_aktualni = X[-1].reshape(1, -1)
        
        X_model = X[:-predikce_na_dni]
        y_model = y[:-predikce_na_dni]
        
        X_train, X_val = X_model[:-60], X_model[-60:]
        y_train, y_val = y_model[:-60], y_model[-60:]
        
        model = XGBClassifier(
            max_depth=3,            
            learning_rate=0.03,      
            n_estimators=130,       
            subsample=0.8,          
            colsample_bytree=0.8,   
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
            st.metric(label="Ověřená přesnost směrového signálu (Accuracy)", value=f"{uprocenta:.2f} %", help="Úspěšnost modelu na posledních 60 dnech.")
        
        # Bezpečné vytáhnutí čistého čísla
        predikce_raw = model.predict(X_aktualni)
        vysledek = int(predikce_raw.item())
        
        # OPRAVENO: Správná 2D indexace [0][vysledek] pro vytažení přesného skaláru z matice
        pravdepodobnosti = model.predict_proba(X_aktualni)
        pravdepodobnost = float(pravdepodobnosti[0][vysledek]) * 100
        
        with col2:
            if vysledek == 1:
                st.success(f"🤖 VERDIKT AI: OČEKÁVÁ SE RŮST (Pravděpodobnost: {pravdepodobnost:.1f} %)")
            else:
                st.warning(f"🤖 VERDIKT AI: OČEKÁVÁ SE POKLES (Pravděpodobnost: {pravdepodobnost:.1f} %)")
                
        st.info(f"Tato predikce udává směr trendu na následujících **{predikce_na_dni} obchodních dní**.")

        # --- KALKULAČKA ŘÍZENÍ RIZIKA ---
        st.markdown("---")
        st.subheader("🧮 Inteligentní kalkulačka obchodního rizika")
        st.write("Výpočty na základě aktuální volatility trhu (ATR) a matematického poměru zisku k riziku RRR 1:2.")
        
        aktualni_cena_akcie = float(close_prices.iloc[-1])
        aktualni_atr = float(data['ATR'].iloc[-1])
        
        col_calc1, col_calc2 = st.columns(2)
        
        with col_calc1:
            st.info(f"**Aktuální cena akcie:** ${aktualni_cena_akcie:,.2f}")
            st.write(f"Průměrný denní pohyb (ATR): ${aktualni_atr:.2f}")
        
        with col_calc2:
            if vysledek == 1:
                stop_loss = aktualni_cena_akcie - (aktualni_atr * 1.5)
                target_profit = aktualni_cena_akcie + (aktualni_atr * 3.0)
                
                st.write("👉 **Doporučené nastavení pro nákup (LONG):**")
                st.error(f"🛑 **Stop-Loss (Ukončení ztráty):** ${stop_loss:.2f}")
                st.success(f"🎯 **Take-Profit (Výběr zisku):** ${target_profit:.2f}")
            else:
                stop_loss = aktualni_cena_akcie + (aktualni_atr * 1.5)
                target_profit = aktualni_cena_akcie - (aktualni_atr * 3.0)
                
                st.write("👉 **Doporučené nastavení pro spekulaci na pokles (SHORT):**")
                st.error(f"🛑 **Stop-Loss (Ukončení ztráty):** ${stop_loss:.2f}")
                st.success(f"🎯 **Take-Profit (Výběr zisku):** ${target_profit:.2f}")
                
        st.markdown("---")

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
