import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def analyzovat_akcii(ticker):
    ticker = ticker.upper().strip()
    try:
        # 1. Bezpečné stažení dat
        raw_data = yf.download(ticker, period="5y", interval="1d")
        if raw_data.empty:
            return "Chyba: Nepodařilo se stáhnout žádná data. Zkontrolujte ticker.", None

        data = raw_data.copy()

        # Očištění MultiIndexu z Yahoo Finance
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(ticker, axis=1, level=1)
            except:
                data.columns = data.columns.get_level_values(0)

        if 'Close' not in data.columns:
            return f"Chyba: V datech chybí sloupec 'Close'.", None

        # Převedení cen na jednorozměrné čisté pole
        close_prices = pd.Series(data['Close'].values.flatten(), index=data.index)

        # 2. Výpočet indikátorů přes čistý Pandas
        data['SMA20'] = close_prices.rolling(window=20).mean()
        data['SMA50'] = close_prices.rolling(window=50).mean()

        delta = close_prices.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        data['RSI'] = 100 - (100 / (1 + rs))
        data['MACD'] = close_prices.rolling(window=12).mean() - close_prices.rolling(window=26).mean()

        data = data.dropna()

        # 3. Příprava dat pro strojové učení
        predikce_na_dni = 5
        target_values = np.where(close_prices.shift(-predikce_na_dni) > close_prices, 1, 0)
        data['Target'] = target_values[:len(data)]

        X = data[['SMA20', 'SMA50', 'RSI', 'MACD']].values
        y = data['Target'].values

        # BEZPEČNÉ VYTAŽENÍ POSLEDNÍHO ŘÁDKU (Zajištění 2D tvaru pro model)
        X_aktualni = X[-1].reshape(1, -1)

        X_model = X[:-predikce_na_dni]
        y_model = y[:-predikce_na_dni]

        X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)

        # Trénování AI modelu
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        uprocenta = model.score(X_test, y_test) * 100

        # 4. Samotná predikce a bezpečné vytažení pravděpodobnosti
        vysledek = int(model.predict(X_aktualni)[0])
        pravdepodobnosti = model.predict_proba(X_aktualni)[0]
        pravdepodobnost_vysledku = pravdepodobnosti[vysledek] * 100

        smer = "🚀 RŮST" if vysledek == 1 else "📉 POKLES"
        textovy_vystup = f"Analýza pro ticker: {ticker}\n\n" \
                         f"🤖 AI PŘEDPOVÍDÁ (na příštích 5 dní): {smer}\n" \
                         f"🎯 Pravděpodobnost: {pravdepodobnost_vysledku:.1f} %\n" \
                         f"📊 Úspěšnost modelu na historii (Accuracy): {uprocenta:.2f} %"

        # 5. Generování interaktivního grafu
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=close_prices.loc[data.index], name='Cena akcie', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20 (Trend)', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50 (Trend)', line=dict(color='green', dash='dot')))
        fig.update_layout(title=f"Graf vývoje ceny {ticker}", template="plotly_white", xaxis_title="Datum", yaxis_title="Cena ($)")

        return textovy_vystup, fig

    except Exception as e:
        return f"Došlo k chybě: {e}", None
