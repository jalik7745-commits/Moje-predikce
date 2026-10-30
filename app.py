import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def priprav_data_s_atr(ticker, perioda="2y"):
    # 1. Stáhnutí historických dat
    df = yf.download(ticker, period=perioda)
    if df.empty:
        raise ValueError("Nepodařilo se stáhnout data.")
    
    # Odstranění MultiIndexu
    df.columns = [col if isinstance(col, tuple) else col for col in df.columns]
    
    # 2. Výpočet ATR (Average True Range) - perioda 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 3. Výpočet dynamických Fibonacciho hladin (50 dní)
    okno_fib = 50
    df['High_50'] = df['High'].rolling(window=okno_fib).max()
    df['Low_50'] = df['Low'].rolling(window=okno_fib).min()
    df['Fib_Range'] = df['High_50'] - df['Low_50']
    
    df['Fib_382'] = df['High_50'] - (df['Fib_Range'] * 0.382)
    df['Fib_500'] = df['High_50'] - (df['Fib_Range'] * 0.500)
    df['Fib_618'] = df['High_50'] - (df['Fib_Range'] * 0.618)
    
    # Příznaky (Features) pro model - vzdálenosti v % a normalizované ATR
    df['Dist_Fib_382'] = (df['Close'] - df['Fib_382']) / df['Close']
    df['Dist_Fib_500'] = (df['Close'] - df['Fib_500']) / df['Close']
    df['Dist_Fib_618'] = (df['Close'] - df['Fib_618']) / df['Close']
    df['ATR_Percent'] = df['ATR'] / df['Close'] # Vyjadřuje volatilitu vůči ceně
    
    # Pomocné indikátory (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Target: Směr pro příští den
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# --- KONFIGURACE (Americké akcie, např. AAPL, NVDA, TSLA, MSFT) ---
ticker_akcie = "NVDA"  
data = priprav_data_s_atr(ticker_akcie, perioda="2y")

features = ['Dist_Fib_382', 'Dist_Fib_500', 'Dist_Fib_618', 'RSI', 'ATR_Percent']
X = data[features]
y = data['Target']

# Příprava dat pro dnešek a historii
X_dnes = X.tail(1)
X_historie = X.iloc[:-1]
y_historie = y.iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(X_historie, y_historie, test_size=0.2, shuffle=False)

# Trénování XGBoost
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Predikce
predikce = model.predict(X_dnes)
pravdepodobnost = model.predict_proba(X_dnes)

# Aktuální hodnoty z posledního řádku dat
aktualni_cena = float(data['Close'].iloc[-1])
atr = float(data['ATR'].iloc[-1])
fib_618 = float(data['Fib_618'].iloc[-1])
fib_382 = float(data['Fib_382'].iloc[-1])
high_50 = float(data['High_50'].iloc[-1])
low_50 = float(data['Low_50'].iloc[-1])

# --- LOGIKA VÝPOČTU SL A TP S POUŽITÍM ATR ---
nasobek_atr = 1.0 # Kolikrát ATR chceme odečíst/přičíst k Fib hladině

if predikce == 1:  # BUY (Nákup)
    smer_text = "NÁKUP (BUY) 📈"
    jistota = pravdepodobnost[0][1] * 100
    # Stop-Loss: Zlatý řez MÍNUS bezpečný polštář ATR
    stop_loss = fib_618 - (atr * nasobek_atr)
    # Take-Profit: Rezistence pod Swing High
    take_profit = high_50 - (atr * 0.5)
else:  # SELL (Prodej/Short)
    smer_text = "PRODEJ (SELL/SHORT) 📉"
    jistota = pravdepodobnost[0][0] * 100
    # Stop-Loss: Fib 38.2% PLUS bezpečný polštář ATR
    stop_loss = fib_382 + (atr * nasobek_atr)
    # Take-Profit: Support nad Swing Low
    take_profit = low_50 + (atr * 0.5)

# --- VÝSTUP PRO RUČNÍ ZADÁNÍ DO XTB ---
print("\n" + "="*60)
print(f" NÁVOD PRO PŘÍKAZ U XTB (CFD na {ticker_akcie})")
print("="*60)
print(f"Doporučený směr:      {smer_text}")
print(f"Pravděpodobnost:      {jistota:.2f} %")
print(f"Aktuální tržní cena:  {aktualni_cena:.2f} USD")
print(f"Aktuální 14D ATR:     {atr:.2f} USD (Průměrný denní pohyb)")
print("-"*60)
print(f"-> DO XTB NASTAV STOP-LOSS:   {stop_loss:.2f} USD")
print(f"-> DO XTB NASTAV TAKE-PROFIT:  {take_profit:.2f} USD")
print("="*60)

# --- GRAFICKÁ VIZUALIZACE ---
plt.figure(figsize=(12, 6))
posledni_dny = data.tail(60)
plt.plot(posledni_dny.index, posledni_dny['Close'], color='black', label=f'Cena {ticker_akcie}', linewidth=2)
plt.plot(posledni_dny.index, posledni_dny['Fib_618'], color='purple', linestyle='--', alpha=0.5, label='Fib 61.8%')
plt.plot(posledni_dny.index, posledni_dny['Fib_382'], color='orange', linestyle='--', alpha=0.5, label='Fib 38.2%')

# Zobrazení doporučených úrovní
plt.axhline(y=stop_loss, color='red', linestyle=':', linewidth=2, label=f'SL s ATR polštářem ({stop_loss:.2f})')
plt.axhline(y=take_profit, color='green', linestyle=':', linewidth=2, label=f'TP ({take_profit:.2f})')

plt.title(f"Pokročilá CFD analýza pro {ticker_akcie} (Fibonacci + ATR)")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.show()
