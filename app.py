def priprav_data_s_atr(ticker, perioda="2y"):
    try:
        # Stažení dat bez automatického formátování na skupiny
        df = yf.download(ticker, period=perioda, group_by='column')
        if df.empty:
            st.error(f"Nepodařilo se stáhnout žádná data pro ticker {ticker}. Ověřte, zda je správný.")
            return None
        
        # AGRESIVNÍ OŠETŘENÍ HLAVIČKY (MULTIINDEXU)
        # Pokud hlavička obsahuje více úrovní (např. High a pod tím NVDA), 
        # vezmeme pouze první úroveň a převedeme ji na text.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]
            
        # Pro jistotu odstraníme případné mezery v názvech sloupců
        df.columns = df.columns.str.strip()
        
        # Kontrola, zda máme všechny potřebné sloupce
        potrebne_sloupce = ['High', 'Low', 'Close', 'Open']
        if not all(col in df.columns for col in potrebne_sloupce):
            st.error(f"Stažená data neobsahují standardní sloupce. Nalezené sloupce: {list(df.columns)}")
            return None
        
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
