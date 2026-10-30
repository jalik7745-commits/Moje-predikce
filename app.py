    # Trénování modelu
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Zjištění historické úspěšnosti na testovacích datech
    y_pred_test = model.predict(X_test)
    historicka_presnost = accuracy_score(y_test, y_pred_test) * 100

    # PREDOPVĚĎ - OPRAVENO: vytáhneme nultý prvek z pole [0]
    vysledek_predikce = model.predict(X_dnes)
    predikce = int(vysledek_predikce[0]) 
    
    vysledek_pravdepodobnosti = model.predict_proba(X_dnes)
    # Pro nákup (1) bereme pravděpodobnost na indexu 1, pro prodej (0) na indexu 0
    jistota = float(vysledek_pravdepodobnosti[0][predikce]) * 100

    # Bezpečné stažení skalárních hodnot
    aktualni_cena = float(data['Close'].iloc[-1])
    atr = float(data['ATR'].iloc[-1])
    fib_618 = float(data['Fib_618'].iloc[-1])
    fib_382 = float(data['Fib_382'].iloc[-1])
    high_50 = float(data['High_50'].iloc[-1])
    low_50 = float(data['Low_50'].iloc[-1])

    nasobek_atr = 1.0

    if predikce == 1:
        smer_text = "NÁKUP (BUY) 📈"
        stop_loss = fib_618 - (atr * nasobek_atr)
        take_profit = high_50 - (atr * 0.5)
    else:
        smer_text = "PRODEJ (SELL/SHORT) 📉"
        stop_loss = fib_382 + (atr * nasobek_atr)
        take_profit = low_50 + (atr * 0.5)

    # Zobrazení výsledků ve Streamlitu
    st.subheader(f"📋 Pokyny pro ruční zadání do XTB ({ticker_akcie})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Doporučený směr", value=smer_text)
        st.metric(label="Aktuální cena", value=f"{aktualni_cena:.2f} USD")
    with col2:
        st.metric(label="Jistota pro tento obchod", value=f"{jistota:.2f} %")
        st.metric(label="Denní volatilita (ATR)", value=f"{atr:.2f} USD")
    with col3:
        st.metric(label="Historická přesnost modelu", value=f"{historicka_presnost:.1f} %", 
                  help="Úspěšnost modelu na historických datech této konkrétní akcie, která nebyla použita k trénování.")

    st.info(f"**STOP-LOSS (SL):** {stop_loss:.2f} USD | **TAKE-PROFIT (TP):** {take_profit:.2f} USD")

    # --- GRAF PRO STREAMLIT ---
    fig, ax = plt.subplots(figsize=(10, 4))
    posledni_dny = data.tail(60)
    ax.plot(posledni_dny.index, posledni_dny['Close'], color='black', label=f'Cena {ticker_akcie}', linewidth=2)
    ax.plot(posledni_dny.index, posledni_dny['Fib_618'], color='purple', linestyle='--', alpha=0.5, label='Fib 61.8%')
    ax.plot(posledni_dny.index, posledni_dny['Fib_382'], color='orange', linestyle='--', alpha=0.5, label='Fib 38.2%')
    ax.axhline(y=stop_loss, color='red', linestyle=':', linewidth=2, label=f'SL ({stop_loss:.2f})')
    ax.axhline(y=take_profit, color='green', linestyle=':', linewidth=2, label=f'TP ({take_profit:.2f})')
    
    ax.set_title(f"Technická mapa pro {ticker_akcie}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
