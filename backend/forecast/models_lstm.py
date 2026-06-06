import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from assets.models import Asset, Price
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)

LOOKBACK = 60

def build_lstm_model(lookback: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_lstm_forecast(asset: Asset, horizon_days: int = 30) -> dict:
    """
    Verilen asset için LSTM modeli ile fiyat tahmini yapar.

    Parameters
    ----------
    asset : Asset
        Tahmin yapılacak kripto varlık
    horizon_days : int
        Kaç günlük tahmin yapılacak (varsayılan: 30)

    Returns
    -------
    dict : {predictions, mae, rmse}
        predictions: [{date, predicted, lower_ci, upper_ci}]
        mae: float
        rmse: float
    """
    if not 1 <= horizon_days <= 365:
        raise ValueError(f"horizon_days 1-365 arasında olmalı, verilen: {horizon_days}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # 1. Veriyi DB'den çek
    prices = list(Price.objects.filter(
        asset=asset,
        date__gte=date.today() - timedelta(days=730)
    ).order_by('date').values_list('date', 'close'))

    data = np.array([float(close) for _, close in prices]).reshape(-1, 1)

    if len(data) < LOOKBACK + 50:
        raise ValueError(f"{asset.symbol} için yeterli veri yok. "
                        f"Mevcut: {len(data)} gün, gereken: {LOOKBACK + 50} gün.")

    # 2. Normalize et
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # 3. Sequence'ları oluştur
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    # 4. Train/test split (%80/%20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Modeli eğit
    model = build_lstm_model(LOOKBACK)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    # 6. Recursive forecasting
    last_window = scaled[-LOOKBACK:].tolist()
    predictions_scaled = []
    for _ in range(horizon_days):
        window = np.array(last_window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred = model.predict(window, verbose=0)[0][0]
        predictions_scaled.append(pred)
        last_window.append([pred])

    predictions_raw = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()

    # 7. Test MAE/RMSE
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = float(np.abs(y_pred_inv - y_test_inv).mean())
    rmse = float(np.sqrt(((y_pred_inv - y_test_inv) ** 2).mean()))

    # 8. Tarihleri ekle
    last_date = prices[-1][0]
    predictions = [
        {
            'date': (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            'predicted': round(float(v), 4),
            'lower_ci': round(float(v) * 0.92, 4),
            'upper_ci': round(float(v) * 1.08, 4),
        }
        for i, v in enumerate(predictions_raw)
    ]

    logger.info(f"LSTM forecast tamamlandi: {asset.symbol}, "
               f"horizon={horizon_days}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    return {'predictions': predictions, 'mae': mae, 'rmse': rmse}
