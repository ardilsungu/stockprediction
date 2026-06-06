import pandas as pd
from prophet import Prophet
from assets.models import Asset, Price
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)

def run_prophet_forecast(asset: Asset, horizon_days: int = 30) -> dict:
    """
    Verilen asset için Prophet modeli ile fiyat tahmini yapar.

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

    # 1. Veriyi DB'den çek
    prices = Price.objects.filter(
        asset=asset,
        date__gte=date.today() - timedelta(days=730)
    ).order_by('date').values('date', 'close')

    df = pd.DataFrame(list(prices))

    if df.empty or len(df) < 60:
        raise ValueError(f"{asset.symbol} için yeterli fiyat verisi yok. "
                        f"Mevcut: {len(df)} gün, gereken: 60 gün.")

    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)

    # 2. Modeli eğit
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)

    # 3. Tahmin yap
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    # 4. Sadece gelecek günleri döndür
    result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon_days)
    predictions = [
        {
            'date': row['ds'].strftime('%Y-%m-%d'),
            'predicted': round(float(row['yhat']), 4),
            'lower_ci': round(float(row['yhat_lower']), 4),
            'upper_ci': round(float(row['yhat_upper']), 4),
        }
        for _, row in result_df.iterrows()
    ]

    # 5. In-sample MAE/RMSE hesapla
    in_sample = forecast[forecast['ds'] <= df['ds'].max()]
    merged = df.merge(in_sample[['ds', 'yhat']], on='ds')
    mae = float((merged['y'] - merged['yhat']).abs().mean())
    rmse = float(((merged['y'] - merged['yhat']) ** 2).mean() ** 0.5)

    logger.info(f"Prophet forecast tamamlandı: {asset.symbol}, "
               f"horizon={horizon_days}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    return {'predictions': predictions, 'mae': mae, 'rmse': rmse}
