import numpy as np
import pandas as pd
from prophet import Prophet
from assets.models import Asset, Price
from datetime import date, timedelta
import logging

logger = logging.getLogger(__name__)

# Son %20'lik kronolojik holdout MAE/RMSE için ayrılır; 90 günlük minimumla
# holdout en az 18 gün olur ve metrik anlamlı kalır.
MIN_DATA_DAYS = 90
HOLDOUT_FRAC = 0.2

def _build_prophet_model() -> Prophet:
    return Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
    )

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

    if df.empty or len(df) < MIN_DATA_DAYS:
        raise ValueError(f"{asset.symbol} için yeterli fiyat verisi yok. "
                        f"Mevcut: {len(df)} gün, gereken: {MIN_DATA_DAYS} gün.")

    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)

    # 2. MAE/RMSE — son %20'lik kronolojik holdout üzerinde.
    # Model ilk %80'de fit edilir, görmediği son %20'yi tahmin eder; in-sample
    # metriğin iyimserliği böylece giderilir ve LSTM ile karşılaştırılabilir olur.
    holdout_size = max(1, int(len(df) * HOLDOUT_FRAC))
    train_df = df.iloc[:-holdout_size]
    holdout_df = df.iloc[-holdout_size:]

    eval_model = _build_prophet_model()
    eval_model.fit(train_df)
    holdout_pred = eval_model.predict(holdout_df[['ds']])
    errors = holdout_df['y'].to_numpy() - holdout_pred['yhat'].to_numpy()
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))

    # 3. Kullanıcıya dönen tahminler için final modeli TÜM veriyle eğit
    model = _build_prophet_model()
    model.fit(df)

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

    logger.info(f"Prophet forecast tamamlandı: {asset.symbol}, "
               f"horizon={horizon_days}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    return {'predictions': predictions, 'mae': mae, 'rmse': rmse}
