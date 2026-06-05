import datetime
import yfinance as yf
from celery import shared_task
from celery.utils.log import get_task_logger
from django.utils import timezone

from assets.models import Asset, Price

logger = get_task_logger(__name__)


@shared_task
def daily_price_snapshot():
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    start_str = yesterday.strftime('%Y-%m-%d')
    # yfinance end date exclusive; +1 gün veriyoruz ki dünü kapsasın
    end_str = (yesterday + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    assets = Asset.objects.filter(is_active=True)
    total = assets.count()
    logger.info("daily_price_snapshot başladı: %d aktif asset, tarih=%s", total, yesterday)

    saved = 0
    skipped = 0

    for asset in assets:
        try:
            ticker = yf.Ticker(f'{asset.symbol}-USD')
            df = ticker.history(start=start_str, end=end_str, auto_adjust=True)

            if df.empty:
                logger.warning("[%s] Veri döndürülmedi, atlanıyor.", asset.symbol)
                skipped += 1
                continue

            row = df.iloc[0]
            Price.objects.update_or_create(
                asset=asset,
                date=yesterday,
                defaults={
                    'open':       row['Open'],
                    'high':       row['High'],
                    'low':        row['Low'],
                    'close':      row['Close'],
                    'volume':     int(row['Volume']) if row['Volume'] else None,
                    'source':     'yfinance',
                    'fetched_at': timezone.now(),
                },
            )
            logger.info("[%s] %s kaydedildi.", asset.symbol, yesterday)
            saved += 1

        except Exception as exc:
            logger.error(
                "[%s] HATA: %s: %s — atlanıyor.",
                asset.symbol, type(exc).__name__, exc,
            )
            skipped += 1
            continue

    logger.info(
        "daily_price_snapshot tamamlandı: %d kaydedildi, %d atlandı.",
        saved, skipped,
    )
    return {'saved': saved, 'skipped': skipped, 'date': str(yesterday)}
