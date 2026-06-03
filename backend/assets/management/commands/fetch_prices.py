import datetime
import yfinance as yf
from django.core.management.base import BaseCommand
from django.utils import timezone

from assets.models import Asset, Price


class Command(BaseCommand):
    help = "Aktif asset'ler için yfinance'den geçmiş fiyat verisi çeker ve DB'ye yazar."

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=750,
            help='Kaç günlük geçmiş veri çekileceği (varsayılan: 750)',
        )

    def handle(self, *args, **options):
        days = options['days']
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days)

        assets = Asset.objects.filter(is_active=True)
        total  = assets.count()

        if total == 0:
            self.stdout.write('Aktif asset bulunamadı, işlem tamamlandı.')
            return

        self.stdout.write(
            f'{total} aktif asset için {start} → {end} fiyat verisi çekiliyor...'
        )

        created_total = 0
        skipped       = 0

        for asset in assets:
            try:
                ticker = yf.Ticker(f'{asset.symbol}-USD')
                df = ticker.history(
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    auto_adjust=True,
                )

                if df.empty:
                    self.stdout.write(
                        self.style.WARNING(
                            f'  [{asset.symbol}] WARNING: Veri döndürülmedi, atlanıyor.'
                        )
                    )
                    skipped += 1
                    continue

                prices = []
                for ts, row in df.iterrows():
                    prices.append(Price(
                        asset=asset,
                        date=ts.date(),
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        volume=int(row['Volume']) if row['Volume'] else None,
                        source='yfinance',
                        fetched_at=timezone.now(),
                    ))

                Price.objects.bulk_create(
                    prices,
                    update_conflicts=True,
                    update_fields=['open', 'high', 'low', 'close', 'volume', 'source', 'fetched_at'],
                    unique_fields=['asset', 'date'],
                )

                self.stdout.write(
                    f'  [{asset.symbol}] {len(prices)} kayıt eklendi/güncellendi.'
                )
                created_total += len(prices)

            except Exception as exc:
                self.stdout.write(
                    self.style.ERROR(
                        f'  [{asset.symbol}] HATA: {type(exc).__name__}: {exc} — atlanıyor.'
                    )
                )
                skipped += 1
                continue

        self.stdout.write(
            self.style.SUCCESS(
                f'Tamamlandı: {created_total} kayıt işlendi, {skipped} asset atlandı.'
            )
        )
