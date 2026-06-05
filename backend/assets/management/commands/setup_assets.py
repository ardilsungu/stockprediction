import datetime

import pandas as pd
import yfinance as yf
from django.core.management.base import BaseCommand
from django.utils import timezone

from assets.models import Asset, Price
from core.optimizer.nsga3 import apply_sanity_filter, fetch_top_coins


class Command(BaseCommand):
    help = (
        "CoinGecko top-N coin'i (optimizer ile aynı filtreler) çeker, yfinance'ten "
        "OHLCV indirir, apply_sanity_filter ile temizler ve hayatta kalanları "
        "Asset + Price tablolarına yazar."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--days', type=int, default=750,
            help='Kaç günlük geçmiş OHLCV çekileceği (varsayılan: 750)',
        )
        parser.add_argument(
            '--top-n', type=int, default=100,
            help='CoinGecko\'dan çekilecek coin sayısı (varsayılan: 100)',
        )

    def handle(self, *args, **options):
        days  = options['days']
        top_n = options['top_n']
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days)

        # ── 1) CoinGecko top-N (optimizer ile birebir aynı filtreler) ──────
        self.stdout.write(f'[1/5] CoinGecko top-{top_n} çekiliyor (filtreli)...')
        coins_df = fetch_top_coins(
            n_target=top_n,
            sort_by='market_cap',
            min_market_cap_usd=1e9,
            exclude_stables=True,
            exclude_exchange=True,
            exclude_wrapped=True,
        )
        if coins_df.empty or 'symbol' not in coins_df.columns:
            self.stderr.write(self.style.ERROR(
                'Coin listesi oluşturulamadı (API + fallback boş). İşlem durdu.'
            ))
            return

        symbols  = coins_df['symbol'].tolist()
        name_map = dict(zip(coins_df['symbol'], coins_df['name']))
        self.stdout.write(f'  Filtre sonrası {len(symbols)} coin.')

        # ── 2) yfinance OHLCV indir (SYMBOL-USD) ──────────────────────────
        self.stdout.write(
            f'[2/5] yfinance OHLCV indiriliyor ({start} → {end})...'
        )
        ohlcv = {}
        for sym in symbols:
            try:
                df = yf.Ticker(f'{sym}-USD').history(
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    auto_adjust=True,
                )
                if df.empty:
                    self.stdout.write(self.style.WARNING(
                        f'  [{sym}] veri yok, atlandı.'
                    ))
                    continue
                ohlcv[sym] = df
            except Exception as exc:
                self.stdout.write(self.style.WARNING(
                    f'  [{sym}] HATA {type(exc).__name__}: {exc} — atlandı.'
                ))
                continue
        self.stdout.write(f'  {len(ohlcv)} coin için veri indirildi.')

        if not ohlcv:
            self.stderr.write(self.style.ERROR('Hiç OHLCV indirilemedi. İşlem durdu.'))
            return

        # ── 3) Close fiyatlarından günlük returns ─────────────────────────
        self.stdout.write('[3/5] Günlük returns hesaplanıyor (pct_change)...')
        close_df = pd.DataFrame({sym: df['Close'] for sym, df in ohlcv.items()})
        close_df = close_df.sort_index().ffill(limit=2)
        returns_df = close_df.pct_change().iloc[1:]

        # ── 4) Sanity filter (optimizer ile aynı eşikler) ─────────────────
        self.stdout.write('[4/5] apply_sanity_filter uygulanıyor...')
        clean_df  = apply_sanity_filter(returns_df, verbose=True)
        survivors = list(clean_df.columns)
        self.stdout.write(f'  {len(survivors)} coin filtreyi geçti.')

        # ── 5) Asset + Price kaydet ───────────────────────────────────────
        self.stdout.write('[5/5] Asset + Price yazılıyor...')
        created_assets = 0
        total_prices   = 0
        for sym in survivors:
            asset, created = Asset.objects.get_or_create(
                symbol=sym,
                defaults={
                    'name':      name_map.get(sym, sym),
                    'category':  'crypto',
                    'is_active': True,
                },
            )
            if created:
                created_assets += 1

            df = ohlcv[sym]
            prices = [
                Price(
                    asset=asset,
                    date=ts.date(),
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume']) if row['Volume'] else None,
                    source='yfinance',
                    fetched_at=timezone.now(),
                )
                for ts, row in df.iterrows()
            ]
            Price.objects.bulk_create(
                prices,
                update_conflicts=True,
                update_fields=['open', 'high', 'low', 'close', 'volume', 'source', 'fetched_at'],
                unique_fields=['asset', 'date'],
            )
            total_prices += len(prices)
            self.stdout.write(f'  [{sym}] {len(prices)} Price kaydı eklendi/güncellendi.')

        # ── Özet rapor ────────────────────────────────────────────────────
        self.stdout.write(self.style.SUCCESS(
            f'\nTamamlandı:\n'
            f'  CoinGecko filtre sonrası : {len(symbols)} coin\n'
            f'  yfinance veri indirildi  : {len(ohlcv)} coin\n'
            f'  Sanity filter sonrası    : {len(survivors)} coin\n'
            f'  Yeni Asset kaydı         : {created_assets}\n'
            f'  Toplam Price kaydı       : {total_prices}'
        ))
