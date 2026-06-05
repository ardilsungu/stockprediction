import datetime
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from core.optimizer.nsga3 import load_prices_from_db_or_fetch


def _price_rows(symbol, n_days=400):
    today = datetime.date.today()
    rows = []
    for i in range(n_days):
        rows.append({
            'asset__symbol': symbol,
            'date': today - datetime.timedelta(days=n_days - i),
            'close': 100.0 + i * 0.1,
        })
    return rows


@pytest.mark.django_db
class TestLoadPricesFromDbOrFetch:
    def test_db_sufficient_skips_yfinance(self, db):
        from assets.models import Asset, Price as PriceModel

        asset = Asset.objects.create(symbol='BTC', name='Bitcoin', is_active=True)
        today = datetime.date.today()
        # lookback_days=365 → expected ≈ 365*5/7*0.8 ≈ 209 satır; 370 satır yeterli
        for i in range(370):
            PriceModel.objects.create(
                asset=asset,
                date=today - datetime.timedelta(days=400 - i),
                open=100, high=110, low=90, close=105,
            )

        with patch('core.optimizer.nsga3.load_prices_from_yfinance') as mock_yf:
            result = load_prices_from_db_or_fetch(['BTC'], lookback_days=365)

        mock_yf.assert_not_called()
        assert isinstance(result, pd.DataFrame)
        assert 'BTC' in result.columns

    def test_db_insufficient_calls_yfinance(self, db):
        from assets.models import Asset, Price as PriceModel

        asset = Asset.objects.create(symbol='ETH', name='Ethereum', is_active=True)
        today = datetime.date.today()
        # Sadece 5 kayıt — kesinlikle yetersiz
        for i in range(5):
            PriceModel.objects.create(
                asset=asset,
                date=today - datetime.timedelta(days=10 - i),
                open=2000, high=2100, low=1900, close=2050,
            )

        mock_returns = pd.DataFrame(
            {'ETH': [0.01, -0.02, 0.005] * 20},
            index=pd.date_range('2024-01-01', periods=60),
        )
        with patch('core.optimizer.nsga3.load_prices_from_yfinance',
                   return_value=mock_returns) as mock_yf:
            result = load_prices_from_db_or_fetch(['ETH'], lookback_days=365)

        mock_yf.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_asset_does_not_exist_falls_back_to_yfinance(self, db):
        mock_returns = pd.DataFrame(
            {'XRP': [0.01, -0.02, 0.005] * 20},
            index=pd.date_range('2024-01-01', periods=60),
        )
        with patch('core.optimizer.nsga3.load_prices_from_yfinance',
                   return_value=mock_returns) as mock_yf:
            result = load_prices_from_db_or_fetch(['XRP'], lookback_days=365)

        mock_yf.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_return_type_is_always_dataframe(self, db):
        mock_returns = pd.DataFrame(
            {'SOL': [0.01, 0.02] * 30},
            index=pd.date_range('2024-01-01', periods=60),
        )
        with patch('core.optimizer.nsga3.load_prices_from_yfinance',
                   return_value=mock_returns):
            result = load_prices_from_db_or_fetch(['SOL'], lookback_days=365)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


@pytest.mark.django_db
class TestDbPathDeterminism:
    """DB yolunda sütun sırası deterministik olmalı."""

    def _create_prices(self, symbols, n_days=400):
        from assets.models import Asset, Price as PriceModel
        today = datetime.date.today()
        for symbol in symbols:
            asset = Asset.objects.create(symbol=symbol, name=symbol, is_active=True)
            for i in range(n_days):
                PriceModel.objects.create(
                    asset=asset,
                    date=today - datetime.timedelta(days=n_days - i),
                    open=100, high=110, low=90,
                    close=100.0 + i * 0.1,
                )

    def test_column_order_is_reproducible(self, db):
        # İki ardışık çağrı aynı sütun sırasını döndürmeli
        symbols = ['SOL', 'BTC', 'ETH']
        self._create_prices(symbols)
        df1 = load_prices_from_db_or_fetch(symbols, lookback_days=365)
        df2 = load_prices_from_db_or_fetch(symbols, lookback_days=365)
        assert list(df1.columns) == list(df2.columns)

    def test_columns_are_alphabetically_sorted(self, db):
        # DB yolu: sütunlar her zaman alfabetik sırada olmalı
        symbols = ['SOL', 'BTC', 'ETH']
        self._create_prices(symbols)
        df = load_prices_from_db_or_fetch(symbols, lookback_days=365)
        assert list(df.columns) == sorted(df.columns)

    def test_non_alpha_input_order_gives_sorted_columns(self, db):
        # Girdi ters alfabetik olsa bile çıktı sıralı olmalı
        # (order_by + sorted(prices.columns) her ikisini de doğrular)
        symbols = ['XRP', 'ADA', 'BTC']
        self._create_prices(symbols)
        df = load_prices_from_db_or_fetch(symbols, lookback_days=365)
        assert list(df.columns) == sorted(df.columns)
        for s in symbols:
            assert s in df.columns

    def test_seed_determinism_with_sorted_columns(self, db):
        # Aynı seed ile üretilen ağırlıklar, iki çalışmada aynı varlığa eşlenmeli
        symbols = ['SOL', 'ETH', 'BTC']
        self._create_prices(symbols)
        df1 = load_prices_from_db_or_fetch(symbols, lookback_days=365)
        df2 = load_prices_from_db_or_fetch(symbols, lookback_days=365)
        assert list(df1.columns) == list(df2.columns), "Sütun sırası farklı"
        np.random.seed(42)
        w1 = np.random.dirichlet(np.ones(len(df1.columns)))
        np.random.seed(42)
        w2 = np.random.dirichlet(np.ones(len(df2.columns)))
        for i, (col1, col2) in enumerate(zip(df1.columns, df2.columns)):
            assert col1 == col2
            assert abs(w1[i] - w2[i]) < 1e-12
