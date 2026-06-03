import datetime
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
