"""ML metrik hesaplama testleri.

- TestChronologicalSplitIndices: saf birim testleri, ML kütüphanesi import
  etmeden hızlı koşar.
- Entegrasyon testleri: küçük sentetik seriyle gerçek prophet/tensorflow
  çalıştırır; LSTM testinde LOOKBACK/EPOCHS monkeypatch ile küçültülür.
"""
import math
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from assets.models import Asset, Price
from forecast.data_split import chronological_split_indices, naive_baseline_metrics


class TestChronologicalSplitIndices:
    def test_default_ratios_70_15_15(self):
        train_end, val_end = chronological_split_indices(100)
        assert train_end == 70
        assert val_end == 85

    def test_parts_chronological_disjoint_and_complete(self):
        n = 137
        train_end, val_end = chronological_split_indices(n)
        idx = list(range(n))
        train, val, test = idx[:train_end], idx[train_end:val_end], idx[val_end:]
        # Sıralı dilimleme: parçalar örtüşmez, tüm diziyi kapsar, kronolojiktir
        assert train + val + test == idx
        assert max(train) < min(val)
        assert max(val) < min(test)

    def test_every_part_nonempty_for_small_n(self):
        for n in range(3, 30):
            train_end, val_end = chronological_split_indices(n)
            assert 1 <= train_end < val_end < n, f"n={n} için boş parça oluştu"

    def test_fewer_than_three_samples_raises(self):
        with pytest.raises(ValueError, match='en az 3'):
            chronological_split_indices(2)

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError, match='split oranları'):
            chronological_split_indices(100, train_frac=0.8, val_frac=0.3)


class TestNaiveBaselineMetrics:
    def test_known_series_hand_computed(self):
        # Test dönemi: [4, 7]; persistence tahminleri: [2, 4]
        # Hatalar: 2 ve 3 → MAE = 2.5, RMSE = sqrt((4 + 9) / 2)
        mae, rmse = naive_baseline_metrics([1.0, 2.0, 4.0, 7.0], test_start=2)
        assert mae == pytest.approx(2.5)
        assert rmse == pytest.approx(math.sqrt(6.5))

    def test_perfect_persistence_gives_zero_error(self):
        mae, rmse = naive_baseline_metrics([5.0, 5.0, 5.0, 5.0], test_start=1)
        assert mae == 0.0
        assert rmse == 0.0

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError, match='en az 2 değer'):
            naive_baseline_metrics([1.0], test_start=1)

    def test_test_start_out_of_range_raises(self):
        with pytest.raises(ValueError, match='test_start'):
            naive_baseline_metrics([1.0, 2.0, 3.0], test_start=0)
        with pytest.raises(ValueError, match='test_start'):
            naive_baseline_metrics([1.0, 2.0, 3.0], test_start=3)


@pytest.fixture
def asset(db):
    return Asset.objects.create(
        symbol='BTC',
        name='Bitcoin',
        market_cap=1_000_000_000,
        category='crypto',
    )


def _create_price_series(asset, n_days):
    """Hafif trend + haftalık dalgalanma içeren sentetik günlük seri üretir."""
    today = date.today()
    rows = []
    for i in range(n_days):
        p = Decimal(str(round(100 + 0.1 * i + 5 * math.sin(i / 7), 4)))
        rows.append(Price(
            asset=asset,
            date=today - timedelta(days=n_days - i),
            open=p,
            high=p,
            low=p,
            close=p,
        ))
    Price.objects.bulk_create(rows)


def _assert_valid_forecast_result(result, horizon_days):
    for key in ('mae', 'rmse', 'baseline_mae', 'baseline_rmse'):
        assert result[key] is not None
        assert isinstance(result[key], float)
        assert result[key] >= 0
    preds = result['predictions']
    assert len(preds) == horizon_days
    for p in preds:
        assert set(p.keys()) == {'date', 'predicted', 'lower_ci', 'upper_ci'}
        datetime.strptime(p['date'], '%Y-%m-%d')
        assert p['lower_ci'] <= p['predicted'] <= p['upper_ci']
    dates = [p['date'] for p in preds]
    assert dates == sorted(dates)


@pytest.mark.django_db
class TestProphetMetrics:
    def test_below_min_data_raises_meaningful_error(self, asset):
        from forecast.models_prophet import run_prophet_forecast, MIN_DATA_DAYS
        _create_price_series(asset, MIN_DATA_DAYS - 1)
        with pytest.raises(ValueError, match=f'gereken: {MIN_DATA_DAYS} gün'):
            run_prophet_forecast(asset, horizon_days=5)

    def test_holdout_metrics_and_output_format(self, asset):
        from forecast.models_prophet import run_prophet_forecast
        _create_price_series(asset, 100)
        result = run_prophet_forecast(asset, horizon_days=5)
        _assert_valid_forecast_result(result, horizon_days=5)


@pytest.mark.django_db
class TestLSTMMetrics:
    def test_below_min_data_raises_meaningful_error(self, asset):
        from forecast import models_lstm
        required = models_lstm.LOOKBACK + models_lstm.MIN_SAMPLES
        _create_price_series(asset, required - 1)
        with pytest.raises(ValueError, match=f'gereken: {required} gün'):
            models_lstm.run_lstm_forecast(asset, horizon_days=5)

    def test_metrics_and_output_format_with_tiny_training(self, asset, monkeypatch):
        from forecast import models_lstm
        # Eğitimi test için küçült: kısa lookback + 2 epoch
        monkeypatch.setattr(models_lstm, 'LOOKBACK', 10)
        monkeypatch.setattr(models_lstm, 'EPOCHS', 2)
        _create_price_series(asset, 130)  # gereken: 10 + 100 = 110
        result = models_lstm.run_lstm_forecast(asset, horizon_days=3)
        _assert_valid_forecast_result(result, horizon_days=3)
