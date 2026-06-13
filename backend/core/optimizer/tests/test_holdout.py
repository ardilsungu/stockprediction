"""Holdout (out-of-sample) değerlendirme testleri — Prompt B.

Tümü NSGA-III ÇALIŞTIRMADAN hızlı koşar: train/test bölme saf bir yardımcı,
metrik-pencere mantığı ise select_portfolio_strategies'e elle kurulmuş küçük
pareto girdisiyle doğrudan çağrılarak test edilir.
"""
import numpy as np
import pandas as pd
import pytest

from core.optimizer.nsga3 import (
    chronological_train_test_split,
    select_portfolio_strategies,
    compute_portfolio_return,
    HOLDOUT_TEST_FRACTION,
    MIN_TRAIN_OBS,
    MIN_TEST_OBS,
)


def _df(n, n_assets=1, start='2024-01-01'):
    rng = np.random.default_rng(0)
    cols = [f'C{i}' for i in range(n_assets)]
    return pd.DataFrame(
        rng.normal(0.001, 0.02, size=(n, n_assets)),
        columns=cols,
        index=pd.date_range(start, periods=n),
    )


class TestChronologicalSplit:
    def test_proportions_and_no_overlap(self):
        df = _df(200)
        train, test = chronological_train_test_split(df, test_fraction=0.2)
        assert len(train) == 160
        assert len(test) == 40
        # birleşim = tüm veri, kesişim boş, sıra korunur
        assert len(train) + len(test) == len(df)
        assert list(train.index) + list(test.index) == list(df.index)

    def test_train_is_earlier_test_is_latest(self):
        # Kronolojik: train tamamen test'ten ÖNCE gelmeli (örtüşmesiz).
        df = _df(200)
        train, test = chronological_train_test_split(df)
        assert train.index.max() < test.index.min()
        assert train.index[0] == df.index[0]      # en eski → train
        assert test.index[-1] == df.index[-1]     # en yeni → test

    @pytest.mark.parametrize('frac', [0.1, 0.2, 0.3])
    def test_various_fractions(self, frac):
        df = _df(300)
        train, test = chronological_train_test_split(df, test_fraction=frac)
        assert len(test) == round(300 * frac)
        assert len(train) == 300 - round(300 * frac)

    def test_default_fraction_constant(self):
        df = _df(500)
        train, test = chronological_train_test_split(df)
        assert len(test) == round(500 * HOLDOUT_TEST_FRACTION)

    def test_insufficient_total_raises(self):
        df = _df(50)  # 50 → train 40, test 10 → ikisi de eşik altı
        with pytest.raises(ValueError, match='yetersiz veri'):
            chronological_train_test_split(df)

    def test_insufficient_test_raises(self):
        # total=100, frac=0.05 → test=5 < MIN_TEST_OBS, train yeterli olsa da hata
        df = _df(100)
        with pytest.raises(ValueError):
            chronological_train_test_split(df, test_fraction=0.05)

    def test_boundary_just_enough_passes(self):
        # train=MIN_TRAIN_OBS, test=MIN_TEST_OBS tam sınırda geçmeli
        n = MIN_TRAIN_OBS + MIN_TEST_OBS
        df = _df(n)
        train, test = chronological_train_test_split(
            df, test_fraction=MIN_TEST_OBS / n,
        )
        assert len(train) >= MIN_TRAIN_OBS
        assert len(test) >= MIN_TEST_OBS


def _make_pareto(n_assets=6, n_sol=6, seed=0):
    """Geçerli (toplam=1) rastgele ağırlık vektörleri + sentetik F."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n_sol, n_assets))
    W = raw / raw.sum(axis=1, keepdims=True)
    F = rng.random((n_sol, 2))
    return W, F


class TestOutOfSampleMetricWindow:
    def test_metrics_computed_on_eval_window(self):
        cols = [f'C{i}' for i in range(6)]
        rng = np.random.default_rng(1)
        train = pd.DataFrame(rng.normal(0.002, 0.03, (150, 6)), columns=cols,
                             index=pd.date_range('2023-01-01', periods=150))
        test = pd.DataFrame(rng.normal(-0.001, 0.05, (60, 6)), columns=cols,
                            index=pd.date_range('2023-06-01', periods=60))
        W, F = _make_pareto()
        strat = select_portfolio_strategies(W, F, train, eval_returns_df=test)
        # annual_return DOĞRUDAN test penceresinden hesaplanmış olmalı.
        for s in strat.values():
            w = s['weights']
            expected = compute_portfolio_return(w, test.mean().values) * 365
            assert abs(s['annual_return'] - expected) < 1e-9

    def test_in_sample_vs_out_of_sample_differ(self):
        # Aynı ağırlıklar, farklı pencere → metrikler farklı olmalı.
        cols = [f'C{i}' for i in range(6)]
        rng = np.random.default_rng(2)
        train = pd.DataFrame(rng.normal(0.003, 0.02, (150, 6)), columns=cols,
                             index=pd.date_range('2023-01-01', periods=150))
        test = pd.DataFrame(rng.normal(-0.002, 0.06, (60, 6)), columns=cols,
                            index=pd.date_range('2023-06-01', periods=60))
        W, F = _make_pareto()
        ins = select_portfolio_strategies(W, F, train)
        oos = select_portfolio_strategies(W, F, train, eval_returns_df=test)
        diffs = [abs((ins[k]['sharpe'] or 0.0) - (oos[k]['sharpe'] or 0.0))
                 for k in ins]
        assert max(diffs) > 1e-6

    def test_selection_uses_train_not_eval(self):
        # eval_returns_df verilse de SEÇİLEN indeksler train'e dayanmalı:
        # eval=None ve eval=test çağrılarında pareto_index'ler aynı kalmalı.
        cols = [f'C{i}' for i in range(6)]
        rng = np.random.default_rng(5)
        train = pd.DataFrame(rng.normal(0.002, 0.03, (150, 6)), columns=cols,
                             index=pd.date_range('2023-01-01', periods=150))
        test = pd.DataFrame(rng.normal(-0.01, 0.07, (60, 6)), columns=cols,
                            index=pd.date_range('2023-06-01', periods=60))
        W, F = _make_pareto()
        ins = select_portfolio_strategies(W, F, train)
        oos = select_portfolio_strategies(W, F, train, eval_returns_df=test)
        for k in ins:
            assert ins[k]['pareto_index'] == oos[k]['pareto_index']

    def test_pareto_index_present_and_in_range(self):
        W, F = _make_pareto(n_sol=6)
        df = _df(120, n_assets=6)
        strat = select_portfolio_strategies(W, F, df)
        for s in strat.values():
            assert 'pareto_index' in s
            assert 0 <= s['pareto_index'] < len(W)

    def test_backward_compatible_without_eval(self):
        # eval verilmezse metrikler returns_df (in-sample) üzerinde — eski davranış.
        cols = [f'C{i}' for i in range(6)]
        df = pd.DataFrame(np.random.default_rng(7).normal(0.001, 0.02, (120, 6)),
                          columns=cols, index=pd.date_range('2023-01-01', periods=120))
        W, F = _make_pareto()
        strat = select_portfolio_strategies(W, F, df)
        for s in strat.values():
            w = s['weights']
            expected = compute_portfolio_return(w, df.mean().values) * 365
            assert abs(s['annual_return'] - expected) < 1e-9
