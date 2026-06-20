"""WeightRepair / repair_weights testleri.

🔴 bulgunun (max_weight kısıtının normalizasyon nedeniyle gerçekte
sağlanmaması) kapandığının kanıtı. Testlerin çoğu NSGA-III ÇALIŞTIRMADAN
doğrudan saf repair fonksiyonunu çağırır → hızlıdır. Sondaki tek mini
uçtan-uca optimizasyon küçük pop/gen ile çalışır.
"""
import numpy as np
import pandas as pd
import pytest

from core.optimizer.nsga3 import (
    repair_weight_vector,
    repair_weights,
    WeightRepair,
    run_nsga3_optimization,
)

TOL = 1e-9


def _assert_valid(w, max_weight):
    """Verification target: max(w) ≤ max_weight, sum(w) ≈ 1, min(w) ≥ 0."""
    assert w.max() <= max_weight + TOL, f"max(w)={w.max()} > {max_weight}"
    assert abs(w.sum() - 1.0) <= 1e-9, f"sum(w)={w.sum()}"
    assert w.min() >= -1e-12, f"min(w)={w.min()}"


class TestRepairWeightVector:
    @pytest.mark.parametrize("max_weight", [0.05, 0.10, 0.20, 0.5])
    def test_single_dominant_genome(self, max_weight):
        # Tek varlık baskın — eski X/sum(X) bunu ~%97-100 yapardı.
        x = np.array([100.0] + [0.0] * 19)
        w = repair_weight_vector(x, max_weight)
        _assert_valid(w, max_weight)

    @pytest.mark.parametrize("max_weight", [0.05, 0.10, 0.20])
    def test_all_equal_genome(self, max_weight):
        x = np.ones(30)
        w = repair_weight_vector(x, max_weight)
        _assert_valid(w, max_weight)

    def test_all_near_zero_genome(self, max_weight=0.10):
        # Hepsi sıfıra çok yakın → dejenere; yine de geçerli vektör üretmeli.
        x = np.full(20, 1e-15)
        w = repair_weight_vector(x, max_weight)
        _assert_valid(w, max_weight)

    def test_negatives_are_clipped(self):
        x = np.array([-5.0, 3.0, -1.0, 8.0, 0.0, 2.0] + [0.0] * 6)
        w = repair_weight_vector(x, 0.20)
        _assert_valid(w, 0.20)
        assert w.min() >= 0.0

    @pytest.mark.parametrize("seed", range(25))
    def test_random_genomes_always_valid(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(10, 60))
        max_weight = float(rng.choice([0.05, 0.10, 0.20, 0.34]))
        # max_weight * n >= 1 fizibilitesini garanti et
        if max_weight * n < 1.0:
            max_weight = 1.0 / n + 0.01
        x = rng.random(n) * rng.choice([1.0, 100.0])
        w = repair_weight_vector(x, max_weight)
        _assert_valid(w, max_weight)

    def test_idempotent_on_feasible_input(self):
        rng = np.random.default_rng(0)
        w1 = repair_weight_vector(rng.random(30), 0.10)
        w2 = repair_weight_vector(w1, 0.10)
        assert np.allclose(w1, w2, atol=1e-12)

    def test_deterministic(self):
        x = np.array([10.0, 0.0, 0.0, 5.0, 0.0] * 4)
        a = repair_weight_vector(x, 0.10)
        b = repair_weight_vector(x.copy(), 0.10)
        assert np.array_equal(a, b)

    def test_infeasible_precondition_raises(self):
        # max_weight * n < 1 → toplam=1 imkânsız → anlamlı hata.
        with pytest.raises(ValueError, match="imkânsız"):
            repair_weight_vector(np.array([1.0, 1.0]), 0.10)  # 0.10*2 = 0.2 < 1

    def test_boundary_precondition_feasible(self):
        # max_weight * n == 1 → tam sınır, feasible (her gen tam max_weight).
        w = repair_weight_vector(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.20)
        _assert_valid(w, 0.20)
        assert np.allclose(w, 0.20)


class TestRepairWeightsBatch:
    def test_batch_matches_rowwise(self):
        rng = np.random.default_rng(1)
        X = rng.random((8, 25))
        batch = repair_weights(X, 0.10)
        for i in range(X.shape[0]):
            assert np.allclose(batch[i], repair_weight_vector(X[i], 0.10))

    def test_batch_all_valid(self):
        rng = np.random.default_rng(2)
        X = rng.random((20, 30))
        for w in repair_weights(X, 0.10):
            _assert_valid(w, 0.10)

    def test_1d_input_returns_1d(self):
        w = repair_weights(np.array([1.0, 2.0, 3.0] * 10), 0.10)
        assert w.ndim == 1


class TestWeightRepairOperator:
    def test_operator_do_delegates_to_repair(self):
        rng = np.random.default_rng(3)
        X = rng.random((6, 20))
        op = WeightRepair(0.10)
        Xp = op._do(None, X)
        assert np.allclose(Xp, repair_weights(X, 0.10))
        for w in Xp:
            _assert_valid(w, 0.10)


class TestMiniEndToEnd:
    """Küçük pop/gen ile gerçek NSGA-III: dönen NİHAİ ağırlıkların hiçbiri
    max_weight'i aşmamalı (🔴 bulgunun uçtan-uca kapandığının kanıtı).

    max_assets >= n_assets seçilir ki cardinality üst sınırı bağlayıcı olmasın
    ve test küçük pop/gen ile hızlı + deterministik feasible kalsın.
    """

    def test_final_weights_respect_max_weight(self):
        rng = np.random.default_rng(7)
        n_assets = 12
        df = pd.DataFrame(
            rng.normal(0.001, 0.03, size=(180, n_assets)),
            columns=[f"C{i}" for i in range(n_assets)],
            index=pd.date_range("2024-01-01", periods=180),
        )
        max_weight = 0.20
        res = run_nsga3_optimization(
            df, n_obj=2, pop_size=24, n_gen=10, max_weight=max_weight,
            min_assets=3, max_assets=n_assets, seed=42, verbose=False,
        )
        W = res["pareto_weights"]
        assert len(W) > 0
        for w in W:
            _assert_valid(w, max_weight)
        assert W.max() <= max_weight + TOL
