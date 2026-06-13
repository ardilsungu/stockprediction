"""Prompt A düzeltmeleri için testler:
  - FIX 2: inf/-inf/NaN metrikleri tutarlı şekilde None'a serileştirilir
           (inf→0 ters sinyali ve omega Infinity→jsonb çökmesi giderildi).
  - FIX 3: n_obj validasyonu (yalnızca 2 veya 3; n_obj:10 → 400).
  - FIX 4: job POST throttle (limit aşılınca 429; GET serbest).

Throttle cache'i conftest.py'deki autouse fixture ile her test arası temizlenir.
"""
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from django.urls import reverse
from rest_framework.test import APIClient

from users.models import User
from portfolio.models import PortfolioJob, PortfolioResult
from portfolio.serializers import PortfolioJobSerializer
from portfolio.tasks import run_portfolio_optimization, _finite_or_none


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email='fixes@test.com', username='fixesuser', password='Test1234!',
    )


@pytest.fixture
def auth_client(user):
    client = APIClient()
    resp = client.post(reverse('login'), {
        'email': 'fixes@test.com', 'password': 'Test1234!',
    }, format='json')
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")
    return client


# ── FIX 2: inf serileştirme ────────────────────────────────────────────────

class TestFiniteOrNone:
    def test_inf_becomes_none(self):
        assert _finite_or_none(float('inf')) is None

    def test_neg_inf_becomes_none(self):
        assert _finite_or_none(float('-inf')) is None

    def test_nan_becomes_none(self):
        assert _finite_or_none(float('nan')) is None

    def test_finite_is_rounded(self):
        assert _finite_or_none(1.234567, 4) == 1.2346

    def test_zero_is_preserved_not_none(self):
        # 0 geçerli bir değer; None'a düşmemeli (inf↔0 karışmamalı).
        assert _finite_or_none(0.0) == 0.0


_MINIMAL_COINS_DF = pd.DataFrame({
    'symbol':       ['BTC', 'ETH'],
    'coingecko_id': ['bitcoin', 'ethereum'],
    'name':         ['Bitcoin', 'Ethereum'],
    'volume_24h':   [1e9, 5e8],
    'market_cap':   [1e12, 5e11],
})


def _inf_strategies():
    """sortino/calmar/omega sonsuz olan tek-strateji sözlüğü."""
    return {
        'max_sharpe': {
            'name':          'Max Sharpe',
            'weights':       np.array([0.5, 0.5]),
            'annual_return': 0.5,
            'annual_vol':    0.3,
            'sharpe':        1.2,
            'sortino':       float('inf'),
            'calmar':        float('inf'),
            'omega':         float('inf'),
            'cvar_daily':    0.02,
            'cvar_annual':   0.4,
            'max_drawdown':  0.1,
            'n_active':      2,
            'is_duplicate':  False,
        }
    }


@pytest.mark.django_db
class TestInfSerialization:
    def _run_with_inf(self, user):
        job = PortfolioJob.objects.create(user=user, params={})
        mock_returns = pd.DataFrame(
            {'BTC': [0.01, -0.02] * 30, 'ETH': [0.02, -0.01] * 30},
            index=pd.date_range('2024-01-01', periods=60),
        )
        mock_opt = {
            'pareto_weights': np.array([[0.5, 0.5]]),
            'pareto_F':       np.array([[0.1, 0.05]]),
            'tickers':        ['BTC', 'ETH'],
            'res':            MagicMock(),
            'problem':        MagicMock(),
        }
        with patch('core.optimizer.nsga3.fetch_top_coins',
                   return_value=_MINIMAL_COINS_DF), \
             patch('core.optimizer.nsga3.load_prices_from_db_or_fetch',
                   return_value=mock_returns), \
             patch('core.optimizer.nsga3.apply_sanity_filter',
                   return_value=mock_returns), \
             patch('core.optimizer.nsga3.winsorize_returns',
                   return_value=mock_returns), \
             patch('core.optimizer.nsga3.run_nsga3_optimization',
                   return_value=mock_opt), \
             patch('core.optimizer.nsga3.select_portfolio_strategies',
                   return_value=_inf_strategies()):
            run_portfolio_optimization.apply(args=[str(job.id), {}])
        return PortfolioResult.objects.get(job=job)

    def test_inf_metrics_serialized_as_none_not_zero(self, user):
        result = self._run_with_inf(user)
        strat = result.strategies['max_sharpe']
        # inf → None (0 DEĞİL; 0 ters sinyal olurdu)
        assert strat['sortino'] is None
        assert strat['calmar'] is None
        assert strat['omega'] is None
        # finite olan korunur
        assert strat['sharpe'] == 1.2

    def test_no_infinity_reaches_jsonfield(self, user):
        result = self._run_with_inf(user)
        # Strict JSON (allow_nan=False) Infinity/NaN olsa patlardı; geçmesi
        # jsonb'ye asla Infinity gitmediğinin kanıtı.
        json.dumps(result.strategies, allow_nan=False)


# ── FIX 3: n_obj validasyonu ───────────────────────────────────────────────

@pytest.mark.django_db
class TestNObjValidation:
    @pytest.mark.parametrize('valid', [2, 3])
    def test_valid_n_obj_accepted(self, valid):
        s = PortfolioJobSerializer(data={'params': {'n_obj': valid}})
        assert s.is_valid(), s.errors

    @pytest.mark.parametrize('invalid', [1, 4, 10, 0, -1])
    def test_out_of_range_n_obj_rejected(self, invalid):
        s = PortfolioJobSerializer(data={'params': {'n_obj': invalid}})
        assert not s.is_valid()
        assert 'n_obj' in s.errors['params']

    @pytest.mark.parametrize('invalid', [2.5, '2', True])
    def test_non_int_n_obj_rejected(self, invalid):
        s = PortfolioJobSerializer(data={'params': {'n_obj': invalid}})
        assert not s.is_valid()
        assert 'n_obj' in s.errors['params']

    def test_omitted_n_obj_is_fine(self):
        s = PortfolioJobSerializer(data={'params': {'n_coins': 50}})
        assert s.is_valid(), s.errors

    def test_api_rejects_n_obj_10(self, auth_client):
        with patch('portfolio.tasks.run_portfolio_optimization.delay'):
            resp = auth_client.post(reverse('job_list'),
                                    {'params': {'n_obj': 10}}, format='json')
        assert resp.status_code == 400
        assert PortfolioJob.objects.count() == 0

    def test_api_accepts_n_obj_3(self, auth_client):
        with patch('portfolio.tasks.run_portfolio_optimization.delay'):
            resp = auth_client.post(reverse('job_list'),
                                    {'params': {'n_obj': 3}}, format='json')
        assert resp.status_code == 201


# ── FIX 4: job POST throttle ───────────────────────────────────────────────

def _post_job(auth_client):
    with patch('portfolio.tasks.run_portfolio_optimization.delay'):
        return auth_client.post(reverse('job_list'),
                                {'params': {}}, format='json')


@pytest.mark.django_db
class TestPortfolioCreateThrottle:
    def test_create_returns_429_after_limit(self, auth_client):
        # PortfolioCreateThrottle: 10/min — 11. POST throttle'a takılmalı.
        statuses = [_post_job(auth_client).status_code for _ in range(11)]
        assert statuses[:10] == [201] * 10
        assert statuses[10] == 429

    def test_get_list_is_not_throttled(self, auth_client):
        url = reverse('job_list')
        for _ in range(15):
            assert auth_client.get(url).status_code == 200
