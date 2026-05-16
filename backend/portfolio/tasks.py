from datetime import datetime, timedelta
import requests
import redis
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def _mark_failed(job_id):
    from .models import PortfolioJob
    try:
        job = PortfolioJob.objects.get(id=job_id)
        job.status = 'failed'
        job.save()
    except Exception:
        pass



def _build_weights_dict(strategies: dict, tickers: list) -> dict:
    """Her strateji için {ticker: weight} sözlüğü döndürür."""
    import numpy as np

    result = {}
    for key, s in strategies.items():
        w = s["weights"]
        result[key] = {
            ticker: round(float(weight), 6)
            for ticker, weight in zip(tickers, w)
            if weight > 0.001  # sıfıra yakın ağırlıkları atla
        }
    return result


def _build_pareto_list(pareto_F, pareto_weights, tickers: list) -> list:
    """Pareto cephesini JSON-safe liste olarak döndürür."""
    result = []
    for i, (f_row, w_row) in enumerate(zip(pareto_F, pareto_weights)):
        result.append({
            "neg_return": round(float(f_row[0]), 6),
            "cvar":       round(float(f_row[1]), 6),
            "expected_return": round(float(-f_row[0]), 6),
        })
    return result


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def run_portfolio_optimization(self, job_id, params):
    from .models import PortfolioJob, PortfolioResult
    from core.optimizer.nsga3 import (
        fetch_top_coins,
        load_prices_from_yfinance,
        apply_sanity_filter,
        winsorize_returns,
        run_nsga3_optimization,
        select_portfolio_strategies,
    )

    try:
        job = PortfolioJob.objects.get(id=job_id)
        job.status = 'running'
        job.celery_task_id = self.request.id
        job.save()
        logger.info(f"Portfolio optimization started: job_id={job_id}")

        # ── Params ──────────────────────────────────────────────────────────
        n_coins      = params.get('n_coins', 50)
        lookback     = params.get('lookback_days', 365)
        n_gen        = params.get('n_gen', 200)
        pop_size     = params.get('pop_size', 100)
        n_obj        = params.get('n_obj', 2)
        max_weight   = params.get('max_weight', 0.10)
        min_assets   = params.get('min_assets', 5)
        max_assets   = params.get('max_assets', 20)

        end_date   = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=lookback + 30)).strftime('%Y-%m-%d')

        # ── 1. CoinGecko: coin listesi ──────────────────────────────────────
        logger.info("Step 1/5: Fetching top coins from CoinGecko...")
        coins_df = fetch_top_coins(n_target=n_coins, sort_by='market_cap')
        symbols  = coins_df['symbol'].str.upper().tolist()
        logger.info(f"  {len(symbols)} coin çekildi")

        # ── 2. yFinance: fiyat verisi ───────────────────────────────────────
        logger.info("Step 2/5: Loading price data from yFinance...")
        returns_df = load_prices_from_yfinance(
            symbols=symbols,
            start=start_date,
            end=end_date,
            lookback_days=lookback,
        )
        logger.info(f"  {len(returns_df.columns)} coin için veri yüklendi, {len(returns_df)} gün")

        # ── 3. Sanity filter ────────────────────────────────────────────────
        logger.info("Step 3/5: Applying sanity filter...")
        returns_df = apply_sanity_filter(returns_df, verbose=False)
        surviving  = returns_df.columns.tolist()
        logger.info(f"  {len(surviving)} coin sanity filter'dan geçti")

        # ── 4. Winsorization ────────────────────────────────────────────────
        logger.info("Step 4/5: Winsorizing returns...")
        returns_df = winsorize_returns(returns_df, verbose=False)

        # ── 5. NSGA-III optimizasyon ────────────────────────────────────────
        logger.info("Step 5/5: Running NSGA-III optimization...")
        opt_result = run_nsga3_optimization(
            returns_df=returns_df,
            n_obj=n_obj,
            pop_size=pop_size,
            n_gen=n_gen,
            max_weight=max_weight,
            min_assets=min_assets,
            max_assets=max_assets,
            verbose=False,
        )

        pareto_weights = opt_result['pareto_weights']
        pareto_F       = opt_result['pareto_F']
        tickers        = opt_result['tickers']

        # ── 6. Strateji seçimi ──────────────────────────────────────────────
        strategies = select_portfolio_strategies(
            pareto_weights=pareto_weights,
            pareto_F=pareto_F,
            returns_df=returns_df,
        )

        # ── 7. Sonuçları serialize et ve kaydet ─────────────────────────────
        import numpy as np
        strategies_dict = {}
        for key, s in strategies.items():
            strategies_dict[key] = {
                "name":          s["name"],
                "annual_return": round(float(s["annual_return"]), 6),
                "annual_vol":    round(float(s["annual_vol"]), 6),
                "sharpe":        round(float(s["sharpe"]), 4),
                "sortino":       round(float(s["sortino"]) if np.isfinite(s["sortino"]) else 0, 4),
                "calmar":        round(float(s["calmar"]) if np.isfinite(s["calmar"]) else 0, 4),
                "omega":         round(float(s["omega"]), 4),
                "cvar_daily":    round(float(s["cvar_daily"]), 6),
                "cvar_annual":   round(float(s["cvar_annual"]), 6),
                "max_drawdown":  round(float(s["max_drawdown"]), 6),
                "n_active":      int(s["n_active"]),
            }

        weights_dict = {}
        for key, s in strategies.items():
            weights_dict[key] = {
                ticker: round(float(w), 6)
                for ticker, w in zip(tickers, s["weights"])
                if w > 0.001
            }

        pareto_list = _build_pareto_list(pareto_F, pareto_weights, tickers)

        PortfolioResult.objects.create(
            job=job,
            surviving_assets=surviving,
            pareto_solutions=pareto_list,
            strategies=strategies_dict,
            weights=weights_dict,
        )

        job.status = 'completed'
        job.save()
        logger.info(f"Portfolio optimization completed: job_id={job_id}, "
                    f"surviving={len(surviving)}, pareto={len(pareto_list)}")
        return {'status': 'completed', 'job_id': str(job_id)}

    except PortfolioJob.DoesNotExist:
        logger.error(f"Job not found: job_id={job_id}")
        raise

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
        redis.exceptions.ConnectionError,
    ) as exc:
        logger.warning(f"Transient error, retrying: job_id={job_id}, error={exc}")
        _mark_failed(job_id)
        raise self.retry(exc=exc, countdown=5, max_retries=3)

    except (ValueError, KeyError, TypeError) as exc:
        logger.error(f"Non-retryable error: job_id={job_id}, error={exc}")
        _mark_failed(job_id)

    except Exception as exc:
        logger.error(f"Unexpected error: job_id={job_id}, error={exc}")
        _mark_failed(job_id)
