import math
from datetime import datetime, timedelta
import requests
import redis
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def _finite_or_none(value, ndigits: int = 6):
    """Metrik serileştirmesi için TEK ve tutarlı finite politikası.

    inf/-inf/NaN → None. Eskiden sortino/calmar için inf→0 yazılıyordu (inf
    "mükemmel" iken 0 "berbat" demek → ters sinyal), omega için ise finite
    kontrolü hiç yoktu → float('inf') doğrudan JSONField'a gidip Postgres jsonb
    tarafından reddedilebiliyordu (task çökmesi). None ile hem ters sinyal hem
    çökme yolu kapanır ve nsga3.main()'deki _safe() ile aynı davranış sağlanır.
    """
    v = float(value)
    if not math.isfinite(v):
        return None
    return round(v, ndigits)

# yFinance bazı günleri (hafta sonu, tatil, ilk işlem günü öncesi) atlar.
# `lookback_days` kadar veri elde edebilmek için talep aralığını bir miktar
# geriye doğru genişletiyoruz; bu sayede sanity filter sonrası yine de
# istenen kadar günlük getiri kalıyor.
YFINANCE_FETCH_BUFFER_DAYS = 30


def _mark_failed(job_id):
    from .models import PortfolioJob
    try:
        job = PortfolioJob.objects.get(id=job_id)
        job.status = 'failed'
        job.save()
    except Exception:
        pass


def _detect_data_source() -> str:
    """CoinGecko'ya hızlı bir /ping atıp 'coingecko' veya 'fallback' döndürür.

    Bu, run_portfolio_optimization sırasında fetch_top_coins'in
    gerçekten API'yi mi yoksa hardcoded fallback listesini mi
    kullanacağını yaklaşık olarak teşhis etmek için kullanılır.
    """
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        return 'coingecko' if resp.status_code == 200 else 'fallback'
    except Exception:
        return 'fallback'



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
        load_prices_from_db_or_fetch,
        apply_sanity_filter,
        winsorize_returns,
        chronological_train_test_split,
        run_nsga3_optimization,
        select_portfolio_strategies,
        HOLDOUT_TEST_FRACTION,
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

        # Pencereyi job.created_at'e göre sabitle — datetime.today() her
        # çağrıda farklı bir saate denk geldiği için aynı params'la yapılan
        # ardışık run'larda farklı veri penceresi oluşuyordu. end_date'i bir
        # gün geriye çekiyoruz, böylece yfinance'in henüz kapanmamış olan
        # günlük çubuğu (partial bar) da pencere dışında kalır.
        end_date_dt   = job.created_at.date() - timedelta(days=1)
        start_date_dt = end_date_dt - timedelta(days=lookback + YFINANCE_FETCH_BUFFER_DAYS)
        end_date      = end_date_dt.strftime('%Y-%m-%d')
        start_date    = start_date_dt.strftime('%Y-%m-%d')

        # CoinGecko'ya erişilebiliyor mu? (fallback listesi devreye girer mi?)
        data_source = _detect_data_source()

        # Reproducibility metadata'sını erkenden yaz: optimizer patlasa bile
        # hangi pencere ve veri kaynağıyla denendiğini öğrenebilelim.
        job.params = {
            **job.params,
            'actual_start_date': start_date,
            'actual_end_date':   end_date,
            'data_source':       data_source,
        }
        job.save()

        # ── 1. CoinGecko: coin listesi ──────────────────────────────────────
        # ⚠ SURVIVORSHIP BIAS (bilinçli kabul): Evren BUGÜNÜN market-cap
        # top-N'inden seçiliyor; geçmişte var olup bugün listede olmayan coinler
        # hiç dahil edilmiyor. Bu, geçmiş getiri tahminlerini yukarı yanlı
        # yapabilir. Tarihsel (point-in-time) market-cap verimiz olmadığı için
        # kod ile çözülmüyor; metrikler bu sınır dahilinde yorumlanmalı. (Detay:
        # nsga3.py bölüm 2 başlığındaki not.)
        logger.info("Step 1/6: Fetching top coins from CoinGecko (source=%s)...", data_source)
        coins_df = fetch_top_coins(n_target=n_coins, sort_by='market_cap')
        symbols  = coins_df['symbol'].str.upper().tolist()
        logger.info(f"  {len(symbols)} coin çekildi")

        # ── 2. Fiyat verisi — DB önce, yfinance fallback ────────────────────
        logger.info("Step 2/6: Loading price data (DB-first, yfinance fallback)...")
        returns_df = load_prices_from_db_or_fetch(
            symbols=symbols,
            lookback_days=lookback,
        )
        logger.info(f"  {len(returns_df.columns)} coin için veri yüklendi, {len(returns_df)} gün")

        # ── 3. Sanity filter ────────────────────────────────────────────────
        logger.info("Step 3/6: Applying sanity filter...")
        returns_df = apply_sanity_filter(returns_df, verbose=False)
        surviving  = returns_df.columns.tolist()
        logger.info(f"  {len(surviving)} coin sanity filter'dan geçti")

        # ── 4. Winsorization ────────────────────────────────────────────────
        logger.info("Step 4/6: Winsorizing returns...")
        returns_df = winsorize_returns(returns_df, verbose=False)

        # Optimizer'a giren gerçek coin listesi (sanity filter sonrası).
        # NSGA-III ağırlık vektörleri bu sıralamaya göre.
        actual_tickers = returns_df.columns.tolist()
        job.params = {
            **job.params,
            'actual_tickers': actual_tickers,
        }
        job.save()

        # ── 5. Holdout (out-of-sample) bölme ────────────────────────────────
        # DÜRÜST METRİK: optimizasyonu TRAIN'de yap, raporlanan metrikleri
        # DOKUNULMAMIŞ TEST'te hesapla → in-sample (çifte-seçim) iyimserliği
        # giderilir. Sunulan nihai ağırlıklar TRAIN'de seçilen portföyden gelir;
        # raporlanan metrikler bu AYNI ağırlıkların test penceresindeki
        # out-of-sample performansıdır (forecast holdout yaklaşımıyla tutarlı,
        # tek optimizasyon koşusu — bkz. commit notu).
        logger.info("Step 5/6: Chronological holdout split (train/test)...")
        train_df, test_df = chronological_train_test_split(returns_df)
        logger.info(f"  holdout → train={len(train_df)} gün, test={len(test_df)} gün")

        job.params = {
            **job.params,
            'evaluation': {
                'method':        'holdout',
                'test_fraction': HOLDOUT_TEST_FRACTION,
                'n_train':       len(train_df),
                'n_test':        len(test_df),
                'test_start':    str(test_df.index[0])[:10],
                'test_end':      str(test_df.index[-1])[:10],
            },
        }
        job.save()

        # ── 6. NSGA-III optimizasyon (TRAIN penceresinde) + strateji seçimi ──
        logger.info("Step 6/6: Running NSGA-III optimization on train window...")
        opt_result = run_nsga3_optimization(
            returns_df=train_df,
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

        # Strateji seçimi TRAIN'de yapılır; metrikler eval_returns_df=test_df
        # üzerinde (out-of-sample) hesaplanır.
        strategies = select_portfolio_strategies(
            pareto_weights=pareto_weights,
            pareto_F=pareto_F,
            returns_df=train_df,
            eval_returns_df=test_df,
        )

        # ── 7. Sonuçları serialize et ve kaydet ─────────────────────────────
        # Metrikler out-of-sample (test penceresi) — bkz. adım 5/6. Tüm metrikler
        # _finite_or_none'dan geçer: inf/-inf/NaN → None (JSONField'a asla
        # Infinity/NaN gitmez, jsonb güvenli). pareto_index UI'da grafik
        # işaretçisini cephedeki doğru noktaya konumlandırmak için taşınır.
        strategies_dict = {}
        for key, s in strategies.items():
            strategies_dict[key] = {
                "name":          s["name"],
                "annual_return": _finite_or_none(s["annual_return"], 6),
                "annual_vol":    _finite_or_none(s["annual_vol"], 6),
                "sharpe":        _finite_or_none(s["sharpe"], 4),
                "sortino":       _finite_or_none(s["sortino"], 4),
                "calmar":        _finite_or_none(s["calmar"], 4),
                "omega":         _finite_or_none(s["omega"], 4),
                "cvar_daily":    _finite_or_none(s["cvar_daily"], 6),
                "cvar_annual":   _finite_or_none(s["cvar_annual"], 6),
                "max_drawdown":  _finite_or_none(s["max_drawdown"], 6),
                "n_active":      int(s["n_active"]),
                "is_duplicate":  bool(s.get("is_duplicate", False)),
                "pareto_index":  int(s["pareto_index"]),
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
        requests.exceptions.SSLError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
        redis.exceptions.ConnectionError,
    ) as exc:
        logger.warning(f"Bağlantı hatası (retry yapılacak): job_id={job_id}, error={exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)

    except (ValueError, KeyError, TypeError) as exc:
        logger.error(f"Non-retryable error: job_id={job_id}, error={exc}")
        _mark_failed(job_id)

    except Exception as exc:
        logger.error(f"Unexpected error: job_id={job_id}, error={exc}")
        _mark_failed(job_id)
