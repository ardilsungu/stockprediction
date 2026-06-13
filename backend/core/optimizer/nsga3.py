"""
Kripto Portföy Optimizasyonu — NSGA-III + CVaR  (v2)
=====================================================
v2 YENİLİKLERİ:
  ✔ CoinGecko API → En yüksek hacimli 100 coin otomatik çekilir
  ✔ Stablecoin, borsa tokeni, wrapped token filtresi
  ✔ Sortino oranı eklendi  (downside-risk ayarlı getiri)
  ✔ Sharpe + Sortino → Pareto seçim stratejileri arasında
  ✔ 100 varlıklı evren için kısıt ve görselleştirme adaptasyonu

Makale referansı: Zhao et al. (2025), Expert Systems With Applications
Yöntemler:
  - Çok amaçlı optimizasyon: NSGA-III (pymoo)
  - Risk ölçümü: CVaR + Downside Deviation
  - Kısıtlar: Cardinality, Bound, Transaction Cost, No-Short-Selling
  - Hedefler: [1] Beklenen Getiriyi Maksimize Et
               [2] CVaR'ı Minimize Et
               [3] Volatiliteyi Minimize Et   (isteğe bağlı 3. hedef)

Kurulum:
  pip install pymoo numpy pandas scipy matplotlib yfinance requests
"""

# ─────────────────────────────────────────────
# 0. IMPORT
# ─────────────────────────────────────────────
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
import random
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# pymoo imports
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination import get_termination
from pymoo.util import default_random_state

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:
    _threadpool_limits = None


class NumpyDuplicateElimination(ElementwiseDuplicateElimination):
    """Tolerans tabanlı duplicate eliminasyonu (np.allclose).

    pymoo'nun default davranışı `eliminate_duplicates=True` Python built-in
    hash() kullanır; Python 3.3+ default `PYTHONHASHSEED` rastgele olduğu için
    aynı seed=42 ile bile her process'te farklı kromozomlar elenir →
    reproducibility kırılır. NumPy `allclose` ile karşılaştırma yaparak hash
    randomization'a olan bağımlılığı tamamen ortadan kaldırıyoruz.
    """
    def is_equal(self, a, b):
        return np.allclose(a.X, b.X, atol=1e-8)


@default_random_state
def _deterministic_comp(pop, P, random_state=None, **kwargs):
    """Deterministik tournament seçimi.

    pymoo'nun built-in comp_by_cv_then_random fonksiyonu eşit CV değerli
    infeasible çiftlerde compare() çağrısına random_state GEÇMİYOR — bu
    sebeple her process'te yeni np.random.default_rng(None) yaratılıp
    farklı sonuç üretiyor. Bu wrapper aynı mantığı uygular ama tüm
    rastgele seçimlerde passed random_state kullanır.
    """
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        cv_a = float(pop[a].CV[0])
        cv_b = float(pop[b].CV[0])
        if cv_a > 0.0 or cv_b > 0.0:
            if cv_a < cv_b:
                S[i] = a
            elif cv_a > cv_b:
                S[i] = b
            else:
                S[i] = random_state.choice([a, b])
        else:
            S[i] = random_state.choice([a, b])
    return S[:, None].astype(int)


# ─────────────────────────────────────────────
# 1. FİLTRE LİSTELERİ  (stablecoin / borsa / wrapped)
# ─────────────────────────────────────────────

# USD/EUR'a peg'li stabil coinler — volatilite ~0, optimizasyona dahil edilmez
STABLECOINS = {
    # Klasik merkezi stabiller
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "USDD", "FRAX", "GUSD",
    "LUSD", "USDE", "PYUSD", "FDUSD", "USDS", "SUSDE", "USD1", "RLUSD",
    "USDY", "USTC", "USDL", "CUSD", "VUSD", "MUSD",
    # DeFi / lending protokolü stabilleri  (v2 sonuçlarında GHO %27 paya sahipti!)
    "GHO", "CRVUSD", "MKUSD", "DOLA", "MIM", "ALUSD", "SUSD", "DUSD",
    "USDF", "USDT0", "FRXUSD", "RGUSD", "MAI", "DJED", "LISUSD", "HAY",
    "USDX", "USDM", "USDA", "USR", "EUSD", "SUSDA", "USDTB", "USDQ",
    "USDC0", "USDE0", "OUSD", "USDO", "USDV", "CGUSD", "USDG", "USDB",
    "USDJ", "USDK", "USDN", "USDZ", "USD3", "USTB", "OUSG",
    # EUR peg
    "EURS", "EURC", "EURT", "AEUR", "CEUR", "EUROS", "EURE",
    # Altına peg'li → volatilite kripto'dan çok farklı
    "XAUT", "PAXG", "KAU", "KAG",
    # Sepet / SDR tipi
    "XSGD", "BIDR", "IDRT", "BRZ", "BKRW", "TRYB",
}

# Borsa tokenleri — borsa başarısına bağlı, kripto piyasasından bağımsız riskleri var
EXCHANGE_TOKENS = {
    "BNB",    # Binance
    "CRO",    # Crypto.com
    "LEO",    # Bitfinex
    "OKB",    # OKX
    "HT",     # Huobi
    "KCS",    # KuCoin
    "GT",     # Gate.io
    "MX",     # MEXC
    "BGB",    # Bitget
    "WBT",    # WhiteBIT
    "BMX",    # BitMart
    "FTT",    # FTX (iflas etti ama hâlâ listeleniyor)
    "BITB",   # Bitbank
    "NEXO",   # Nexo (CeFi kredi platformu)
}

# Başka bir varlığı 1:1 saran tokenler — BTC/ETH ile neredeyse tam korelasyon
WRAPPED_TOKENS = {
    "WBTC", "WETH", "STETH", "WSTETH", "CBETH", "WEETH", "RETH",
    "WBNB", "WMATIC", "WAVAX", "BETH", "EZETH", "METH", "OETH",
    "WBETH", "RSETH", "RSWETH", "LSETH", "ANKRETH", "SFRXETH", "FRXETH",
    "TBTC", "HBTC", "RENBTC", "SBTC", "BTCB", "WSTBTC",
    "WSOL", "JITOSOL", "MSOL", "JUPSOL", "BNSOL",
    "WTRX", "WDOGE", "WXRP", "WLTC",
}


def is_filtered_coin(symbol: str, exclude_stables=True,
                     exclude_exchange=True, exclude_wrapped=True) -> bool:
    """Coin sembolü filtre listelerinden birinde mi?"""
    s = symbol.upper()
    if exclude_stables and s in STABLECOINS:
        return True
    if exclude_exchange and s in EXCHANGE_TOKENS:
        return True
    if exclude_wrapped and s in WRAPPED_TOKENS:
        return True
    return False


# ─────────────────────────────────────────────
# 2. COINGECKO'DAN TOP-N COIN ÇEK  (piyasa değeri / hacim)
# ─────────────────────────────────────────────
#
# ⚠ SURVIVORSHIP BIAS — bilinçli kabul edilen yapısal kısıt:
# Evren, BUGÜNÜN market-cap top-N'inden seçiliyor. Yani geçmişte var olup
# bugün listede olmayan (çökmüş/delist olmuş) coinler hiç dahil edilmiyor;
# yalnızca "hayatta kalanlar" optimize ediliyor. Bu, geçmiş getiri/risk
# tahminlerini YUKARI yanlı (iyimser) yapabilir. Tarihsel (point-in-time)
# market-cap sıralaması verimiz olmadığı için bu KOD ile çözülmüyor; bilinçli
# bir sınır olarak kabul ediliyor. Raporlanan tüm metrikler bu sınır dahilinde
# yorumlanmalıdır. (Holdout/out-of-sample metrikler in-sample iyimserliği azaltır
# ama survivorship bias'ı gidermez.)

def _get_fallback_coin_list() -> list:
    """
    CoinGecko API'ye erişilemediğinde kullanılacak manuel fallback.
    Piyasa değeri sırasıyla bilinen köklü coinler (~2025 sonu snapshot).
    Stablecoin/wrapped/exchange token'ları ZATEN içermez → temiz liste.

    Market cap değerleri yaklaşıktır — sadece $1B eşiğinden geçsin diye.
    Gerçek değerler yfinance ile indirilen fiyatlardan hesaplanacak,
    burada sırf filtre için kullanılıyor.

    Bu liste, API'ye tekrar erişim sağlanana kadar sistemin çalışmaya
    devam etmesini garanti eder.
    """
    # Format: (symbol, coingecko_id, name, approximate_market_cap_usd)
    # ~2025 sonu tipik büyüklükler (değişebilir, sadece eşik geçişi için)
    raw = [
        ("BTC", "bitcoin", "Bitcoin", 1_300_000_000_000),
        ("ETH", "ethereum", "Ethereum", 400_000_000_000),
        ("XRP", "ripple", "XRP", 100_000_000_000),
        ("SOL", "solana", "Solana", 90_000_000_000),
        ("DOGE", "dogecoin", "Dogecoin", 45_000_000_000),
        ("TRX", "tron", "TRON", 25_000_000_000),
        ("ADA", "cardano", "Cardano", 22_000_000_000),
        ("LINK", "chainlink", "Chainlink", 18_000_000_000),
        ("AVAX", "avalanche-2", "Avalanche", 15_000_000_000),
        ("TON", "the-open-network", "Toncoin", 14_000_000_000),
        ("SHIB", "shiba-inu", "Shiba Inu", 13_000_000_000),
        ("DOT", "polkadot", "Polkadot", 11_000_000_000),
        ("BCH", "bitcoin-cash", "Bitcoin Cash", 10_000_000_000),
        ("NEAR", "near", "NEAR Protocol", 9_000_000_000),
        ("SUI", "sui", "Sui", 9_000_000_000),
        ("LTC", "litecoin", "Litecoin", 8_500_000_000),
        ("UNI", "uniswap", "Uniswap", 8_000_000_000),
        ("APT", "aptos", "Aptos", 7_500_000_000),
        ("ICP", "internet-computer", "Internet Computer", 7_000_000_000),
        ("ETC", "ethereum-classic", "Ethereum Classic", 6_500_000_000),
        ("XLM", "stellar", "Stellar", 6_000_000_000),
        ("XMR", "monero", "Monero", 5_500_000_000),
        ("FIL", "filecoin", "Filecoin", 5_000_000_000),
        ("ATOM", "cosmos", "Cosmos", 4_500_000_000),
        ("ARB", "arbitrum", "Arbitrum", 4_000_000_000),
        ("IMX", "immutable-x", "Immutable", 3_800_000_000),
        ("OP", "optimism", "Optimism", 3_500_000_000),
        ("HBAR", "hedera-hashgraph", "Hedera", 3_300_000_000),
        ("VET", "vechain", "VeChain", 3_000_000_000),
        ("INJ", "injective-protocol", "Injective", 2_800_000_000),
        ("AAVE", "aave", "Aave", 2_500_000_000),
        ("ALGO", "algorand", "Algorand", 2_300_000_000),
        ("MKR", "maker", "Maker", 2_200_000_000),
        ("GRT", "the-graph", "The Graph", 2_000_000_000),
        ("RENDER", "render-token", "Render", 1_900_000_000),
        ("LDO", "lido-dao", "Lido DAO", 1_800_000_000),
        ("QNT", "quant-network", "Quant", 1_700_000_000),
        ("FTM", "fantom", "Fantom", 1_600_000_000),
        ("STX", "blockstack", "Stacks", 1_600_000_000),
        ("FLOW", "flow", "Flow", 1_500_000_000),
        ("RUNE", "thorchain", "THORChain", 1_400_000_000),
        ("KAS", "kaspa", "Kaspa", 1_400_000_000),
        ("EGLD", "elrond-erd-2", "MultiversX", 1_300_000_000),
        ("SAND", "the-sandbox", "The Sandbox", 1_300_000_000),
        ("THETA", "theta-token", "Theta", 1_200_000_000),
        ("AXS", "axie-infinity", "Axie Infinity", 1_200_000_000),
        ("MANA", "decentraland", "Decentraland", 1_100_000_000),
        ("XTZ", "tezos", "Tezos", 1_100_000_000),
        ("NEO", "neo", "NEO", 1_000_000_000),
        ("CHZ", "chiliz", "Chiliz", 1_000_000_000),
        ("EOS", "eos", "EOS", 1_000_000_000),
    ]
    # API'den gelen aynı formata dönüştür
    return [
        {
            "symbol":       s,
            "id":           cg_id,
            "name":         name,
            "total_volume": 0,
            "market_cap":   mcap,
        }
        for (s, cg_id, name, mcap) in raw
    ]


def fetch_top_coins(
    n_target: int = 100,
    sort_by: str = "market_cap",
    min_market_cap_usd: float = 1e9,
    exclude_stables: bool = True,
    exclude_exchange: bool = True,
    exclude_wrapped: bool = True,
) -> pd.DataFrame:
    """
    CoinGecko /coins/markets endpoint'inden top-N coin'i çeker ve
    stablecoin/borsa/wrapped tokenleri + minimum piyasa değeri altındakileri
    süzer.

    Parameters
    ----------
    n_target : int
        API'den çekilecek coin sayısı (filtre öncesi). Filtre sonrası
        bu sayıdan az coin kalır.
    sort_by : {"market_cap", "volume"}
        Sıralama kriteri.
    min_market_cap_usd : float
        Bu eşiğin altındaki coinler elenir. $1B varsayılan — likidite ve
        veri kalitesi için minimum sınır. Düşük market-cap coinler:
          - İlk listelendikleri günlerde fiyat keşif dalgalanması yaşar
          - Yahoo Finance'ta veri kalitesi düşük olur
          - Kovaryans tahminini bozar

    Returns
    -------
    DataFrame  [symbol, coingecko_id, name, volume_24h, market_cap]
    """
    import requests

    order_map = {
        "market_cap": "market_cap_desc",
        "volume":     "volume_desc",
    }
    if sort_by not in order_map:
        raise ValueError(f"sort_by must be one of {list(order_map)}, got '{sort_by}'")

    url = "https://api.coingecko.com/api/v3/coins/markets"
    all_coins = []
    had_errors = False
    per_page = min(250, n_target)
    n_pages = int(np.ceil(n_target / per_page))

    sort_label = "piyasa değeri" if sort_by == "market_cap" else "hacim"
    print(f"\n[CoinGecko] Top {n_target} coin {sort_label}ne göre çekiliyor...")
    for page in range(1, n_pages + 1):
        params = {
            "vs_currency":  "usd",
            "order":        order_map[sort_by],
            "per_page":     per_page,
            "page":         page,
            "sparkline":    False,
        }
        # Retry mantığı — üstel bekleme (exponential backoff)
        # Bağlantı kesintileri ve rate-limit kaynaklı 429'lar için
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                # User-Agent ekle — bazı CDN'ler User-Agent'sız istekleri engeller
                headers = {"User-Agent": "Mozilla/5.0 (crypto-portfolio-optimizer)"}
                r = requests.get(url, params=params, headers=headers, timeout=30)
                r.raise_for_status()
                batch = r.json()
                all_coins.extend(batch)
                success = True
                if page < n_pages:
                    time.sleep(1.5)   # rate-limit nefesi (ücretsiz plan ~30 req/dk)
                break
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"  [Sayfa {page} deneme {attempt+1}/{max_retries}] "
                          f"Ağ hatası {type(e).__name__}: {e}  →  {wait}s bekle")
                    time.sleep(wait)
                else:
                    print(f"  [Uyarı] Sayfa {page} ağ hatası, fallback'e düşülecek: {e}")
            except requests.exceptions.HTTPError as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"  [Sayfa {page} deneme {attempt+1}/{max_retries}] "
                          f"HTTP hatası {e.response.status_code}: {e}  →  {wait}s bekle")
                    time.sleep(wait)
                else:
                    print(f"  [Uyarı] Sayfa {page} HTTP hatası, fallback'e düşülecek: {e}")
            except Exception as e:
                print(f"  [Hata] Sayfa {page} beklenmeyen hata: {type(e).__name__}: {e}")

        if not success:
            had_errors = True
            continue

    # ── FALLBACK: CoinGecko'ya erişilemezse bilinen top-100'ü kullan ──
    # Kısmi hata (bazı sayfalar başarısız) durumunda da tutarsız veri yerine
    # bilinen sabit listeyi kullanmak daha güvenlidir.
    if len(all_coins) == 0 or had_errors:
        print(f"\n[CoinGecko] ⚠ API'ye erişilemedi. FALLBACK listesi kullanılıyor.")
        print(f"  (İnternet/rate-limit problemi olabilir — daha sonra tekrar deneyin)")
        all_coins = _get_fallback_coin_list()

    # İstenen sayıya kırp (API bazen fazla döndürebilir)
    all_coins = all_coins[:n_target]

    # Filtrele
    filtered = []
    excluded = {"stable": [], "exchange": [], "wrapped": [], "small_cap": []}
    for c in all_coins:
        if c.get("symbol") is None:
            continue
        sym = c["symbol"].upper()
        mcap = c.get("market_cap") or 0

        if exclude_stables and sym in STABLECOINS:
            excluded["stable"].append(sym); continue
        if exclude_exchange and sym in EXCHANGE_TOKENS:
            excluded["exchange"].append(sym); continue
        if exclude_wrapped and sym in WRAPPED_TOKENS:
            excluded["wrapped"].append(sym); continue
        if mcap < min_market_cap_usd:
            excluded["small_cap"].append(f"{sym}({mcap/1e6:.0f}M)"); continue

        filtered.append({
            "symbol":        sym,
            "coingecko_id":  c["id"],
            "name":          c["name"],
            "volume_24h":    c.get("total_volume", 0),
            "market_cap":    mcap,
        })

    df = pd.DataFrame(filtered)

    print(f"[CoinGecko] Çekilen: {len(all_coins)} coin, Filtre sonrası: {len(df)} coin")
    print(f"  Elenen stablecoin  : {len(excluded['stable'])}  → "
          f"{', '.join(excluded['stable'][:8])}{'...' if len(excluded['stable'])>8 else ''}")
    print(f"  Elenen borsa tokeni: {len(excluded['exchange'])} → "
          f"{', '.join(excluded['exchange'][:8])}")
    print(f"  Elenen wrapped     : {len(excluded['wrapped'])}  → "
          f"{', '.join(excluded['wrapped'][:8])}{'...' if len(excluded['wrapped'])>8 else ''}")
    print(f"  Elenen düşük cap   : {len(excluded['small_cap'])} "
          f"(< ${min_market_cap_usd/1e6:.0f}M)  → "
          f"{', '.join(excluded['small_cap'][:8])}{'...' if len(excluded['small_cap'])>8 else ''}")
    return df


# ─────────────────────────────────────────────
# 3. FİYAT VERİSİ: DB ÖNCE, YFINANCE FALLBACK
# ─────────────────────────────────────────────

def load_prices_from_db_or_fetch(
    symbols: list,
    lookback_days: int = 500,
    min_lookback_days: int = 250,
) -> pd.DataFrame:
    """
    Önce DB'yi kontrol eder. Tüm sembollerin beklenen kaydın %80'i DB'de mevcutsa
    DB'den okur ve returns hesaplar. Herhangi bir sembol eksikse tüm veri seti için
    yfinance'e düşer — bu sayede tarih hizalaması tutarlı kalır.

    Parameters
    ----------
    symbols        : uppercase sembol listesi  (ör. ['BTC', 'ETH'])
    lookback_days  : arka planda bakılacak gün sayısı
    min_lookback_days : hizalama sırasında minimum pencere
    """
    import datetime
    from assets.models import Asset, Price

    end_date   = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=lookback_days + 30)
    # Hafta sonları çıkarılınca beklenen işlem günü yaklaşımı
    expected_rows = int(lookback_days * 5 / 7 * 0.8)

    db_missing = []
    for symbol in symbols:
        try:
            asset = Asset.objects.get(symbol=symbol)
            count = Price.objects.filter(
                asset=asset,
                date__gte=start_date,
                date__lte=end_date,
            ).count()
            if count < expected_rows:
                print(f"[DB] {symbol}: {count}/{expected_rows} kayıt — yetersiz.")
                db_missing.append(symbol)
        except Asset.DoesNotExist:
            print(f"[DB] {symbol}: Asset kaydı yok.")
            db_missing.append(symbol)

    # Tüm semboller DB'de yeterliyse → DB'den oku
    if not db_missing:
        print(f"[DB] Tüm {len(symbols)} sembol DB'den okunuyor "
              f"({start_date} → {end_date})...")
        rows = list(
            Price.objects
            .filter(
                asset__symbol__in=symbols,
                date__gte=start_date,
                date__lte=end_date,
            )
            .values('asset__symbol', 'date', 'close')
            .order_by('date', 'asset__symbol')
        )
        df_raw  = pd.DataFrame(rows)
        prices  = df_raw.pivot(index='date', columns='asset__symbol', values='close')
        prices  = prices[sorted(prices.columns)]
        prices.index = pd.to_datetime(prices.index)
        prices  = prices.astype(float).sort_index().ffill(limit=2).dropna(how='all')
        returns_df = prices.pct_change().dropna()

        if len(returns_df) >= 50:
            print(f"[DB] ✓ {len(returns_df.columns)} coin, {len(returns_df)} gözlem "
                  f"(pencere {lookback_days}g)")
            return returns_df

        print("[DB] DB verisi yetersiz gözlem (<50), yfinance'e düşülüyor...")
    else:
        print(f"[DB] {len(db_missing)} sembol eksik, yfinance kullanılıyor...")

    # yfinance fallback
    end_str   = end_date.strftime('%Y-%m-%d')
    start_str = start_date.strftime('%Y-%m-%d')
    print(f"[yfinance] {len(symbols)} sembol için {start_str} → {end_str} çekiliyor...")
    return load_prices_from_yfinance(
        symbols,
        start=start_str,
        end=end_str,
        lookback_days=lookback_days,
        min_lookback_days=min_lookback_days,
    )


def load_prices_from_yfinance(
    symbols: list,
    start: str,
    end: str,
    lookback_days: int = 500,
    min_lookback_days: int = 250,
) -> pd.DataFrame:
    """
    Verilen sembol listesini yfinance'tan indirir (SYMBOL-USD formatında)
    ve akıllı hizalama yapar:

    1. Tüm ticker'ları [start, end] aralığında indirir.
    2. SON `lookback_days` gün penceresinde tam veriye sahip coinleri tutar.
    3. Yetersiz coin kalırsa pencereyi `min_lookback_days` adıma kadar
       otomatik daraltır.

    Bu yaklaşım, yeni listelenmiş coinler (HYPE, BERA, MORPHO gibi) nedeniyle
    ortak-kesişim setinin boşalması sorununu çözer.

    Returns
    -------
    returns_df : DataFrame  — sütunlar: mevcut coinler, index: tarih
    """
    import yfinance as yf

    tickers_yf = [f"{s}-USD" for s in symbols]
    print(f"\n[yfinance] {len(tickers_yf)} ticker için "
          f"{start} → {end} fiyat verisi indiriliyor...")

    raw = yf.download(
        tickers_yf, start=start, end=end, progress=False,
        auto_adjust=True, threads=True,
    )

    # yfinance "Close" kolonu: tek ticker → Series, çoklu → DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers_yf[:1]

    # SYMBOL-USD → SYMBOL
    prices.columns = [c.replace("-USD", "") for c in prices.columns]

    # Tamamen boş sütunları hemen at
    prices = prices.dropna(axis=1, how="all")
    prices = prices.sort_index()

    if prices.empty:
        raise RuntimeError(
            "[yfinance] Hiçbir ticker için veri indirilemedi — bağlantıyı "
            "ve sembollerin yfinance'ta geçerli olduğunu kontrol edin."
        )

    # Diagnostik: her coinin veri uzunluğu
    obs_per_coin = prices.notna().sum().sort_values(ascending=False)
    print(f"[yfinance] Ham veri: {len(prices)} gün × {len(prices.columns)} coin")
    print(f"  Gözlem sayısı       → en uzun {obs_per_coin.iloc[0]}, "
          f"medyan {int(obs_per_coin.median())}, en kısa {obs_per_coin.iloc[-1]}")

    # ── AKILLI HİZALAMA (trailing-window) ─────────────────────────────
    # Son N gün penceresinde tam veriye sahip coinleri tut.
    # Yetersiz coin kalırsa pencereyi daralt.
    tried = []
    aligned = None
    for window_size in range(lookback_days, min_lookback_days - 1, -50):
        if len(prices) < window_size:
            tried.append((window_size, None, "veri kısa"))
            continue
        window = prices.tail(window_size)
        complete_cols = window.columns[window.notna().all()]
        tried.append((window_size, len(complete_cols), "ok"))
        # En az 15 coin kalsın — diversifikasyon için alt sınır
        if len(complete_cols) >= 15:
            aligned = window[complete_cols].copy()
            chosen_window = window_size
            break

    if aligned is None:
        # Son çare: en uzun pencereyi al
        best = max(tried, key=lambda t: t[1] or 0)
        chosen_window = best[0]
        window = prices.tail(chosen_window)
        complete_cols = window.columns[window.notna().all()]
        aligned = window[complete_cols].copy()

    print(f"[yfinance] Hizalama denemeleri:")
    for w, n, s in tried:
        mark = "✓" if n is not None and n >= 15 else " "
        print(f"    {mark} pencere={w:>3}g  →  {n if n is not None else '—':>3} tam-veri coin  [{s}]")

    # Küçük gap'ler için forward-fill (tatil günü vs.)
    aligned = aligned.ffill(limit=2).dropna(how="any")

    returns_df = aligned.pct_change().dropna()

    if returns_df.empty or len(returns_df) < 50:
        raise RuntimeError(
            f"[yfinance] Hizalama sonrası yetersiz veri "
            f"({len(returns_df)} gözlem). Tarih aralığını genişletin "
            f"(start='2022-01-01' dene) veya `min_lookback_days` değerini düşürün."
        )

    print(f"\n[yfinance] ✓ Kullanılabilir: {len(returns_df.columns)} coin, "
          f"{len(returns_df)} gözlem  (pencere {chosen_window} gün)")
    print(f"  Dönem: {returns_df.index.min().date()} → "
          f"{returns_df.index.max().date()}")

    # Düşenleri raporla
    missing = set(symbols) - set(returns_df.columns)
    if missing:
        missing_sorted = sorted(missing)
        print(f"[yfinance] Veri dışı kalan ({len(missing)}): "
              f"{', '.join(missing_sorted[:20])}"
              f"{'...' if len(missing_sorted) > 20 else ''}")

    return returns_df


# ─────────────────────────────────────────────
# 3.5. VERİ TEMİZLEME  (sanity filtresi + winsorization)
# ─────────────────────────────────────────────

def apply_sanity_filter(
    returns_df: pd.DataFrame,
    min_annualized_vol: float = 0.20,
    max_mean_daily_return: float = 0.03,
    max_single_day_abs_return: float = 3.0,
    periods_per_year: int = 365,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Veri-bazlı sanity filter. Sembol listesini aşan problemli coinleri
    istatistiksel olarak tespit eder ve atar.

    Üç ayrı teşhis yapar:

    1) Gizli stablecoin tespiti  (yıllık volatilite < min_annualized_vol)
       Sembolü USD/EUR içermeyen ama fiyatı USD'ye peg olan coinler vardır
       (GHO, USDF, CRVUSD gibi). Kripto varlıkların tipik yıllık volatilitesi
       %50-150 arasındadır — %20'nin altı neredeyse kesin bir stablecoin.

    2) Anormal ortalama getiri  (|mean daily| > max_mean_daily_return)
       Günlük ortalama getiri %3 = yıllık ~2900%. Bu gerçekçi değil;
       genelde tek bir outlier gün (listing günü, airdrop, fiyat glitch)
       ortalamayı bozmuştur. Kovaryans matrisini güvenilmez yapar.

    3) Veri glitch'i tespiti  (herhangi bir gün |return| > max_single_day_abs_return)
       Tek bir günde %300+ hareket neredeyse her zaman Yahoo Finance'taki
       split-adjustment hatası veya listing-day fiyat keşif anıdır.
       O coin'in tüm istatistiklerini bozar.

    Parameters
    ----------
    min_annualized_vol : Stablecoin eşiği (default: 0.20 = %20)
    max_mean_daily_return : Günlük ortalama üst sınır (default: 0.03 = %3)
    max_single_day_abs_return : Tek gün maksimum mutlak getiri (default: 3.0 = %300)

    Returns
    -------
    DataFrame : temizlenmiş returns_df
    """
    # yfinance bazen ffill sonrası bile NaN bırakır (ör. listing günü öncesi
    # yok-veri); NaN sütunu kovaryans matrisini ve risk metriklerini bozar →
    # sanity filter eşiklerinden önce baştan elenmeli.
    nan_cols = returns_df.columns[returns_df.isna().any()].tolist()
    if nan_cols:
        if verbose:
            head = ", ".join(nan_cols[:10])
            tail = "..." if len(nan_cols) > 10 else ""
            print(f"[Sanity Filter] NaN içeren {len(nan_cols)} coin önden elendi: {head}{tail}")
        returns_df = returns_df.drop(columns=nan_cols)

    daily_means = returns_df.mean()
    daily_vols  = returns_df.std()
    annual_vols = daily_vols * np.sqrt(periods_per_year)
    max_abs_rtn = returns_df.abs().max()

    # Teşhisler
    hidden_stables = annual_vols[annual_vols < min_annualized_vol].index.tolist()
    abnormal_mean  = daily_means[daily_means.abs() > max_mean_daily_return].index.tolist()
    data_glitches  = max_abs_rtn[max_abs_rtn > max_single_day_abs_return].index.tolist()

    removed = set(hidden_stables) | set(abnormal_mean) | set(data_glitches)
    survivors = [c for c in returns_df.columns if c not in removed]

    if verbose:
        print(f"\n[Sanity Filter] {len(returns_df.columns)} coin incelendi")
        if hidden_stables:
            print(f"  Gizli stablecoin ({len(hidden_stables)}, yıllık vol < "
                  f"{min_annualized_vol:.0%}):")
            for c in hidden_stables[:10]:
                print(f"    {c:8s}  yıl.vol = {annual_vols[c]:.1%}")
            if len(hidden_stables) > 10:
                print(f"    ... ve {len(hidden_stables)-10} tane daha")
        if abnormal_mean:
            print(f"  Anormal ortalama getiri ({len(abnormal_mean)}, günlük > "
                  f"{max_mean_daily_return:.0%}):")
            for c in abnormal_mean[:10]:
                print(f"    {c:8s}  günlük ort = {daily_means[c]:.2%}  "
                      f"(yıllık ~{daily_means[c]*periods_per_year:.0%})")
        if data_glitches:
            print(f"  Veri glitch'i ({len(data_glitches)}, tek gün > "
                  f"{max_single_day_abs_return:.0%}):")
            for c in data_glitches[:10]:
                print(f"    {c:8s}  max |return| = {max_abs_rtn[c]:.0%}")

        print(f"  Toplam elenen: {len(removed)}  →  {len(survivors)} coin kaldı")

    if len(survivors) < 10:
        raise RuntimeError(
            f"Sanity filter sonrası sadece {len(survivors)} coin kaldı — "
            f"eşikleri gevşetin veya veri kaynağını değiştirin."
        )

    return returns_df[survivors].copy()


def winsorize_returns(
    returns_df: pd.DataFrame,
    lower_pct: float = 0.005,
    upper_pct: float = 0.995,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Her coin için günlük getirileri [lower_pct, upper_pct] yüzdelikleri
    aralığına kırpar (winsorization).

    Neden? Sanity filter'dan geçen coinlerde bile 1-2 uç gün olabilir
    (borsadaki kısa süreli flash crash, yetersiz likidite). Bu günler:
        - Ortalama getiriyi çarpıtır
        - Kovaryans matrisini bozar  → NSGA-III yanlış yöne gider
        - Sharpe/Sortino'yu yapay şişirir

    Default %0.5 - %99.5 aralığı:  750 günde en yüksek/düşük ~3-4 gün kırpılır.
    Bu ana dağılımı korur, sadece "açık outlier"ları yumuşatır.

    Returns
    -------
    DataFrame : winsorize edilmiş getiriler (aynı şekil)
    """
    result  = returns_df.copy()
    clipped_per_coin = {}

    for col in result.columns:
        s = result[col]
        lo, hi = s.quantile(lower_pct), s.quantile(upper_pct)
        n_clipped = ((s < lo) | (s > hi)).sum()
        result[col] = s.clip(lower=lo, upper=hi)
        clipped_per_coin[col] = n_clipped

    if verbose:
        total = sum(clipped_per_coin.values())
        pct   = total / (len(result) * len(result.columns)) * 100
        print(f"\n[Winsorization] %{lower_pct*100:.1f} - %{upper_pct*100:.1f} "
              f"aralığına kırpıldı")
        print(f"  Toplam {total} gözlem etkilendi ({pct:.2f}% tüm veri içinde)")

        # En çok kırpılan 5 coin (veri kalitesi hakkında ipucu)
        most_clipped = sorted(clipped_per_coin.items(), key=lambda x: -x[1])[:5]
        print(f"  En çok kırpılan 5 coin:")
        for coin, n in most_clipped:
            print(f"    {coin:8s}  {n} gün")

    return result


# ─────────────────────────────────────────────
# 4. RİSK METRİKLERİ  (Sharpe + Sortino dahil)
# ─────────────────────────────────────────────

def compute_portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return float(weights @ mean_returns)


def compute_portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov_matrix @ weights))


def compute_cvar(weights: np.ndarray, returns_matrix: np.ndarray,
                 alpha: float = 0.05) -> float:
    """
    Tarihsel simülasyonla CVaR.
    CVaRα = E(ξ | ξ ≤ VaRα)   — Zhao et al. (2025) formülü
    Pozitif döndürür → minimize edilecek kayıp büyüklüğü.
    """
    if np.isnan(returns_matrix).any():
        raise ValueError("compute_cvar: returns_matrix NaN içeriyor")
    port_returns = returns_matrix @ weights
    if port_returns.size == 0:
        raise ValueError(
            "compute_cvar: returns_matrix boş. Muhtemelen veri hizalama "
            "başarısız oldu — yfinance indirimini kontrol edin."
        )
    var_threshold = np.percentile(port_returns, alpha * 100)
    tail_losses   = port_returns[port_returns <= var_threshold]
    if tail_losses.size == 0:
        return 0.0
    return -float(np.mean(tail_losses))


def compute_sharpe(weights: np.ndarray, mean_returns: np.ndarray,
                   cov_matrix: np.ndarray, risk_free_rate: float = 0.0,
                   periods_per_year: int = 365) -> float:
    """
    Sharpe oranı (yıllıklaştırılmış).
    Kripto 7/24 işlem gördüğü için yıllıklaştırma faktörü = 365.
    """
    if np.isnan(mean_returns).any() or np.isnan(cov_matrix).any():
        raise ValueError("compute_sharpe: mean_returns veya cov_matrix NaN içeriyor")
    ret = compute_portfolio_return(weights, mean_returns) * periods_per_year
    vol = compute_portfolio_volatility(weights, cov_matrix) * np.sqrt(periods_per_year)
    return (ret - risk_free_rate) / (vol + 1e-9)


def compute_sortino(weights: np.ndarray, returns_matrix: np.ndarray,
                    risk_free_rate: float = 0.0, target: float = 0.0,
                    periods_per_year: int = 365) -> float:
    """
    Sortino oranı (yıllıklaştırılmış).
    Sharpe'tan farkı: paydada SADECE negatif (downside) getirilerin
    standart sapması. Yukarı yönlü volatilite cezalandırılmaz → yatırımcı
    için daha anlamlı bir risk-ayarlı getiri ölçüsüdür.

        Sortino = (E[Rp] - Rf) / DownsideDev

    target : minimum kabul edilebilir getiri (MAR), varsayılan 0
    """
    if np.isnan(returns_matrix).any():
        raise ValueError("compute_sortino: returns_matrix NaN içeriyor")
    port_returns = returns_matrix @ weights
    ann_return   = port_returns.mean() * periods_per_year

    downside = port_returns[port_returns < target] - target
    if len(downside) == 0:
        return float("inf")   # hiç kayıp yoksa mükemmel
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
    return (ann_return - risk_free_rate) / (downside_dev + 1e-9)


def compute_max_drawdown(weights: np.ndarray, returns_matrix: np.ndarray) -> float:
    port_returns = returns_matrix @ weights
    cumulative   = (1 + port_returns).cumprod()
    running_max  = np.maximum.accumulate(cumulative)
    drawdowns    = (cumulative - running_max) / running_max
    return float(-np.min(drawdowns))


def compute_calmar(weights: np.ndarray, mean_returns: np.ndarray,
                   returns_matrix: np.ndarray,
                   periods_per_year: int = 365) -> float:
    """
    Calmar Oranı (yıllıklaştırılmış).
        Calmar = Yıllık Getiri / Max Drawdown

    Sortino ortalama downside'a bakarken, Calmar "tarihte gördüğün EN KÖTÜ
    düşüş" üzerinden değerlendirir. Kripto gibi çok-volatil varlıklarda
    worst-case odaklı bir risk-ayarlı getiri ölçüsüdür. Genelde:
        < 1.0  → zayıf
        1-3    → iyi
        > 3    → mükemmel  (ama kısa pencerede aldatıcı olabilir)
    """
    ann_return = compute_portfolio_return(weights, mean_returns) * periods_per_year
    max_dd     = compute_max_drawdown(weights, returns_matrix)
    if max_dd < 1e-9:
        return float("inf")   # hiç drawdown yok → mükemmel
    return ann_return / max_dd


def compute_omega(weights: np.ndarray, returns_matrix: np.ndarray,
                  threshold: float = 0.0) -> float:
    """
    Omega Oranı.
        Ω(τ) = E[max(R - τ, 0)] / E[max(τ - R, 0)]
             = (eşik üstü toplam kazanç) / (eşik altı toplam kayıp)

    Sharpe sadece ortalama ve standart sapmayı kullanır — dağılımın
    ilk iki momenti. Omega dağılımın TAMAMINI kullanır: skewness ve
    kurtosis (fat-tail) etkileri dahil. Kripto için çok daha doğru.
        Ω > 1  → eşik üstü kazançlar, eşik altı kayıplardan büyük
        Ω = 1  → eşit
        Ω < 1  → kayıplar baskın

    threshold : minimum kabul edilebilir getiri (günlük, default 0).
    """
    port_returns = returns_matrix @ weights
    excess       = port_returns - threshold
    gains        = excess[excess > 0].sum()
    losses       = -excess[excess < 0].sum()
    if losses < 1e-9:
        return float("inf")
    return float(gains / losses)


def compute_var(weights: np.ndarray, returns_matrix: np.ndarray,
                alpha: float = 0.05) -> float:
    """
    Historical VaR (Value at Risk).
        VaRα = -percentile(returns, α * 100)

    CVaR ile farkı:
        VaR  = "%α olasılıkla en kötü kayıp EŞİĞİ"       (sınır)
        CVaR = "o eşiğin ALTINDAKİ ortalama kayıp"        (kuyruğun derinliği)

    Örnek: VaR %5 = %3 ise, "günlerin %5'inde en az %3 kaybedebilirim"
          CVaR %5 = %7 ise, "o kötü günlerde tipik kaybım %7"

    VaR standart bir regülasyon/raporlama metriği, CVaR daha ihtiyatlı.
    İkisi birlikte sunulduğunda kuyruk-riski tam resim çizer.
    Pozitif döndürür (kayıp büyüklüğü).
    """
    port_returns = returns_matrix @ weights
    return -float(np.percentile(port_returns, alpha * 100))


# ─────────────────────────────────────────────
# 4.5. AĞIRLIK REPAIR  (kısıt projeksiyonu)
# ─────────────────────────────────────────────

def repair_weight_vector(x: np.ndarray, max_weight: float,
                         tol: float = 1e-9) -> np.ndarray:
    """Ham genomu geçerli bir long-only ağırlık vektörüne projekte eder.

    Garantiler (verification target):
        toplam = 1,  her wᵢ ≤ max_weight,  her wᵢ ≥ 0.

    Neden basit X/sum(X) yetmez: normalizasyon toplam=1 sağlar AMA wᵢ ≤
    max_weight'i sağlamaz; sum(X) < 1 olduğunda tek bir gen normalize sonrası
    sınırı kat kat aşabilir (eski hatada tek-varlık ~%97). Burada capped/
    iteratif "water-filling" projeksiyonu kullanılır: sınırı aşan ağırlıklar
    max_weight'e kırpılır, taşan kütle sınıra ulaşmamış AKTİF genlere yeniden
    dağıtılır; aşım kalmayana dek tekrarlanır.

    Seyreklik (cardinality) korunur: yalnızca pozitif genler (aktif set) kütle
    paylaşır; sıfır/negatif genler sıfır kalır. Taşan kütle, aktif uncapped
    genlere ORANTILI dağıtılır (eşit değil) — böylece "dust" (çok küçük) genler
    küçük kalır, kütle zaten anlamlı genlerde yoğunlaşır ve aktif/sıfır yapısı
    bozulmaz. min/max_assets kısıtları repair'in işi DEĞİL — onlar problem G
    vektöründe optimizatöre bırakılır. Repair sadece toplam=1'i mümkün kılacak
    kadar (n_min = ⌈1/max_weight⌉) genin aktif olmasını garanti eder.

    Determinizm: hiçbir rastgelelik yok; aktif set genişletilmesi gerekirse
    genler büyük-x → küçük-index sırasıyla deterministik seçilir. Zaten geçerli
    bir vektörde idempotenttir (aynısını döndürür).

    Fizibilite ön-koşulu: max_weight * n_assets ≥ 1 olmalı; aksi halde toplam=1
    capped sınırlarla imkânsızdır → anlamlı ValueError.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    n = x.size
    if max_weight * n < 1.0 - tol:
        raise ValueError(
            f"max_weight={max_weight:.4g} ile {n} varlık için toplam=1 imkânsız "
            f"(max_weight * n_assets = {max_weight * n:.4g} < 1). "
            f"max_weight'i artırın veya varlık sayısını çoğaltın."
        )

    # toplam=1'e cap'lerle ulaşabilmek için gereken minimum aktif gen sayısı
    n_min = int(np.ceil(1.0 / max_weight - tol))
    n_min = max(1, min(n_min, n))

    active = x > 0.0
    if active.sum() < n_min:
        # Aktif set yetersiz → toplam=1 fizibilitesi için deterministik genişlet:
        # büyük-x öncelikli, eşitlikte küçük-index.
        order = np.lexsort((np.arange(n), -x))
        active = np.zeros(n, dtype=bool)
        active[order[:n_min]] = True

    w = np.zeros(n)
    xa = x[active]
    xa_sum = xa.sum()
    if xa_sum <= tol:
        w[active] = 1.0 / active.sum()
    else:
        w[active] = xa / xa_sum

    # İteratif water-filling cap. Her tur sınırı aşan genler max_weight'e
    # sabitlenir (bir daha azalmaz) → capped küme tekdüze büyür, en çok n turda
    # yakınsar; range(n+1) güvenli üst sınır. Taşan kütle orantılı dağıtılır;
    # under-toplam ihmal edilebilirse (tüm uncapped genler dust) eşit dağıtıma
    # düşülür ki kütle bir yere gitsin.
    for _ in range(n + 1):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = active & (w < max_weight)
        n_under = int(under.sum())
        if n_under == 0:
            break
        under_sum = float(w[under].sum())
        if under_sum > tol:
            w[under] += excess * (w[under] / under_sum)
        else:
            w[under] += excess / n_under

    return w


def repair_weights(X: np.ndarray, max_weight: float) -> np.ndarray:
    """`repair_weight_vector`'ı tekil (1B) veya popülasyon (2B) girdiye uygular."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return repair_weight_vector(X, max_weight)
    return np.vstack([repair_weight_vector(row, max_weight) for row in X])


class WeightRepair(Repair):
    """pymoo Repair operatörü: her bireyin genomunu geçerli ağırlık vektörüne
    (toplam=1, wᵢ≤max_weight, wᵢ≥0) projekte eder.

    Mating sırasında offspring genomlarına uygulanır; arama feasible bölgede
    kalır. Başlangıç popülasyonu pymoo tarafından repair edilmediğinden,
    `_evaluate` ayrıca aynı projeksiyonu uygular (idempotent) → metrikler ve
    kısıtlar daima sunulacak ağırlıklarla hesaplanır.
    """

    def __init__(self, max_weight: float):
        super().__init__()
        self.max_weight = max_weight

    def _do(self, problem, X, **kwargs):
        return repair_weights(X, self.max_weight)


# ─────────────────────────────────────────────
# 5. PYMOO PROBLEMİ
# ─────────────────────────────────────────────

class CryptoPortfolioProblem(Problem):
    """
    Çok Amaçlı Kripto Portföy Optimizasyon Problemi.

    Hedefler:
        f1 = -Beklenen Getiri        (min → aslında maksimize)
        f2 =  CVaR                   (min)
        f3 =  Volatilite  [opsiyonel](min)
    Kısıtlar (2 eşitsizlik):
        g1 = min_assets - active     (≥ min_assets aktif varlık)
        g2 = active - max_assets     (≤ max_assets aktif varlık)

    Not: Ağırlık fizibilitesi (Σwᵢ = 1, wᵢ ≤ max_weight, wᵢ ≥ 0) artık bir
    KISIT değil; `WeightRepair` / `repair_weights` ile projeksiyon yoluyla
    GARANTİ edilir (bkz. _evaluate). Eski |Σwᵢ-1| kısıtı, W normalize edildiği
    için daima sağlanan vakum bir kısıttı ve kaldırıldı; geriye yalnızca
    cardinality (aktif varlık sayısı) gerçek kısıtları kaldı.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        n_obj: int = 2,
        alpha: float = 0.05,
        min_weight: float = 0.0,
        max_weight: float = 0.10,
        min_assets: int = 5,
        max_assets: int = 20,
        transaction_cost: float = 0.001,
        current_weights: np.ndarray = None,
        weight_threshold: float = 0.005,
    ):
        self.returns_matrix   = returns_df.values
        self.mean_returns     = returns_df.mean().values
        self.cov_matrix       = returns_df.cov().values
        self.tickers          = list(returns_df.columns)
        self.n_assets         = len(self.tickers)
        self.alpha            = alpha
        self.transaction_cost = transaction_cost
        self.min_assets       = min_assets
        self.max_assets       = max_assets
        self.max_weight       = max_weight
        self.n_obj            = n_obj
        self.weight_threshold = weight_threshold

        # Fizibilite ön-koşulu: cap'li ağırlıklarla toplam=1'e ulaşılabilmesi
        # için max_weight * n_assets ≥ 1 olmalı (aksi halde repair toplam=1
        # üretemez). Erken ve net hata ver.
        if max_weight * self.n_assets < 1.0 - 1e-9:
            raise ValueError(
                f"max_weight={max_weight:.4g} ile {self.n_assets} varlık için "
                f"toplam=1 imkânsız (max_weight * n_assets = "
                f"{max_weight * self.n_assets:.4g} < 1). max_weight'i artırın "
                f"veya daha fazla varlık kullanın."
            )

        if current_weights is None:
            self.current_weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.current_weights = current_weights

        super().__init__(
            n_var=self.n_assets,
            n_obj=n_obj,
            n_ieq_constr=2,
            xl=np.full(self.n_assets, min_weight),
            xu=np.full(self.n_assets, max_weight),
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        pop_size = X.shape[0]
        # Ağırlık fizibilitesini projeksiyonla GARANTİ et (toplam=1,
        # wᵢ≤max_weight, wᵢ≥0). Başlangıç popülasyonu pymoo tarafından repair
        # edilmediği için burada da uygulanır; WeightRepair ile birlikte
        # idempotenttir. Metrikler ve kısıtlar böylece daima gerçekten
        # sunulacak ağırlıklarla hesaplanır.
        W = repair_weights(X, self.max_weight)

        F = np.zeros((pop_size, self.n_obj))
        G = np.zeros((pop_size, 2))

        n_days = self.returns_matrix.shape[0]
        for i in range(pop_size):
            w = W[i]
            # TC bir kerelik rebalance maliyetidir; lookback gün sayısına
            # amortize ederek günlük getiriden düş. Eskiden tüm TC günlük
            # getiriden çıkıyordu → 5 varlıkta ~%60 yapay yıllık kayıp.
            tc_daily = self.transaction_cost * np.sum(np.abs(w - self.current_weights)) / n_days
            ret  = compute_portfolio_return(w, self.mean_returns) - tc_daily
            cvar = compute_cvar(w, self.returns_matrix, self.alpha)

            F[i, 0] = -ret
            F[i, 1] =  cvar
            if self.n_obj == 3:
                F[i, 2] = compute_portfolio_volatility(w, self.cov_matrix)

            # Yalnızca cardinality gerçek kısıt; toplam=1 ve max_weight repair
            # ile garanti olduğundan G'den çıkarıldı (vakum kısıt temizliği).
            active  = np.sum(w > self.weight_threshold)
            G[i, 0] = self.min_assets - active
            G[i, 1] = active - self.max_assets

        out["F"] = F
        out["G"] = G


# ─────────────────────────────────────────────
# 5.5. HOLDOUT (TRAIN/TEST) BÖLME — out-of-sample değerlendirme
# ─────────────────────────────────────────────

# Holdout (out-of-sample) değerlendirme parametreleri. Optimizasyon TRAIN'de
# yapılır, raporlanan metrikler DOKUNULMAMIŞ TEST'te hesaplanır → in-sample
# (çifte-seçim) iyimserliği giderilir.
HOLDOUT_TEST_FRACTION = 0.2   # son %20 kronolojik test penceresi
MIN_TRAIN_OBS = 60            # train hâlâ anlamlı optimize edilebilmeli
MIN_TEST_OBS = 30             # test ölçülebilir uzunlukta olmalı (CVaR/Sharpe)


def chronological_train_test_split(
    returns_df: pd.DataFrame,
    test_fraction: float = HOLDOUT_TEST_FRACTION,
    min_train_obs: int = MIN_TRAIN_OBS,
    min_test_obs: int = MIN_TEST_OBS,
) -> tuple:
    """Returns_df'i kronolojik (zaman sıralı) train/test'e böler.

    Bölme örtüşmesiz ve kronolojiktir: train = ilk (1 - test_fraction) oranı
    (geçmiş), test = son test_fraction oranı (en yeni). Index'in artan zaman
    sırasında olduğu varsayılır — pipeline (DB/yfinance yolları) sort_index
    uyguladığı için bu garanti.

    Out-of-sample değerlendirme mantığı: ağırlıklar TRAIN üzerinde
    optimize/seçilir; performans metrikleri DOKUNULMAMIŞ TEST üzerinde
    hesaplanır. Böylece raporlanan Sharpe/Sortino/CVaR/getiri, ağırlıkların
    seçildiği veriden ayrı bir pencerede ölçülür.

    Eşik altı veri → anlamlı ValueError (train optimize edilebilir, test
    ölçülebilir kalmalı).

    Returns
    -------
    (train_df, test_df)
    """
    n = len(returns_df)
    n_test = int(round(n * test_fraction))
    n_train = n - n_test
    if n_train < min_train_obs or n_test < min_test_obs:
        raise ValueError(
            f"Holdout (out-of-sample) değerlendirme için yetersiz veri: "
            f"{n} gözlemden train={n_train}, test={n_test} çıkıyor "
            f"(gerekli: train ≥ {min_train_obs}, test ≥ {min_test_obs}). "
            f"lookback_days'i artırın veya daha uzun geçmişi olan coinler kullanın."
        )
    train_df = returns_df.iloc[:n_train]
    test_df = returns_df.iloc[n_train:]
    return train_df, test_df


# ─────────────────────────────────────────────
# 6. OPTİMİZASYON ÇALIŞTIR
# ─────────────────────────────────────────────

def run_nsga3_optimization(
    returns_df: pd.DataFrame,
    n_obj: int = 2,
    pop_size: int = 300,
    n_gen: int = 400,
    alpha: float = 0.05,
    max_weight: float = 0.10,
    min_assets: int = 5,
    max_assets: int = 20,
    transaction_cost: float = 0.001,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    # Reproducibility: PYTHONHASHSEED process başında set edilmeli (etkili
    # olması için start_celery.sh / start_django.sh wrapper'larını kullan).
    # Aşağıdaki satır mevcut process'in hash seed'ini değiştirmez, sadece
    # child process'lere işaret eder; asıl deterministiklik garantisi
    # NumpyDuplicateElimination + numpy/random seed reset ile sağlanıyor.
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)

    print("\n" + "="*62)
    print("  NSGA-III Kripto Portföy Optimizasyonu")
    print(f"  Varlık sayısı : {len(returns_df.columns)}")
    print(f"  Hedef sayısı  : {n_obj}")
    print(f"  Popülasyon    : {pop_size}")
    print(f"  Nesil sayısı  : {n_gen}")
    print(f"  max_weight    : {max_weight:.1%}  |  cardinality: {min_assets}-{max_assets}")
    print(f"  CVaR alpha    : {alpha*100:.0f}%")
    print("="*62)

    problem = CryptoPortfolioProblem(
        returns_df=returns_df,
        n_obj=n_obj,
        alpha=alpha,
        max_weight=max_weight,
        min_assets=min_assets,
        max_assets=max_assets,
        transaction_cost=transaction_cost,
    )

    # 2 hedef için daha ince, 3 hedef için daha az ref_dir
    n_partitions = 20 if n_obj == 2 else 8
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        selection=TournamentSelection(func_comp=_deterministic_comp),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / problem.n_var, eta=20),
        # Offspring genomlarını feasible ağırlık simpleksine projekte et
        # (toplam=1, wᵢ≤max_weight, wᵢ≥0). Arama feasible bölgede kalır.
        repair=WeightRepair(max_weight),
        eliminate_duplicates=NumpyDuplicateElimination(),
    )

    if _threadpool_limits is not None:
        with _threadpool_limits(limits=1):
            res = minimize(
                problem, algorithm, get_termination("n_gen", n_gen),
                seed=seed, save_history=False, verbose=verbose,
            )
    else:
        res = minimize(
            problem, algorithm, get_termination("n_gen", n_gen),
            seed=seed, save_history=False, verbose=verbose,
        )

    X_opt = res.X
    F_opt = res.F
    if X_opt is None or len(X_opt) == 0:
        raise RuntimeError(
            "NSGA-III feasible çözüm bulamadı. "
            "Kısıtları gevşetin: max_weight artırın, "
            "min_assets azaltın veya n_gen artırın."
        )
    # Nihai ağırlıklar da aynı repair'den geçer → çıktı kısıtları (toplam=1,
    # wᵢ≤max_weight) optimizasyon sırasındakiyle birebir aynı garantiyle taşır.
    # Eski ham X/sum(X) max_weight'i aşabiliyordu (kapatılan 🔴 bulgu).
    pareto_weights = repair_weights(X_opt, max_weight)

    print(f"\n  Pareto-optimal çözüm sayısı: {len(pareto_weights)}")

    return {
        "pareto_weights": pareto_weights,
        "pareto_F":       F_opt,
        "res":            res,
        "problem":        problem,
        "tickers":        returns_df.columns.tolist(),
    }


# ─────────────────────────────────────────────
# 7. PARETO ÇÖZÜM SEÇİMİ  (Sortino dahil)
# ─────────────────────────────────────────────

def select_portfolio_strategies(
    pareto_weights: np.ndarray,
    pareto_F: np.ndarray,
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
    periods_per_year: int = 365,
    eval_returns_df: pd.DataFrame = None,
) -> dict:
    """
    Pareto cephesinden 5 strateji seç:
        1. Max Sharpe   — klasik risk-getiri dengesi
        2. Max Sortino  — downside-risk ayarlı en iyi
        3. Min CVaR     — en güvenli (tail-risk minimum)
        4. Max Return   — en agresif
        5. Balanced     — ideal noktaya en yakın (ortalama)

    Tie-break: Her seçimde önce hiç sahiplenilmemiş indeksler arasından, hepsi
    sahiplenilmişse en az kez seçilenler arasından en iyi skor seçilir. Cephe
    ≥5 noktaysa 5 benzersiz strateji çıkar; <5 noktaysa zorunlu paylaşım
    minimum tekrarla yapılır ve tekrarlı stratejiler `is_duplicate=True`
    bayrağıyla işaretlenir (UI'da uyarı için).

    Önceliklendirme: sharpe → min_cvar → max_return → max_sortino → balanced.

    SEÇİM vs METRİK penceresi:
      - SEÇİM (hangi Pareto noktası hangi strateji) DAİMA `returns_df`
        (= optimizasyon/train penceresi) üzerinde yapılır; karar anında elde
        olan veri budur.
      - METRİK TABLOSU (annual_return/Sharpe/Sortino/CVaR/...) `eval_returns_df`
        verildiyse ORADA (out-of-sample/holdout test penceresi) hesaplanır;
        verilmediyse geriye-uyumlu olarak `returns_df` (in-sample) kullanılır.
    Bu ayrım, raporlanan metriklerin ağırlıkların seçildiği veriden AYRI bir
    pencerede ölçülmesini sağlar (in-sample iyimserliğini giderir).

    Her stratejiye `pareto_index` (cephedeki konum indeksi) eklenir; UI grafik
    işaretçilerini metrik değerine bağlı mesafe eşlemesi yerine bu indeksle
    konumlandırır (out-of-sample metrik in-sample cephe konumundan ayrıştığında
    işaretçilerin kaymaması için).
    """
    mean_ret = returns_df.mean().values
    cov_mat  = returns_df.cov().values
    ret_mat  = returns_df.values

    n = len(pareto_weights)
    pick_count: dict[int, int] = {i: 0 for i in range(n)}

    def _pick(scores: np.ndarray, minimize: bool) -> tuple[int, bool]:
        """En az sahiplenilmiş indeksler arasından en iyi skoru olanı seç.

        Returns (idx, is_duplicate) — is_duplicate True ise bu indeks bu
        çağrıdan önce zaten başka bir strateji tarafından seçilmiş demektir.
        """
        arr = np.asarray(scores, dtype=float)
        min_count = min(pick_count.values())
        candidates = [i for i, c in pick_count.items() if c == min_count]
        if minimize:
            idx = min(candidates, key=lambda i: arr[i])
        else:
            idx = max(candidates, key=lambda i: arr[i])
        is_dup = pick_count[idx] > 0
        pick_count[idx] += 1
        return idx, is_dup

    # Skorları bir kez hesapla; pick sırası farklı olsa da metrikler değişmez.
    sharpes = np.array([
        compute_sharpe(pareto_weights[i], mean_ret, cov_mat,
                       periods_per_year=periods_per_year)
        for i in range(n)
    ])
    sortinos = np.array([
        compute_sortino(pareto_weights[i], ret_mat,
                        periods_per_year=periods_per_year)
        for i in range(n)
    ])
    sortinos_finite = np.where(np.isfinite(sortinos), sortinos, -np.inf)
    F_norm = (pareto_F - pareto_F.min(0)) / (np.ptp(pareto_F, axis=0) + 1e-12)
    dists  = np.linalg.norm(F_norm, axis=1)

    # Önceliklendirilmiş seçim (claim sırası).
    sharpe_idx,     sharpe_dup     = _pick(sharpes,         minimize=False)
    min_cvar_idx,   min_cvar_dup   = _pick(pareto_F[:, 1],  minimize=True)
    max_return_idx, max_return_dup = _pick(pareto_F[:, 0],  minimize=True)
    sortino_idx,    sortino_dup    = _pick(sortinos_finite, minimize=False)
    balanced_idx,   balanced_dup   = _pick(dists,           minimize=True)

    # Görüntü/insertion sırası UI ile uyumlu kalır (önceki davranış).
    strategies: dict = {
        "max_sharpe": {
            "weights":      pareto_weights[sharpe_idx],
            "name":         "Max Sharpe",
            "color":        "#1D9E75",
            "marker":       "★",
            "is_duplicate": sharpe_dup,
            "pareto_index": int(sharpe_idx),
        },
        "max_sortino": {
            "weights":      pareto_weights[sortino_idx],
            "name":         "Max Sortino",
            "color":        "#E8A33D",
            "marker":       "✦",
            "is_duplicate": sortino_dup,
            "pareto_index": int(sortino_idx),
        },
        "min_cvar": {
            "weights":      pareto_weights[min_cvar_idx],
            "name":         "Min CVaR",
            "color":        "#378ADD",
            "marker":       "◆",
            "is_duplicate": min_cvar_dup,
            "pareto_index": int(min_cvar_idx),
        },
        "max_return": {
            "weights":      pareto_weights[max_return_idx],
            "name":         "Max Return",
            "color":        "#D85A30",
            "marker":       "▲",
            "is_duplicate": max_return_dup,
            "pareto_index": int(max_return_idx),
        },
        "balanced": {
            "weights":      pareto_weights[balanced_idx],
            "name":         "Balanced",
            "color":        "#7F77DD",
            "marker":       "●",
            "is_duplicate": balanced_dup,
            "pareto_index": int(balanced_idx),
        },
    }

    # Metrik penceresi: eval_returns_df verildiyse metrikler ORADA (out-of-sample
    # holdout) hesaplanır; aksi halde returns_df (in-sample, geriye-uyumlu).
    # n_active ağırlıkların yapısal özelliği olduğu için pencereden bağımsızdır.
    metric_df = eval_returns_df if eval_returns_df is not None else returns_df
    m_mean = metric_df.mean().values
    m_cov  = metric_df.cov().values
    m_ret  = metric_df.values

    # Her strateji için metrik tablosu
    ann = periods_per_year
    for key, strat in strategies.items():
        w = strat["weights"]
        strat["annual_return"] = compute_portfolio_return(w, m_mean) * ann
        strat["annual_vol"]    = compute_portfolio_volatility(w, m_cov) * np.sqrt(ann)
        strat["sharpe"]        = compute_sharpe(w, m_mean, m_cov,
                                                periods_per_year=ann)
        strat["sortino"]       = compute_sortino(w, m_ret, periods_per_year=ann)
        strat["calmar"]        = compute_calmar(w, m_mean, m_ret,
                                                periods_per_year=ann)
        strat["omega"]         = compute_omega(w, m_ret, threshold=0.0)
        strat["var_daily"]     = compute_var(w, m_ret, alpha)
        strat["cvar_daily"]    = compute_cvar(w, m_ret, alpha)
        strat["cvar_annual"]   = strat["cvar_daily"] * np.sqrt(ann)
        strat["max_drawdown"]  = compute_max_drawdown(w, m_ret)
        strat["n_active"]      = int(np.sum(w > 0.005))

    return strategies


# ─────────────────────────────────────────────
# 8. SONUÇ YAZDIRMA  (Sortino dahil)
# ─────────────────────────────────────────────

def print_results(strategies: dict, tickers: list,
                  top_k_weights: int = 15) -> None:
    """Terminale tablo halinde yazdır (100 coin için özet modu)."""
    print("\n" + "="*90)
    print("  PARETO-OPTİMAL PORTFÖY STRATEJİLERİ")
    print("="*90)

    header = f"{'Metrik':<22}"
    for strat in strategies.values():
        header += f"  {strat['name']:>13}"
    print(header)
    print("-"*90)

    metrics = [
        ("Yıllık Getiri",     "annual_return",  "{:.1%}"),
        ("Yıllık Volatilite", "annual_vol",      "{:.1%}"),
        ("Sharpe Oranı",      "sharpe",          "{:.3f}"),
        ("Sortino Oranı",     "sortino",         "{:.3f}"),
        ("Calmar Oranı",      "calmar",          "{:.3f}"),
        ("Omega Oranı",       "omega",           "{:.3f}"),
        ("VaR %5 (günlük)",   "var_daily",       "{:.3%}"),
        ("CVaR %5 (günlük)",  "cvar_daily",      "{:.3%}"),
        ("CVaR %5 (yıllık)",  "cvar_annual",     "{:.1%}"),
        ("Max Drawdown",      "max_drawdown",    "{:.1%}"),
        ("Aktif Varlık",      "n_active",        "{:d}"),
    ]
    # Sonsuzluk dönebilen (payda sıfırlanabilen) metrikler
    inf_safe = {"sortino", "calmar", "omega"}
    for label, key, fmt in metrics:
        row = f"  {label:<20}"
        for strat in strategies.values():
            val = strat[key]
            if key in inf_safe and not np.isfinite(val):
                row += f"  {'inf':>13}"
            else:
                row += f"  {fmt.format(val):>13}"
        print(row)

    # 100 varlıkta tüm ağırlıkları yazdırmak anlamsız — strateji bazında
    # sadece >0.5% paya sahip olanlar gösterilir.
    print("\n" + "─"*90)
    print(f"  EN YÜKSEK PAYLI {top_k_weights} VARLIK (strateji başına, sıfır olmayan)")
    print("─"*90)
    for strat in strategies.values():
        w     = strat["weights"]
        order = np.argsort(w)[::-1]
        top   = [(tickers[i], w[i]) for i in order if w[i] > 0.005][:top_k_weights]
        print(f"\n  {strat['marker']} {strat['name']}  "
              f"(toplam {strat['n_active']} aktif varlık)")
        line = "    "
        for k, (t, ww) in enumerate(top, 1):
            line += f"{t}:{ww:.1%}  "
            if k % 5 == 0 and k != len(top):
                line += "\n    "
        print(line)


# ─────────────────────────────────────────────
# 9. GÖRSELLEŞTİRME  (100 coin uyumlu)
# ─────────────────────────────────────────────

def plot_results(
    strategies: dict,
    pareto_weights: np.ndarray,
    pareto_F: np.ndarray,
    returns_df: pd.DataFrame,
    save_path: str = "portfolio_results.png",
    top_k_bars: int = 20,
    periods_per_year: int = 365,
) -> None:
    """
    4 panel:
      (a) Pareto Cephesi  — CVaR vs Getiri
      (b) Ağırlık dağılımı — en çok seçilen TOP-K varlık (birleşik)
      (c) Kümülatif getiri
      (d) Risk-getiri scatter  (stratejiler + benchmark)
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "font.size":        10,
    })

    fig = plt.figure(figsize=(17, 13))
    fig.suptitle(
        "NSGA-III Kripto Portföy Optimizasyonu  —  Top-100 Hacimli Evren\n"
        "CVaR & Getiri Çok Amaçlı Pareto  |  Sharpe + Sortino Seçim",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    tickers  = returns_df.columns.tolist()
    ret_mat  = returns_df.values
    mean_ret = returns_df.mean().values
    cov_mat  = returns_df.cov().values
    ann      = periods_per_year

    # ── (a) Pareto Cephesi ─────────────────────────────────────────────
    pareto_ret  = -pareto_F[:, 0] * ann
    pareto_cvar =  pareto_F[:, 1] * np.sqrt(ann)
    ax1.scatter(pareto_cvar, pareto_ret, c="steelblue", alpha=0.35, s=20,
                label="Pareto çözümleri", zorder=2)
    for strat in strategies.values():
        ax1.scatter(strat["cvar_annual"], strat["annual_return"],
                    color=strat["color"], s=160, zorder=5,
                    edgecolors="white", linewidths=1.5, label=strat["name"])
        ax1.annotate(f"{strat['marker']} {strat['name']}",
                     (strat["cvar_annual"], strat["annual_return"]),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=8, color=strat["color"], fontweight="bold")
    ax1.set_xlabel("CVaR Yıllık (risk)", fontsize=10)
    ax1.set_ylabel("Beklenen Yıllık Getiri", fontsize=10)
    ax1.set_title("(a) Pareto Cephesi", fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax1.legend(fontsize=8, loc="lower right")

    # ── (b) Stratejilerde en çok seçilen TOP-K varlık ──────────────────
    weight_stack = np.stack([s["weights"] for s in strategies.values()])
    total_w      = weight_stack.sum(axis=0)
    top_idx      = np.argsort(total_w)[::-1][:top_k_bars]
    top_tickers  = [tickers[i] for i in top_idx]

    n_strats  = len(strategies)
    x         = np.arange(len(top_tickers))
    bar_width = 0.8 / n_strats
    offsets   = np.linspace(-(n_strats - 1) / 2, (n_strats - 1) / 2, n_strats) * bar_width

    for i, (key, strat) in enumerate(strategies.items()):
        w_pct = strat["weights"][top_idx] * 100
        ax2.bar(x + offsets[i], w_pct, bar_width, label=strat["name"],
                color=strat["color"], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(top_tickers, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Ağırlık (%)", fontsize=10)
    ax2.set_title(f"(b) En Çok Seçilen {top_k_bars} Varlık", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ── (c) Kümülatif getiri ───────────────────────────────────────────
    for strat in strategies.values():
        w      = strat["weights"]
        cumret = (1 + ret_mat @ w).cumprod() - 1
        ax3.plot(returns_df.index, cumret * 100, label=strat["name"],
                 color=strat["color"], linewidth=1.8)

    # Eşit ağırlıklı benchmark
    eq_w   = np.ones(len(tickers)) / len(tickers)
    eq_cum = (1 + ret_mat @ eq_w).cumprod() - 1
    ax3.plot(returns_df.index, eq_cum * 100,
             label="Eşit Ağırlık (benchmark)",
             color="gray", linewidth=1.2, linestyle="--", alpha=0.7)

    ax3.set_ylabel("Kümülatif Getiri (%)", fontsize=10)
    ax3.set_title("(c) Kümülatif Getiri", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax3.tick_params(axis="x", rotation=30, labelsize=8)

    # ── (d) Risk-getiri scatter (bireysel varlık + stratejiler) ────────
    ind_rets = mean_ret * ann
    ind_vols = np.sqrt(np.diag(cov_mat)) * np.sqrt(ann)
    ax4.scatter(ind_vols, ind_rets, c="lightgray", s=35, zorder=2,
                label="Tekil varlıklar", alpha=0.6)

    # 100 coin için annotate → sadece en yüksek Sharpe'a sahip 10 tanesi
    ind_sharpe = ind_rets / (ind_vols + 1e-9)
    top_ind    = np.argsort(ind_sharpe)[::-1][:10]
    for i in top_ind:
        ax4.annotate(tickers[i], (ind_vols[i], ind_rets[i]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points", color="gray")

    for strat in strategies.values():
        ax4.scatter(strat["annual_vol"], strat["annual_return"],
                    color=strat["color"], s=170, zorder=5,
                    edgecolors="white", linewidths=1.5, label=strat["name"])

    ax4.set_xlabel("Yıllık Volatilite", fontsize=10)
    ax4.set_ylabel("Yıllık Getiri", fontsize=10)
    ax4.set_title("(d) Risk-Getiri Uzayı", fontweight="bold")
    ax4.legend(fontsize=7, loc="best")
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Grafik] Kaydedildi → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 10. BACKTEST  (değişmedi — sadece Sortino ekle)
# ─────────────────────────────────────────────

def walk_forward_backtest(
    returns_df: pd.DataFrame,
    train_window: int = 365,
    rebalance_every: int = 30,
    n_gen: int = 150,
    pop_size: int = 200,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Walk-forward backtest:
      - Her rebalance_every günde portföyü NSGA-III ile optimize et
      - Son train_window gün = eğitim penceresi
      - Balanced stratejisi uygulanır
    """
    print("\n[Backtest] Walk-forward başlatılıyor...")
    n_days  = len(returns_df)
    results = []

    for start_idx in range(train_window, n_days, rebalance_every):
        train = returns_df.iloc[start_idx - train_window:start_idx]
        test  = returns_df.iloc[start_idx:start_idx + rebalance_every]
        if len(test) == 0:
            break

        opt = run_nsga3_optimization(
            train, n_obj=2, pop_size=pop_size, n_gen=n_gen, verbose=False,
        )
        strats = select_portfolio_strategies(
            opt["pareto_weights"], opt["pareto_F"], train,
        )
        w = strats["balanced"]["weights"]

        test_returns  = test.values @ w
        period_cumret = (1 + test_returns).prod() - 1

        results.append({
            "date":          test.index[-1],
            "period_return": period_cumret,
            "sharpe":        compute_sharpe(w, train.mean().values, train.cov().values),
            "sortino":       compute_sortino(w, train.values),
            "cvar":          compute_cvar(w, train.values),
            "n_active":      int(np.sum(w > 0.005)),
        })
        if verbose:
            print(f"  [{test.index[0].date()} → {test.index[-1].date()}] "
                  f"Getiri: {period_cumret:.2%}  CVaR: {results[-1]['cvar']:.3%}")

    bt_df = pd.DataFrame(results).set_index("date")
    bt_df["cumulative_return"] = (1 + bt_df["period_return"]).cumprod() - 1

    print("\n[Backtest Sonuç]")
    print(f"  Toplam getiri       : {bt_df['cumulative_return'].iloc[-1]:.1%}")
    print(f"  Ort. Sharpe (period): {bt_df['sharpe'].mean():.3f}")
    print(f"  Ort. Sortino        : {bt_df['sortino'].mean():.3f}")
    print(f"  Ort. CVaR (günlük)  : {bt_df['cvar'].mean():.3%}")
    return bt_df


# ─────────────────────────────────────────────
# 11. ANA PROGRAM
# ─────────────────────────────────────────────

def main(
    top_n: int = 100,
    min_assets: int = 5,
    max_assets: int = 20,
    seed: int = 42,
) -> dict:
    # ── A) Top-N coin (piyasa değerine göre, filtreli) ────────────────
    top_coins_df = fetch_top_coins(
        n_target=top_n,
        sort_by="market_cap",
        min_market_cap_usd=1e9,
        exclude_stables=True,
        exclude_exchange=True,
        exclude_wrapped=True,
    )

    if top_coins_df.empty or "symbol" not in top_coins_df.columns:
        raise RuntimeError(
            "Coin listesi oluşturulamadı — CoinGecko API'ye erişim yok ve "
            "fallback listesi de boş."
        )

    symbols = top_coins_df["symbol"].tolist()

    # ── B) Fiyat verisi — DB önce, yfinance fallback ─────────────────
    returns_df = load_prices_from_db_or_fetch(
        symbols,
        lookback_days=750,
        min_lookback_days=400,
    )

    # ── C) Sanity filter ──────────────────────────────────────────────
    returns_df = apply_sanity_filter(
        returns_df,
        min_annualized_vol=0.20,
        max_mean_daily_return=0.03,
        max_single_day_abs_return=3.0,
    )

    # ── D) Winsorization ──────────────────────────────────────────────
    returns_df = winsorize_returns(
        returns_df,
        lower_pct=0.005,
        upper_pct=0.995,
    )

    if len(returns_df.columns) < 10:
        raise RuntimeError(
            f"Tüm filtreler sonrası sadece {len(returns_df.columns)} coin kaldı."
        )

    # ── E) NSGA-III optimizasyonu ─────────────────────────────────────
    result = run_nsga3_optimization(
        returns_df,
        n_obj=2,
        pop_size=300,
        n_gen=400,
        alpha=0.05,
        max_weight=0.10,
        min_assets=min_assets,
        max_assets=max_assets,
        transaction_cost=0.001,
        seed=seed,
        verbose=True,
    )

    # ── F) Strateji seçimi ────────────────────────────────────────────
    strategies = select_portfolio_strategies(
        result["pareto_weights"],
        result["pareto_F"],
        returns_df,
        alpha=0.05,
    )

    # ── G) JSON-serializable çıktı ────────────────────────────────────
    tickers = returns_df.columns.tolist()
    output_strategies = {}
    for key, strat in strategies.items():
        w = strat["weights"]
        order = sorted(range(len(w)), key=lambda i: -w[i])
        holdings = {
            tickers[i]: round(float(w[i]), 6)
            for i in order if w[i] > 0.005
        }

        def _safe(v):
            return float(v) if np.isfinite(v) else None

        output_strategies[key] = {
            "name":          strat["name"],
            "annual_return": float(strat["annual_return"]),
            "annual_vol":    float(strat["annual_vol"]),
            "sharpe":        _safe(strat["sharpe"]),
            "sortino":       _safe(strat["sortino"]),
            "calmar":        _safe(strat["calmar"]),
            "omega":         _safe(strat["omega"]),
            "var_daily":     float(strat["var_daily"]),
            "cvar_daily":    float(strat["cvar_daily"]),
            "cvar_annual":   float(strat["cvar_annual"]),
            "max_drawdown":  float(strat["max_drawdown"]),
            "n_active":      int(strat["n_active"]),
            "holdings":      holdings,
        }

    return {
        "n_coins":        len(tickers),
        "n_observations": len(returns_df),
        "tickers":        tickers,
        "strategies":     output_strategies,
    }


# ─────────────────────────────────────────────
# ÇALIŞTIR
# ─────────────────────────────────────────────
if __name__ == "__main__":
    result, strategies, returns_df = main()
