"""Zaman serisi için kronolojik train/val/test bölme yardımcıları.

Bu modül bilerek ML kütüphanelerinden (tensorflow/prophet) bağımsız tutuldu;
testlerde ağır import olmadan hızlıca doğrulanabilir.
"""


def chronological_split_indices(n_samples: int,
                                train_frac: float = 0.7,
                                val_frac: float = 0.15) -> tuple:
    """
    n_samples uzunluğundaki sıralı bir diziyi kronolojik olarak üçe bölen
    (train_end, val_end) indekslerini döndürür:

        train = [0, train_end)
        val   = [train_end, val_end)
        test  = [val_end, n_samples)

    Zaman serisi olduğu için karıştırma yapılmaz; parçalar örtüşmez ve
    her parçanın en az 1 örnek alması garanti edilir.
    """
    if n_samples < 3:
        raise ValueError(
            f"Üçlü split için en az 3 örnek gerekir, verilen: {n_samples}")
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError(
            f"Geçersiz split oranları: train={train_frac}, val={val_frac}; "
            f"her ikisi pozitif ve toplamı 1'den küçük olmalı.")

    train_end = max(1, int(n_samples * train_frac))
    val_end = max(train_end + 1, int(n_samples * (train_frac + val_frac)))
    val_end = min(val_end, n_samples - 1)
    train_end = min(train_end, val_end - 1)
    return train_end, val_end


def naive_baseline_metrics(values, test_start: int) -> tuple:
    """
    Naive persistence baseline'ın test dönemi MAE/RMSE'sini döndürür.

    Persistence tahmini: test dönemindeki her gün için tahmin, bir önceki
    günün GERÇEK değeridir (yarın = bugün). `values` kronolojik sıralı tam
    seri, `test_start` test/holdout setinin başladığı indekstir; metrik
    [test_start, len(values)) aralığındaki günler üzerinde hesaplanır.

    MAE/RMSE model fonksiyonlarıyla aynı formülle (ortalama mutlak hata,
    kök ortalama kare hata) hesaplanır ki kıyas aynı ölçekte olsun.
    """
    n = len(values)
    if n < 2:
        raise ValueError(
            f"Baseline için en az 2 değer gerekir, verilen: {n}")
    if not 1 <= test_start < n:
        raise ValueError(
            f"test_start 1 ile {n - 1} arasında olmalı, verilen: {test_start}")

    errors = [float(values[i]) - float(values[i - 1])
              for i in range(test_start, n)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
    return mae, rmse
