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
