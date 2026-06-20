import pytest
from django.core.cache import cache


@pytest.fixture(autouse=True)
def clear_throttle_cache():
    """Throttle sayaclari cache'te tutulur ve DB rollback'inden etkilenmez;
    testler arasinda sizmamasi icin her testte temizlenir."""
    cache.clear()
    yield
    cache.clear()
