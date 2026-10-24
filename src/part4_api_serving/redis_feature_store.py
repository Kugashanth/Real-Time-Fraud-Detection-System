import time
from collections import defaultdict


class InMemoryVelocityFallback:
    """Sub-millisecond velocity tracker with automatic fallback when Redis is unreachable."""
    def __init__(self):
        self._store: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record_and_fetch_velocity(self, card_id: str, amount: float, window_sec: float = 3600.0) -> dict[str, float]:
        now = time.time()
        history = [item for item in self._store[card_id] if now - item[0] <= window_sec]
        history.append((now, amount))
        self._store[card_id] = history
        tx_count_1h = len(history)
        mean_amt = sum(x[1] for x in history) / tx_count_1h
        return {
            "tx_count_1h": float(tx_count_1h),
            "amount_to_mean_ratio": round(amount / max(mean_amt, 1.0), 4),
        }
