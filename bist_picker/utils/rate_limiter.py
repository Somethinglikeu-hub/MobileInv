"""Thread-safe rate limiter for HTTP requests.

Enforces minimum delay between requests to avoid overwhelming data sources.
Default: 1 second between requests. Configurable per source.
"""

import threading
import time


class RateLimiter:
    """Enforces a minimum delay between consecutive calls.

    Thread-safe via threading.Lock. Each source can have its own
    RateLimiter instance with a different delay.

    Args:
        min_delay: Minimum seconds between requests. Default 1.0.
        name: Optional name for logging purposes.
    """

    def __init__(self, min_delay: float = 1.0, name: str = "default") -> None:
        self._min_delay = min_delay
        self._name = name
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    @property
    def min_delay(self) -> float:
        """Minimum delay in seconds between requests."""
        return self._min_delay

    @property
    def name(self) -> str:
        """Name of this rate limiter instance."""
        return self._name

    def wait(self) -> float:
        """Block until the minimum delay has elapsed since the last call.

        Returns:
            Actual seconds waited (0.0 if no wait was needed).
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait_time = self._min_delay - elapsed

            if wait_time > 0:
                time.sleep(wait_time)
                self._last_call = time.monotonic()
                return wait_time
            else:
                self._last_call = now
                return 0.0

    def __repr__(self) -> str:
        return f"RateLimiter(name={self._name!r}, min_delay={self._min_delay}s)"
