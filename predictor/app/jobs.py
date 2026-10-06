"""Background jobs (file processing, analysis) on one worker thread.

One worker keeps SQLite writes serialised and the machine responsive. Job state for files
and runs lives in the database, so the UI only polls those rows.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from ..utils.logging import get_logger

log = get_logger("jobs")


class JobRunner:
    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predictor-job")
        self._lock = threading.Lock()
        self._pending = 0

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        with self._lock:
            self._pending += 1

        def wrapped():
            try:
                return fn(*args)
            except Exception:  # errors are recorded on the file/run rows by the services
                log.debug("job raised", exc_info=True)
                return None
            finally:
                with self._lock:
                    self._pending -= 1

        return self._pool.submit(wrapped)

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
