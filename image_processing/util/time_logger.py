import logging
import time
from contextlib import contextmanager

class TimeLogger:
    def __init__(self, logger):
        self.logger = logger

    def _log_time(self, title: str, duration: float):
        self.logger.info(f"{title:<30}\t{duration:.4f}s")

    @contextmanager
    def measure(self, title: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self._log_time(title, end_time - start_time)
