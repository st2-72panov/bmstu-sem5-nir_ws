import logging
import time
from contextlib import contextmanager

class TimeLogger:
    def __init__(self, logger):
        self.logger = logger

    @contextmanager
    def measure(self, step: str, title: str, level: int=0):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if step == '':
                msg = title
                self.logger.info(f"{msg}: {duration:.4f}s")
            else:
                space = '  ' * level
                msg = f"{step}\t{space}{title}"
                msg = f"{msg:<30}"
                self.logger.info(f"{msg}\t{space}{duration:.4f}s")
