import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")


# A full run is a multi-minute job; the console handler is what makes it possible to see
# where it is rather than watching a silent terminal.
logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # This will give all the information, we can also set for ERROR
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)


@contextmanager
def timer(label: str):
    """Log wall time for a block."""
    t0 = perf_counter()
    yield
    logging.info("%s -- %.1fs", label, perf_counter() - t0)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger