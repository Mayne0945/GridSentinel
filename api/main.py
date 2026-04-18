import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [api] %(message)s")
log = logging.getLogger(__name__)
if __name__ == "__main__":
    log.info("GridSentinel API Node online.")
    while True:
        time.sleep(60)
