import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [entso-e] %(message)s")
log = logging.getLogger(__name__)
if __name__ == "__main__":
    log.info("ENTSO-E Grid Data Producer ready.")
    while True:
        time.sleep(60)
