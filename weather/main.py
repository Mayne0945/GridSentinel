import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [weather] %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info("Weather Producer online. Standing by for Step 1.4 logic.")
    while True:
        time.sleep(60)
