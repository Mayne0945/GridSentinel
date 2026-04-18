"""chaos/main.py — stub"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [chaos] %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("chaos engine ready. Waiting for Phase implementation.")
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
