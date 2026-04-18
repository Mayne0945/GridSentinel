"""mpc/depot/main.py — stub"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [depot] %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("depot engine ready. Waiting for Phase implementation.")
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
