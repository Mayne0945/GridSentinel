import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [coordinator] %(message)s")
if __name__ == "__main__":
    logging.info("MPC Coordinator Module Online.")
    while True: time.sleep(60)
