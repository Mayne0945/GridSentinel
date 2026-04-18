import logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [depot] %(message)s")
if __name__ == "__main__":
    logging.info("MPC Depot Module Online.")
    while True: time.sleep(60)
