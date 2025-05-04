# core/logger.py

import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,  # Hiện log từ cấp INFO trở lên
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
