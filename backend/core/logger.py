# core/logger.py

import logging

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,  # Hiện log từ cấp DEBUG trở lên
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
