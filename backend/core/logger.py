# core/logger.py

import logging
import sys

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO for production, DEBUG for development
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
        ],
        force=True # Override any previous logging configuration
    )

    # Optional: Set specific log levels for uvicorn and other libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
