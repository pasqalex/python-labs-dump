import logging
import os

from logging.handlers import RotatingFileHandler
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
path = os.path.join(PROJECT_ROOT, 'tmp', 'logs')

def configure_logging() -> None:
    main_handler = logging.FileHandler(f"{path}DEBUG.log", delay=True)
    main_handler.setLevel(logging.DEBUG)
    
    main_handler = RotatingFileHandler(
        filename=f'{path}DEBUG.log',
        maxBytes=1 * 10240 * 10240,
        backupCount=5,
        encoding='utf-8'
    )
    main_handler.setFormatter(
        logging.Formatter(
            ('%(levelname)-4s | %(name)s, %(asctime)s, %(message)s')
        )
    )
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)
    main_logger.addHandler(main_handler)