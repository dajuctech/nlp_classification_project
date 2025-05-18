# Utility functions
import logging

def init_logging(log_file='outputs/logs/run.log'):
    logging.basicConfig(filename=log_file, level=logging.INFO)
    return logging.getLogger(__name__)