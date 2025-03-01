import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger

def validate_sensor_data(data):
    # Logic to validate incoming sensor data
    pass
