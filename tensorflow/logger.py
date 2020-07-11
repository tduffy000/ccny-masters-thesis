import logging

FORMAT = "[%(levelname)-2s] {%(name)-2s} %(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("tensorflow")
logger.setLevel(logging.DEBUG)
