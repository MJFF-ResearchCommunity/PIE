import logging
import sys

logger = logging.getLogger("PIE")
logger.setLevel(logging.INFO)
if not logger.handlers:  # pie_clean configures the same "PIE" logger; one handler, not two
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

# Make pie a proper package
