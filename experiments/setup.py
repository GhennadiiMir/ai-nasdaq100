import logging
import os

modelName = "cnn_lstm"

# check if results directory exists
if not os.path.isdir("./results/%s" %(modelName)):
    os.makedirs("./results/%s" %(modelName))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
dlogger = logging.getLogger(__name__)
dlogger.setLevel(logging.DEBUG)

# create file handler
log = logging.FileHandler("./results/%s/%s_results.log" %(modelName, modelName))
log.setLevel(logging.INFO)
debug_log = logging.FileHandler("./results/%s/%s_debug.log" %(modelName, modelName))
debug_log.setLevel(logging.DEBUG)

logger.addHandler(log)
dlogger.addHandler(debug_log)