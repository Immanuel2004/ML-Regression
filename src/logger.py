import os 
import logging
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(),"logs")
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE)

logging.basicConfig(
    filemode=LOG_FILE_PATH,
    level=logging.INFO,
    format=(
        "[%(asctime)s] | PID: %(process)d | Thread: %(threadName)s | "
        "File: %(filename)s | Func: %(funcName)s | Line: %(lineno)d | "
        "Level: %(levelname)s | Message: %(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S"
)
