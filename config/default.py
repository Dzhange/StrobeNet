"""
This .py provides basic
"""
import os
from yacs.config import CfgNode as CN

def get_default_cfg():
    """
    The config file from Jiahui's implementation
    """
    cfg = CN()
    # dataset
    cfg.DATASET = ''
    cfg.DATASET_ROOT = ''
    cfg.DATASET_CATES = []
    cfg.DATASET_INDEX = []
    cfg.PROPORTION = 1.0  # how many of the first K samples in the dataset list

    # fundamental setting
    cfg.ROOT_DIR = os.getcwd()
    cfg.CONFIG_FILE = 'None'
    cfg.MODES = ['train', 'vali']
    cfg.GPU = [0]

    cfg.BATCHSIZE = 2
    cfg.DATALOADER_WORKERS = 2

    cfg.MODEL = 'None'
    cfg.MODEL_INIT_PATH = ['None']
    cfg.LR = 0.001
    cfg.EPOCH_TOTAL = 1

    # optimizer config
    cfg.ADAM_BETA1 = 0.5
    cfg.ADAM_BETA2 = 0.9
    cfg.OPTI_DECAY_RATE = 0.5
    cfg.OPTI_DECAY_INTERVAL = 20
    cfg.OPTI_DECAY_MIN = 0.00001

    # log
    cfg.LOG_DIR = 'debug'  # All log in a dir, this param is the name under $ProjectRoot/log/

    # If true, check whether the log dir exists, \
    # if doesn't, do not resume. just start a new one
    cfg.RESUME = False
    cfg.RESUME_EPOCH_ID = 0

    cfg.LOGGER = 'logger_v1'
    cfg.LOGGER_SELECT = ['metric']
    cfg.MODEL_SAVE_PER_N_EPOCH = 5
    cfg.VIS_PER_N_EPOCH = 1
    cfg.VIS_PER_N_BATCH = 1
    cfg.VIS_ONE_PER_BATCH = True
    cfg.VIS_TRAIN_PER_BATCH = 20

    cfg.BACKUP_FILES = []
    
    #added by Ge
    cfg.OUTPUT_DIR = ''
    cfg.SAVE_FREQ = 1
    cfg.DATA_LIMIT = 10
    cfg.IMAGE_SIZE = (320,240)
    cfg.EXPT_NAME = "UNKNOWN"
    return cfg
