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
    cfg.TRANSFORM = True # if set too true, we use the dataset where all mesh/occupancy are transformed into [-0.5, 0.5]    

    cfg.FIX_HACK = False
    
    # for animation
    cfg.ANIM_MODEL = False
    cfg.ANIM_MODEL_ID = 0    
    cfg.ANGLE_START = 0.0 
    cfg.ANGLE_STEP = 0.1

    cfg.VIEW_NUM = 1 # number of seen views
    cfg.VIEW_RECON = 8
    cfg.RANDOM_VIEW = False # select random views from the given set
    cfg.TOTAL_VIEW = 4 # total number of views
    cfg.NUM_SAMPLE_POINTS = 50000 

    cfg.GEN_OUTPUT = 'None'
    cfg.GEN_INPUT = 'None'

    # fundamental setting
    cfg.ROOT_DIR = os.getcwd()
    cfg.CONFIG_FILE = 'None'
    cfg.MODES = ['train', 'vali']
    cfg.TRAIN = True # train or val?
    cfg.GPU = 0

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
    cfg.LOGGER = 'logger_v1'
    cfg.LOGGER_SELECT = ['metric']    

    cfg.BACKUP_FILES = []

    #added by Ge
    cfg.OUTPUT_DIR = '' # path to store everything
    cfg.VAL_DIR = 'ValResults' # name of current validation 
    cfg.SAVE_FREQ = 1 # save weights freguency
    cfg.DATA_LIMIT = 10 # percentage of training data to use(sampled with fixed step)
    cfg.VAL_DATA_LIMIT = 10 # percentage of validation data to use(sampled with fixed step)
    cfg.IMAGE_SIZE = (320, 240)
    cfg.EXPT_NAME = "UNKNOWN"
    
    # DATASET PART
    cfg.TEST_ON_TRAIN = False
    # cfg.OUT_CHANNELS = 4 ## discarded! now we calculate output channels automatically
    cfg.TASK = "occupancy"
    cfg.TARGETS = ["nox00"]
    
    cfg.NRNET_TYPE = "out_feature" # inter_feature
    cfg.NRNET_PRETRAIN = False
    cfg.NRNET_PRETRAIN_PATH = "./" 
    
    cfg.UPDATE_SEG = True
    cfg.IF_BN = False # use batch normalization in IF-Net
    cfg.BN = False # use batch normalization in SegNet
    cfg.AGGR_SCATTER = False # first aggregate canonical point cloud among views, then use scatter to voxelize
    cfg.RESOLUTION = 128 # axial resolution for final reconstruction

    cfg.USE_FEATURE = True # ablation: see how network performs when we do not give it feature
    cfg.PRED_FEATURE = True # generate feature from segnet prediction
    cfg.FEATURE_CHANNELS = 64 # dimension of features
    cfg.SEP_POSE = False # repose the union point cloud into individual pose using individual joint position

    cfg.MH = False # use multi-headed version of segnet
    cfg.AS_SEG = False # ??? treat skinning weights as binary(so it's segmentation)

    cfg.BONE_NUM = 16 # The number of joints of target objects
    
    ### Weight for losses
    cfg.NOCS_LOSS = 0
    cfg.MASK_LOSS = 0
    cfg.SKIN_LOSS = 0
    cfg.LOC_LOSS = 0
    cfg.LOC_MAP_LOSS = 0
    cfg.POSE_LOSS = 0
    cfg.POSE_MAP_LOSS = 0
    cfg.RECON_LOSS = 1.0
    cfg.SMOOTH_L1 = False
    cfg.KEEP_SQUARE = False

    cfg.CONSISTENCY = 0.0 # weight of consistency loss
    cfg.DYNAMIC_CONSISTENCY = False # if set to true, will use the chamfer dist from corresponding points to generate a consistency loss
    cfg.SAVE_CRR = False # save consistency intermediate results for evaulation and debugging

    cfg.GLOBAL_FEATURE = False # use global feature from SegNet
    cfg.GLOBAL_ONLY = False # use only global feature for final reconstruction

    cfg.IF_SHALLOW = False
    cfg.SUPER_RES = False
    cfg.IF_IN_DIM = 16
    cfg.CANO_IF = True # if set to True, we use only the canonical occupancy as supervision
    cfg.ONET_CANO = False
    cfg.ONET_MV_CANO = True

    # for final pipeline
    cfg.STAGE_ONE = True # if set to true, we do not supervise the occupancy
    cfg.REPOSE = True # if set to true, we calculate PNNOCS and use it for final reconstruction

    # for optimizer
    cfg.WEIGHT_DECAY = 0.0

    return cfg
