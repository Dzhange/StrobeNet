"""
Main program
May the Force be with you.
Jiahui @ March,2019
"""
from torch.utils.data import DataLoader
import torch
import numpy as np
# from loaders import get_dataset
from loaders.HandDataset import HandDataset
from loaders.HandDatasetLBS import HandDatasetLBS
from loaders.HandOccDataset import HandOccDataset
from loaders.SAPIENDataset import SAPIENDataset
from loaders.MVSAPIENDataset import MVSPDataset
from loaders.MVMPDataset import MVMPDataset

# from inout.logger import get_logger
from models.NOCS import ModelNOCS
from models.Baseline import ModelIFNOCS
from models.LBS import ModelLBSNOCS
from models.SegLBS import ModelSegLBS
from models.PMLBS import PMLBS
from models.LNR import ModelLNRNET
from models.MLNR import ModelMLNRNet
from models.MVMPLNR import ModelMVMPLNRNet

# from models.im2mesh.onet.models import OccupancyNetwork

from models.loss import *
from models.sapien_loss import *
# from core.coach import Coach
from models.trainer import Trainer
from config import get_cfg

# preparer configuration
cfg = get_cfg()
cfg.defrost()
cfg.TRAIN = True
cfg.freeze()

# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()

# lbs = cfg.LBS
task = cfg.TASK

def get_loaders(Dataset):
    for mode in cfg.MODES:
  
        if task in ["lnrnet", 'mlnrnet', 'mvmp']:
            phase_dataset = Dataset(config=cfg, train=mode in ['train'] or cfg.TEST_ON_TRAIN)
        else:
            phase_dataset = Dataset(root=cfg.DATASET_ROOT,
                            train=mode in ['train'] or cfg.TEST_ON_TRAIN,
                            limit=cfg.DATA_LIMIT if mode == 'train' else cfg.VAL_DATA_LIMIT,
                            img_size=cfg.IMAGE_SIZE,
                            frame_load_str=None if "default" in cfg.TARGETS else cfg.TARGETS)


        print("[ INFO ] {} dataset has {} elements.".format(mode, len(phase_dataset)))
        dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                        shuffle=mode in ['train'] or cfg.TEST_ON_TRAIN,
                                        num_workers=cfg.DATALOADER_WORKERS, drop_last=True)
Dataset = None
Model = None

if __name__ == "__main__":

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    if task == "lbs":
        Dataset = HandDatasetLBS
        objective = LBSLoss(cfg)
        Model = ModelLBSNOCS(cfg)
    if task == "lbs_seg":
        Dataset = HandDatasetLBS # set as seg = true
        objective = LBSSegLoss(cfg)
        Model = ModelSegLBS(cfg)
    if task == "occupancy":
        Dataset = HandOccDataset
        objective = MixLoss()
        Model = ModelIFNOCS(cfg)
    if task == "pretrain":
        objective = L2MaskLoss_wtFeature()
    if task == "nocs":
        objective = L2MaskLoss()
    if task == "sapien_lbs":
        Dataset = SAPIENDataset
        objective = PMLBSLoss(cfg)
        Model = PMLBS(cfg)
    if task == "lnrnet":
        Dataset = SAPIENDataset
        # objective = PMLBSLoss(cfg)
        objective = PMLoss(cfg)    
        Model = ModelLNRNET(cfg)
    if task == "mlnrnet":
        Dataset = MVSPDataset
        objective = MVPMLoss(cfg)
        Model = ModelMLNRNet(cfg)
    if task == "mvmp":
        Dataset = MVMPDataset
        objective = MVMPLoss(cfg)
        Model = ModelMVMPLNRNet(cfg)
    ####################################    
    

    get_loaders(Dataset)
    device = torch.device(cfg.GPU)
    print("[ INFO ] Running on device ", device)
    trainer = Trainer(cfg, Model, dataloader_dict, objective, device)
    # start
    trainer.train()
