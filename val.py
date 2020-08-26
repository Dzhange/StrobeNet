"""
Main program
May the Force be with you.
Jiahui @ March,2019
"""
from torch.utils.data import DataLoader
import torch
# from loaders import get_dataset
from loaders.HandDataset import HandDataset
from loaders.HandDatasetLBS import HandDatasetLBS
from loaders.HandOccDataset import HandOccDataset
# from inout.logger import get_logger
from models.NOCS import ModelNOCS
from models.Baseline import ModelIFNOCS
from models.LBS import ModelLBSNOCS
from models.SegLBS import ModelSegLBS
from models.loss import *
# from core.coach import Coach
from models.validater import Validater
from config import get_cfg

# preparer configuration
cfg = get_cfg()


Dataset = HandDataset
Model = ModelNOCS(cfg)

task = cfg.TASK

if task == "lbs":
    Dataset = HandDatasetLBS
    objective = LBSLoss()
    Model = ModelLBSNOCS(cfg)
if task == "lbs_seg":
    Dataset = HandDatasetLBS
    objective = LBSSegLoss()
    Model = ModelSegLBS(cfg)
if task == "occupancy":
    Dataset = HandOccDataset
    objective = MixLoss()
    Model = ModelIFNOCS(cfg)
if task == "pretrain":
    objective = L2MaskLoss_wtFeature()
if task == "nocs":
    objective = L2MaskLoss()


# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()

val_dataset = Dataset(root=cfg.DATASET_ROOT, train=cfg.TEST_ON_TRAIN,
                      limit=cfg.DATA_LIMIT, img_size=cfg.IMAGE_SIZE,
                      frame_load_str=["color00", cfg.TARGETS])

val_dataloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False,
                            num_workers=1, drop_last=True)

# register dataset, models, logger to trainer
device = torch.device(cfg.GPU)
validater = Validater(cfg, Model, val_dataloader, objective, device)

# start
validater.validate()
