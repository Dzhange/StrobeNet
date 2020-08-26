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
from models.trainer import Trainer
from config import get_cfg

# preparer configuration
cfg = get_cfg()

# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()

# lbs = cfg.LBS
task = cfg.TASK


def get_loaders(Dataset):
    for mode in cfg.MODES:
        phase_dataset = Dataset(root=cfg.DATASET_ROOT,
                                train=mode in ['train'] or cfg.TEST_ON_TRAIN,
                                limit=cfg.DATA_LIMIT, img_size=cfg.IMAGE_SIZE,
                                frame_load_str=["color00", cfg.TARGETS])

        print("[ INFO ] {} dataset has {} elements.".format(mode, len(phase_dataset)))
        dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                        shuffle=mode in ['train'] or cfg.TEST_ON_TRAIN,
                                        num_workers=cfg.DATALOADER_WORKERS, drop_last=True)

Dataset = HandDataset
Model = ModelNOCS(cfg)

if task == "lbs":
    Dataset = HandDatasetLBS
    objective = LBSLoss()
    Model = ModelLBSNOCS(cfg)
if task == "lbs_seg":
    Dataset = HandDatasetLBS # set as seg = true
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
if task == "joints":
    pass 

get_loaders(Dataset)

device = torch.device(cfg.GPU)
print("[ INFO ] Running on device ", device)
trainer = Trainer(cfg, Model, dataloader_dict, objective, device)
# start
trainer.train()
