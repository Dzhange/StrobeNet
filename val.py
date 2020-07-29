"""
Main program
May the Force be with you.
Jiahui @ March,2019
"""
import open3d as o3d  # Open3D and Pytorch has conflict when not compiling from source
from torch.utils.data import DataLoader
import torch
# from loaders import get_dataset
from loaders.HandDataset import HandDataset
# from inout.logger import get_logger
from models.ModelNOCS import model_NOCS
from models.loss import L2MaskLoss
# from core.coach import Coach
from models.validater import Validater
from config import get_cfg

# preparer configuration
cfg = get_cfg()

# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()

val_dataset = HandDataset(Root=cfg.DATASET_ROOT, Train=True if cfg.TEST_ON_TRAIN else False,
                                Limit=cfg.DATA_LIMIT, ImgSize=cfg.IMAGE_SIZE,
                                FrameLoadStr=["color00", "nox00"])

val_dataloader = DataLoader(val_dataset, batch_size=1,
                                       shuffle=False,
                                       num_workers=1, pin_memory=True,
                                       drop_last=True)

# prepare models
model = model_NOCS(cfg)

# register dataset, models, logger to trainer
objective = L2MaskLoss()
device = torch.device("cuda:1")
validater = Validater(cfg, model, val_dataloader, objective, device)

# start
validater.validate()
