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
from models.train import Trainer
from config import get_cfg

# preparer configuration
cfg = get_cfg()

# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()
for mode in cfg.MODES:
    phase_dataset = HandDataset(Root=cfg.DATASET_ROOT, Train=True if mode in ['train'] else False,
                                Limit=cfg.DATA_LIMIT, ImgSize=cfg.IMAGE_SIZE,
                                FrameLoadStr=["color00", "nox00"])

    dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                       shuffle=True if mode in ['train'] else False,
                                       num_workers=cfg.DATALOADER_WORKERS, pin_memory=True,
                                       drop_last=True)

# prepare models
model = model_NOCS(cfg)

# prepare logger
# LoggerClass = get_logger(cfg.LOGGER)
# logger = LoggerClass(cfg)

# register dataset, models, logger to trainer
objective = L2MaskLoss()
device = torch.device("cuda:1")
trainer = Trainer(cfg, model, dataloader_dict, objective, device)
# start
trainer.train()
