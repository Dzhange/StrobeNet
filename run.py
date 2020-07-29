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
from models.loss import L2MaskLoss,L2Loss
# from core.coach import Coach
from models.trainer import Trainer
from config import get_cfg

# preparer configuration
cfg = get_cfg()

# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()
for mode in cfg.MODES:
    phase_dataset = HandDataset(root=cfg.DATASET_ROOT, train=True if mode in ['train'] or cfg.TEST_ON_TRAIN else False,
                                limit=cfg.DATA_LIMIT, img_size=cfg.IMAGE_SIZE,
                                frame_load_str=["color00", "nox00"])
    print(len(phase_dataset))
    dataloader_dict[mode] = DataLoader(phase_dataset, batch_size=cfg.BATCHSIZE,
                                       shuffle=True if mode in ['train']else False,
                                       num_workers=cfg.DATALOADER_WORKERS, pin_memory=True,
                                       drop_last=True)

# prepare models
model = model_NOCS(cfg)

# prepare logger
# LoggerClass = get_logger(cfg.LOGGER)
# logger = LoggerClass(cfg)

# register dataset, models, logger to trainer
objective = L2MaskLoss()
# objective = L2Loss()
device = torch.device("cuda:1")
trainer = Trainer(cfg, model, dataloader_dict, objective, device)
# start
trainer.train()
