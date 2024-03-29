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

from models.loss import *
from models.sapien_loss import *
# from core.coach import Coach
from models.validater import Validater
from config import get_cfg

# preparer configuration
cfg = get_cfg()
# cfg.unfreeze()
# print(dir(cfg))
cfg.defrost()
cfg.TRAIN = False
cfg.freeze()
task = cfg.TASK

if task == "lbs":
    Dataset = HandDatasetLBS
    objective = LBSLoss(cfg)
    Model = ModelLBSNOCS(cfg)
if task == "lbs_seg":
    Dataset = HandDatasetLBS
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

# prepare dataset
# DatasetClass = get_dataset(cfg.DATASET)
dataloader_dict = dict()


if "default" in cfg.TARGETS:
    f_str = None
else:
    f_str = cfg.TARGETS

if task in ["lnrnet", 'mlnrnet', 'mvmp']:
    val_dataset = Dataset(config=cfg, train=cfg.TEST_ON_TRAIN)
else:
    val_dataset = Dataset(root=cfg.DATASET_ROOT, train=cfg.TEST_ON_TRAIN,
                      limit=cfg.VAL_DATA_LIMIT, img_size=cfg.IMAGE_SIZE,
                      frame_load_str=f_str)

val_dataloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False,
                            num_workers=cfg.DATALOADER_WORKERS, drop_last=True)

# register dataset, models, logger to trainer
device = torch.device(cfg.GPU)
validater = Validater(cfg, Model, val_dataloader, objective, device)

# start
validater.validate()
