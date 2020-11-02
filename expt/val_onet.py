import os
import traceback
import sys
import torch
import torch.nn as nn
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from models.ModelONet import ModelONet
from loaders.OccNetDataset import OccNetDataset
import argparse
from torch.utils.data import DataLoader
from config import get_cfg

from utils.DataUtils import *

def run(config):

    model = ModelONet(config)
    val_dataset = OccNetDataset(root=config.DATASET_ROOT, train=config.TEST_ON_TRAIN, limit=config.VAL_DATA_LIMIT)
        
    val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=True,
                                num_workers=config.DATALOADER_WORKERS, drop_last=True)

    model.setup_checkpoint(model.device)
        
    cur_epoch = 0
    all_tic = getCurrentEpochTime()    

    for i, batch in enumerate(val_loader):
        model.eval_step(batch, i)
        # val_losses.append(loss.item())
        # val_loss_str = '\rmean val loss: {}'.format(np.mean(np.asarray(val_losses)))
        # sys.stdout.write(val_loss_str.ljust(100))
        # sys.stdout.flush()




if __name__ == "__main__":
    cfg = get_cfg()
    run(cfg)

