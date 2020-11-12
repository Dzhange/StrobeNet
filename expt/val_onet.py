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
from loaders.OccNetMVDataset import OccNetMVDataset
import argparse
from torch.utils.data import DataLoader
from config import get_cfg

from utils.DataUtils import *

def run(config):


    model = ModelONet(config)
    if config.VIEW_NUM == 1:
        val_dataset = OccNetDataset(config, train=config.TEST_ON_TRAIN)
    else:
        val_dataset = OccNetMVDataset(config, train=config.TEST_ON_TRAIN)


    val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=True,
                                num_workers=config.DATALOADER_WORKERS, drop_last=True)

    model.setup_checkpoint(model.device)
        
    cur_epoch = 0
    all_tic = getCurrentEpochTime()    

    for i, batch in enumerate(val_loader):
        model.eval_step(batch, i)
        print("{} finished".format(i))
        # val_losses.append(loss.item())
        # val_loss_str = '\rmean val loss: {}'.format(np.mean(np.asarray(val_losses)))
        # sys.stdout.write(val_loss_str.ljust(100))
        # sys.stdout.flush()




if __name__ == "__main__":
    cfg = get_cfg()
    run(cfg)

