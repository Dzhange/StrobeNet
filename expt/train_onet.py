import os
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
    train_dataset = OccNetDataset(root=config.DATASET_ROOT, train=True)
    val_dataset = OccNetDataset(root=config.DATASET_ROOT, train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCHSIZE,
                                shuffle=True,
                                num_workers=config.DATALOADER_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCHSIZE,
                                shuffle=True,
                                num_workers=config.DATALOADER_WORKERS, drop_last=True)


    cur_epoch = 0
    all_tic = getCurrentEpochTime()
    while cur_epoch < config.EPOCH_TOTAL:
        cur_epoch += 1
    #     scheduler.step()
        tic = getCurrentEpochTime()
        epoch_losses = [] # For all batches in an epoch
        for i, batch in enumerate(train_loader):
            
            loss = model.train_step(batch)
            epoch_losses.append(loss.item())

            toc = getCurrentEpochTime()
            elapsed = math.floor((toc - tic) * 1e-6)
            total_elapsed = math.floor((toc - all_tic) * 1e-6)
            done = int(30 * (i+1) / len(train_loader))
            # Compute ETA
            time_per_batch = (toc - all_tic) / ((cur_epoch * len(train_loader)) + (i+1)) # Time per batch
            ETA = math.floor(time_per_batch * config.EPOCH_TOTAL * len(train_loader) * 1e-6)
            
            progress_str = ('\r[{}>{}] epoch - {}/{}, {}th step train loss - {:.8f} | epoch - {}, total - {} ETA - {} |')\
                                    .format('=' * done, '-' * (30 - done),
                                            model.start_epoch + cur_epoch + 1,
                                            model.start_epoch + config.EPOCH_TOTAL,
                                            i,
                                            np.mean(np.asarray(epoch_losses)),
                                            getTimeDur(elapsed),
                                            getTimeDur(total_elapsed),
                                            getTimeDur(ETA-total_elapsed))

            print(progress_str)



if __name__ == "__main__":
    cfg = get_cfg()
    run(cfg)

