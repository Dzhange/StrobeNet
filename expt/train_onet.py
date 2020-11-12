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
        train_dataset = OccNetDataset(config, train=True)
    else:
        train_dataset = OccNetMVDataset(config, train=True)

    if config.TEST_ON_TRAIN:
        val_dataset = train_dataset
    else:
        if config.VIEW_NUM == 1:
            val_dataset = OccNetDataset(config, train=config.TEST_ON_TRAIN)
        else:
            val_dataset = OccNetMVDataset(config, train=config.TEST_ON_TRAIN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCHSIZE,
                                shuffle=True,
                                num_workers=config.DATALOADER_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCHSIZE,
                                shuffle=True,
                                num_workers=config.DATALOADER_WORKERS, drop_last=True)

    model.setup_checkpoint(model.device)
    
    
    cur_epoch = 0
    all_tic = getCurrentEpochTime()
    while cur_epoch < config.EPOCH_TOTAL:
        try:
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

                sys.stdout.write(progress_str.ljust(100))
                sys.stdout.flush()
                model.loss_history.append(np.mean(np.asarray(epoch_losses)))

            val_losses = []
            for i, batch in enumerate(val_loader):                
                loss = model.val_step(batch)
                val_losses.append(loss.item())
                val_loss_str = '\rmean val loss: {}'.format(np.mean(np.asarray(val_losses)))
                sys.stdout.write(val_loss_str.ljust(100))
                sys.stdout.flush()
            model.val_loss_history.append(np.mean(np.asarray(val_losses)))                        

            if (cur_epoch + 1) % config.SAVE_FREQ == 0:                    
                print("[ INFO ]: Save checkpoint for epoch {}.".format(cur_epoch + 1))
                model.save_checkpoint(cur_epoch, print_str='~'*3)                        

        except (KeyboardInterrupt, SystemExit):
            print('\n[ INFO ]: KeyboardInterrupt detected. Saving checkpoint.')
            model.save_checkpoint(cur_epoch, time_string='eot', print_str='$'*3)
            break
        except Exception as error:
            print(traceback.format_exc())
            print('\n[ WARN ]: Exception detected. *NOT* saving checkpoint. {}'.format(error))
            break

    model.save_checkpoint(cur_epoch, time_string='eot', print_str='$'*3)



if __name__ == "__main__":
    cfg = get_cfg()
    run(cfg)

