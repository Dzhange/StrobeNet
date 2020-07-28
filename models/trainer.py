"""
A basic trainer to train models, 
based on the implementation of Jiahui and Srinath
To be specific:
The training pipeline would basicly comes from Jiahui's code
(like gradient bp part)
But I would use several utils from Srinath(checkpoints)
"""
import traceback
import torch
from utils.DataUtils import *
from loaders.HandDataset import *

class Trainer:
    """
    this class does the training job
    """
    def __init__(self, config, model, data_loader_dict, objective, device):
        self.model = model
        self.config = config
        self.train_data_loader = data_loader_dict['train']
        self.val_data_loader = data_loader_dict['val']
        self.objective = objective
        self.device = device
        self.model.net.to(device)        

    def train(self):
        self.model.setup_checkpoint(self.device)
        self.model.net.train()
        cur_epoch = 0

        AllTic = getCurrentEpochTime()
        while cur_epoch < self.config.EPOCH_TOTAL:
            try:
                epoch_losses = [] # For all batches in an epoch
                Tic = getCurrentEpochTime()
                # for data in self.train_data_loader:
                for i, data in enumerate(self.train_data_loader, 0):  # Get each batch
                    ################### START WEIGHT UPDATE ################################                    
                    self.model.optimizer.zero_grad()
                    net_input, target = self.model.preprocess(data, self.device)
                    output = self.model.net(net_input)
                    loss = self.objective(output, target)
                    loss.backward()
                    self.model.optimizer.step()
                    epoch_losses.append(loss.item())
                    ####################### START MONITOR ################################
                    Toc = getCurrentEpochTime()
                    Elapsed = math.floor((Toc - Tic) * 1e-6)
                    TotalElapsed = math.floor((Toc - AllTic) * 1e-6)
                    done = int(50 * (i+1) / len(self.train_data_loader))
                    # Compute ETA
                    TimePerBatch = (Toc - AllTic) / ((cur_epoch * len(self.train_data_loader)) + (i+1)) # Time per batch
                    ETA = math.floor(TimePerBatch * self.config.EPOCH_TOTAL * len(self.train_data_loader) * 1e-6)

                    ProgressStr = ('\r[{}>{}] epoch - {}/{}, train loss - {:.8f} | epoch - {}, total - {} ETA - {} |')\
                                    .format('=' * done, '-' * (50 - done), self.model.start_epoch + cur_epoch + 1, self.model.start_epoch + self.config.EPOCH_TOTAL,
                                    np.mean(np.asarray(epoch_losses)), getTimeDur(Elapsed), getTimeDur(TotalElapsed), getTimeDur(ETA-TotalElapsed))
                    sys.stdout.write(ProgressStr.ljust(150))
                    sys.stdout.flush()                    
                    ########################## END MONITOR ################################
                sys.stdout.write('\n')
                cur_epoch += 1
            except (KeyboardInterrupt, SystemExit):
                print('\n[ INFO ]: KeyboardInterrupt detected. Saving checkpoint.')
                self.model.save_checkpoint(cur_epoch, time_string='eot', print_str='$'*3)
                break
            except Exception as error:
                print(traceback.format_exc())
                print('\n[ WARN ]: Exception detected. *NOT* saving checkpoint. {}'.format(error))
                break
