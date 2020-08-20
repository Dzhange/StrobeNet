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
import torch.nn as nn
from utils.DataUtils import *
from loaders.HandDataset import *
import gc

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

        all_tic = getCurrentEpochTime()
        while cur_epoch < self.config.EPOCH_TOTAL:            
            try:
                epoch_losses = [] # For all batches in an epoch
                tic = getCurrentEpochTime()
                # for data in self.train_data_loader:
                for i, data in enumerate(self.train_data_loader, 0):  # Get each batch          
                    # break                      
                    ################### START WEIGHT UPDATE ################################
                    self.model.optimizer.zero_grad()
                    net_input, target = self.model.preprocess(data, self.device)
                    output = self.model.net(net_input)
                    
                    loss = self.objective(output, target)
                    loss.backward()
                    self.model.optimizer.step()
                    ####################### START MONITOR ################################
                    epoch_losses.append(loss.item())
                    toc = getCurrentEpochTime()
                    elapsed = math.floor((toc - tic) * 1e-6)
                    total_elapsed = math.floor((toc - all_tic) * 1e-6)
                    done = int(50 * (i+1) / len(self.train_data_loader))
                    # Compute ETA
                    time_per_batch = (toc - all_tic) / ((cur_epoch * len(self.train_data_loader)) + (i+1)) # Time per batch
                    ETA = math.floor(time_per_batch * self.config.EPOCH_TOTAL * len(self.train_data_loader) * 1e-6)

                    progress_str = ('\r[{}>{}] epoch - {}/{}, train loss - {:.8f} | epoch - {}, total - {} ETA - {} |')\
                                            .format('=' * done, '-' * (50 - done),
                                                    self.model.start_epoch + cur_epoch + 1,
                                                    self.model.start_epoch + self.config.EPOCH_TOTAL,
                                                    np.mean(np.asarray(epoch_losses)),
                                                    getTimeDur(elapsed),
                                                    getTimeDur(total_elapsed),
                                                    getTimeDur(ETA-total_elapsed))

                    sys.stdout.write(progress_str.ljust(150))
                    sys.stdout.flush()    
                sys.stdout.write('\n')
                gc.collect()
                self.model.loss_history.append(np.mean(np.asarray(epoch_losses)))
                ######################  DO VALIDATION  ##########################
                val_losses = self.validate()
                self.model.val_loss_history.append(np.mean(np.asarray(val_losses)))
                ########################## SAVE CHECKPOINT ##############################
                if cur_epoch % self.config.SAVE_FREQ == 0 and cur_epoch > 0:
                    
                    print("[ INFO ]: Save checkpoint for epoch {}.".format(cur_epoch))                    
                    self.model.save_checkpoint(cur_epoch, print_str='~'*3)                        
                
                cur_epoch += 1
            except (KeyboardInterrupt, SystemExit):
                print('\n[ INFO ]: KeyboardInterrupt detected. Saving checkpoint.')
                self.model.save_checkpoint(cur_epoch, time_string='eot', print_str='$'*3)
                break
            except Exception as error:
                print(traceback.format_exc())
                print('\n[ WARN ]: Exception detected. *NOT* saving checkpoint. {}'.format(error))
                break

        self.model.save_checkpoint(cur_epoch, time_string='eot', print_str='$'*3)

    def validate(self):
        """
        Do validation, only to record loss, don't save any results
        """    
        self.model.net.eval()         #switch to evaluation mode
        if self.config.TASK == "occupancy":            
            for child in self.model.net.IFNet.children():                        
                if type(child) == nn.BatchNorm3d:
                    child.track_running_stats = False

        val_losses = []
        tic = getCurrentEpochTime()
        # print('Val length:', len(ValDataLoader))
        for i, data in enumerate(self.train_data_loader, 0):  # Get each batch
            
            net_input, target = self.model.preprocess(data, self.device)
            output = self.model.net(net_input)
            loss = self.objective(output, target)
            val_losses.append(loss.item())

            # Print stats
            toc = getCurrentEpochTime()
            elapsed = math.floor((toc - tic) * 1e-6)
            done = int(50 * (i+1) / len(self.val_data_loader))
            sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
                             .format('+' * done, '-' * (50 - done), np.mean(np.asarray(val_losses)), getTimeDur(elapsed)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        self.model.net.train()     #switch back to train mode
        
        return val_losses
