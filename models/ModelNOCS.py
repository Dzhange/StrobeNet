"""
The model to train, validate and test NOCS
"""
import os
import glob
import torch
from models.SegNet import SegNet as old_SegNet
from models.say4n_SegNet import SegNet as new_SegNet
from utils.DataUtils import *

class model_NOCS(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR # set learning rate        
        # self.net = old_SegNet(out_channels=4, withSkipConnections=False)
        self.net = new_SegNet(input_channels=3, output_channels=4)
        self.loss_history = []
        self.val_loss_history = []
        self.start_epoch = 0
        self.expt_dir_path = os.path.join(expandTilde(self.config.OUTPUT_DIR), self.config.EXPT_NAME)
        # print(self.expt_dir_path)
        if os.path.exists(self.expt_dir_path) == False:            
            os.makedirs(self.expt_dir_path)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                          betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))
        
    # def load_check_point(self):
    def setup_checkpoint(self, TrainDevice):
        latest_checkpoint_dict = None
        all_checkpoints = glob.glob(os.path.join(self.expt_dir_path, '*.tar'))
        if len(all_checkpoints) > 0:
            latest_checkpoint_dict = loadLatestPyTorchCheckpoint(self.expt_dir_path, map_location=TrainDevice)
            print('[ INFO ]: Loading from last checkpoint.')

        if latest_checkpoint_dict is not None:
            # Make sure experiment names match
            if self.config.EXPT_NAME == latest_checkpoint_dict['Name']:
                self.net.load_state_dict(latest_checkpoint_dict['ModelStateDict'])
                self.start_epoch = latest_checkpoint_dict['Epoch']
                self.optimizer.load_state_dict(latest_checkpoint_dict['OptimizerStateDict'])
                self.loss_history = latest_checkpoint_dict['LossHistory']
                if 'ValLossHistory' in latest_checkpoint_dict:
                    self.val_loss_history = latest_checkpoint_dict['ValLossHistory']
                else:
                    self.val_loss_history = self.loss_history

                # Move optimizer state to GPU if needed. See https://github.com/pytorch/pytorch/issues/2830
                if TrainDevice is not 'cpu':
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(TrainDevice)
            else:
                print('[ INFO ]: Experiment names do not match. Training from scratch.')

    # def save_check_point(self):
    def save_checkpoint(self, epoch, time_string='humanlocal', print_str='*'*3):
        CheckpointDict = {
            'Name': self.config.EXPT_NAME,
            'ModelStateDict': self.net.state_dict(),
            'OptimizerStateDict': self.optimizer.state_dict(),
            'LossHistory': self.loss_history,
            'ValLossHistory': self.val_loss_history,
            'Epoch': self.start_epoch + epoch + 1,
            'SavedTimeZ': getZuluTimeString(),
        }
        OutFilePath = savePyTorchCheckpoint(CheckpointDict, self.expt_dir_path, TimeString=time_string)
        saveLossesCurve(self.loss_history, self.val_loss_history, out_path=os.path.splitext(OutFilePath)[0] + '.png',
                                xlim=[0, int(self.config.EPOCH_TOTAL + self.start_epoch)],
                                legend=["train loss","val loss"], title=self.config.EXPT_NAME)
        # print('[ INFO ]: Checkpoint saved.')
        print(print_str) # Checkpoint saved. 50 + 3 characters [>]

    def preprocess(self, Data, Device):
        DataTD = []
        for item in Data:
            TupleOrTensor = item
            TupleOrTensorTD = item
            if isinstance(TupleOrTensorTD, tuple) == False and isinstance(TupleOrTensorTD, list) == False:
                TupleOrTensorTD = TupleOrTensor.to(Device)
            else:
                for Ctr in range(len(TupleOrTensor)):
                    if isinstance(TupleOrTensor[Ctr], torch.Tensor):
                        TupleOrTensorTD[Ctr] = TupleOrTensor[Ctr].to(Device)
                    else:
                        TupleOrTensorTD[Ctr] = TupleOrTensor[Ctr]

            DataTD.append(TupleOrTensorTD)
        return DataTD


    def train_step(self):
        pass

    def val_step(self):
        pass
    

