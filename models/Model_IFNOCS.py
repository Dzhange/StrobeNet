"""
The model to train, validate and test NOCS
"""
import os
import glob
import torch
from models.networks.SegNet import SegNet as old_SegNet
from models.networks.say4n_SegNet import SegNet as new_SegNet
from models.networks.NRNet import NRNet
from utils.DataUtils import *

class model_IFNOCS(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR
        device = torch.device(config.GPU)
        
        self.net = NRNet(config, device=device)
        self.loss_history = []
        self.val_loss_history = []
        self.start_epoch = 0
        self.expt_dir_path = os.path.join(expandTilde(self.config.OUTPUT_DIR), self.config.EXPT_NAME)        
        if os.path.exists(self.expt_dir_path) == False:            
            os.makedirs(self.expt_dir_path)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                          betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))
        
        if config.NRNET_PRETRAIN:
            pretrained_dir = config.NRNET_PRETRAIN_PATH
            self.LoadSegNetCheckpoint(device, pretrained_dir)

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

                if TrainDevice is not 'cpu':
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(TrainDevice)
            else:
                print('[ INFO ]: Experiment names do not match. Training from scratch.')

    # def save_check_point(self):
    def save_checkpoint(self, epoch, time_string='humanlocal', print_str='*'*3):
        checkpoint_dict = {
            'Name': self.config.EXPT_NAME,
            'ModelStateDict': self.net.state_dict(),
            'OptimizerStateDict': self.optimizer.state_dict(),
            'LossHistory': self.loss_history,
            'ValLossHistory': self.val_loss_history,
            'Epoch': self.start_epoch + epoch + 1,
            'SavedTimeZ': getZuluTimeString(),
        }
        out_file_path = savePyTorchCheckpoint(checkpoint_dict, self.expt_dir_path, TimeString=time_string)
        saveLossesCurve(self.loss_history, self.val_loss_history, out_path=os.path.splitext(out_file_path)[0] + '.png',
                        xlim=[0, int(self.config.EPOCH_TOTAL + self.start_epoch)],
                        legend=["train loss", "val loss"], title=self.config.EXPT_NAME)
        # print('[ INFO ]: Checkpoint saved.')
        print(print_str) # Checkpoint saved. 50 + 3 characters [>]

    @staticmethod
    def preprocess(data, device):
        """
        put data onto the right device
        """
        data_to_device = []
        for item in data:
            tuple_or_tensor = item
            tuple_or_tensor_td = item
            if isinstance(tuple_or_tensor_td, (tuple, list)):
                for ctr in range(len(tuple_or_tensor)):
                    if isinstance(tuple_or_tensor[ctr], torch.Tensor):
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr].to(device)
                    else:
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr]
                data_to_device.append(tuple_or_tensor_td)
            elif isinstance(tuple_or_tensor_td, (dict)):
                dict_td = {}
                keys = item.keys()
                for key in keys:
                    if isinstance(item[key], torch.Tensor):
                        dict_td[key] = item[key].to(device)
                data_to_device.append(dict_td)
            elif isinstance(tuple_or_tensor, torch.Tensor):
                tuple_or_tensor_td = tuple_or_tensor.to(device)
                data_to_device.append(tuple_or_tensor_td)
            else:
                # for gt mesh
                continue
        
        # print("len is ", len(data_to_device))
        # data_to_device = [COLOR_TENSOR,[TARGET_NOCS_TENSOR,OCCUPANCY{}]]
        # print(type(data_to_device[0]))
        # print(type(data_to_device[1][0]), type(data_to_device[1][1]))
        # inputs = data_to_device[1][0]
        inputs = {}
        inputs['RGB'] = data_to_device[0]
        inputs['grid_coords'] = data_to_device[2]['grid_coords']
        inputs['translation'] = data_to_device[2]['translation']
        inputs['scale'] = data_to_device[2]['scale']

        targets = {}
        targets['NOCS'] = data_to_device[1]
        targets['occupancies'] = data_to_device[2]['occupancies']
        targets['mesh'] = data[2]['mesh']

        return inputs, targets

    def LoadSegNetCheckpoint(self, train_device, TargetPath):
        all_checkpoints = glob.glob(os.path.join(TargetPath, '*.tar'))
        if len(all_checkpoints) > 0:
            latest_checkpoint_dict = loadLatestPyTorchCheckpoint(TargetPath, map_location=train_device)
            print('[ INFO ]: Temp Use, Loading from pretrained model in {}'.format(TargetPath))

        if latest_checkpoint_dict is not None:
            # Make sure experiment names match
            self.net.SegNet.load_state_dict(latest_checkpoint_dict['ModelStateDict'])            
            if train_device is not 'cpu':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(train_device)
