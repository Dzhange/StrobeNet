"""
This model is the FIRST stage of out proposed LBS pipeline
Here we predict the skinning weights, the pose and the confident score
"""
import os, sys, shutil
import glob
import torch
import torch.nn as nn
import trimesh
from models.networks.OccNet import OccupancyNetwork

from models.loss import LBSLoss

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))

# from im2mesh.onet.models import OccupancyNetwork

from utils.DataUtils import *
import time
from torch import distributions as dist



# def get_prior_z(cfg, device):
#     ''' Returns prior distribution for latent code z.

#     Args:
#         cfg (dict): imported yaml config
#         device (device): pytorch device
#     '''
#     z_dim = 64
#     p0_z = dist.Normal(
#         torch.zeros(z_dim, device=device),
#         torch.ones(z_dim, device=device)
#     )

#     return p0_z


    
class ModelONet(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR # set learning rate

        self.loss_history = []
        self.val_loss_history = []
        self.start_epoch = 0
        self.expt_dir_path = os.path.join(expandTilde(self.config.OUTPUT_DIR), self.config.EXPT_NAME)
        if os.path.exists(self.expt_dir_path) == False:
            os.makedirs(self.expt_dir_path)

        device = torch.device(config.GPU)
        self.init_net(device=device)
        
        self.device = device
    
    def init_net(self, device=None):        
        config = self.config        
        
        self.net = OccupancyNetwork(device=device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                        betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
                                        weight_decay=config.WEIGHT_DECAY
                                        )

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
        
        inputs = data[0].to(device=device)
        # print(data[1])
        points = data[1]['grid_coords'].to(device=device)
        occ = data[1]['occupancies'].to(device=device)

        data = {}
        data['inputs'] = inputs
        data['points'] = points
        data['occupancies'] = occ

        return data

    def train_step(self, data):
        
        self.net.train()
        data = self.preprocess(data, self.device)
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        
        return loss


    def compute_loss(self, data):

        device = self.device
        p = data.get('points').to(device)
        occ = data.get('occupancies').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        
        c = self.net.encode_inputs(inputs)
        q_z = self.net.infer_z(p, occ, c)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.net.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.net.decode(p, z, c).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()
        
        return loss
        