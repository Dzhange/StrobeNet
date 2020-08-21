"""
This model is the baseline version
We directly take the input from the pnnocs
and feed it to the IFNet
"""

import os
import shutil
import glob
import torch
import torch.nn as nn
from models.networks.SegNet import SegNet as old_SegNet
from models.networks.say4n_SegNet import SegNet as new_SegNet
from models.networks.NRNet import NRNet
from models.loss import MixLoss
from utils.DataUtils import *
import trimesh, mcubes


class ModelIFNOCS(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR
        device = torch.device(config.GPU)
        self.net = NRNet(config, device=device)
        self.objective = MixLoss()        
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
            # if train_device is not 'cpu':
            #     for state in self.optimizer.state.values():
            #         for k, v in state.items():
            #             if isinstance(v, torch.Tensor):
            #                 state[k] = v.to(train_device)

    def validate(self, val_dataloader, device):

        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)
        
        self.setup_checkpoint(device)
        self.net.eval()

        num_test_sample = 30
        resolution = 128
        batch_points = 100000

        # Make sure we are not tracking running stats
        # Otherwise it perform bad           
        for child in self.net.IFNet.children():
            if type(child) == nn.BatchNorm3d:
                child.track_running_stats = False            

        test_net = self.net
        num_samples = min(num_test_sample, len(val_dataloader))
        print('Testing on ' + str(num_samples) + ' samples')

        grid_coords = test_net.initGrids(resolution)
        grid_points_split = torch.split(grid_coords, batch_points, dim=1)
        for i, data in enumerate(val_dataloader, 0):  # Get each batch
            if i > (num_samples-1): break

            # first pass generate loss
            net_input, target = self.preprocess(data, device)
            output = self.net(net_input)
            loss = self.objective(output, target)
            
            logits_list = []
            for points in grid_points_split:
                with torch.no_grad():
                    net_input, target = self.preprocess(data, device)                    
                    net_input['grid_coords'] = points.to(device)
                    output = test_net(net_input)
                    self.save_img(net_input['RGB'], output[0], target['NOCS'], i)
                    logits_list.append(output[1].squeeze(0).detach().cpu())
            
            # generate predicted mesh from occupancy and save
            logits = torch.cat(logits_list, dim=0).numpy()
            mesh = self.mesh_from_logits(logits, resolution)
            export_pred_path = os.path.join(self.output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
            mesh.export(export_pred_path)

            # Copy ground truth in the val results
            export_gt_path = os.path.join(self.output_dir, "frame_{}_gt.off".format(str(i).zfill(3)))
            shutil.copyfile(target['mesh'][0], export_gt_path)

            # Get the transformation into val results
            export_trans_path = os.path.join(self.output_dir, "frame_{}_trans.npz".format(str(i).zfill(3)))
            trans_path = target['mesh'][0].replace("isosurf_scaled.off", "transform.npz")
            shutil.copyfile(trans_path, export_trans_path)

    def save_img(self, net_input, output, target, i):

        input_img, gt_out_tuple_rgb, gt_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(target, 'cpu'))
        _, pred_out_tuple_rgb, pred_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(output.detach(), 'cpu'), isMaskNOX=True)
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_color00.png').format(str(i).zfill(3)),  cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        out_target_str = [self.config.TARGETS]

        for target_str, gt, pred, gt_mask, pred_mask in zip(out_target_str, gt_out_tuple_rgb, pred_out_tuple_rgb, gt_out_tuple_mask, pred_out_tuple_mask):
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_00gt.png').format(str(i).zfill(3), target_str),
                        cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_01pred.png').format(str(i).zfill(3), target_str),
                        cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_02gtmask.png').format(str(i).zfill(3), target_str),
                        gt_mask)
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_03predmask.png').format(str(i).zfill(3), target_str),
                        pred_mask)

    @staticmethod
    def mesh_from_logits(logits, resolution):
        logits = np.reshape(logits, (resolution,) * 3)
        initThreshold = 0.5
        pmax = 0.5
        pmin = -0.5
        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(initThreshold) - np.log(1. - initThreshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)
        # remove translation due to padding
        vertices -= 1
        # rescale to original scale
        step = (pmax - pmin) / (resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [pmin, pmin, pmin]
        return trimesh.Trimesh(vertices, triangles)