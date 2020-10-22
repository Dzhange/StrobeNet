"""
This is the multi-view version of LNRNOCS
We use Multi-View SegNet for the first stage
and then use the network to predict it back.
"""
import os, sys
import shutil
import glob
import torch
import torch.nn as nn
from models.networks.MLNRNet import MLNRNet
from models.LNR import ModelLNRNET
from models.SegLBS import ModelSegLBS
from utils.DataUtils import *
from utils.lbs import *
import trimesh, mcubes

class ModelMLNRNet(ModelLNRNET):

    def __init__(self, config):
        super().__init__(config)
        self.view_num = config.VIEW_NUM
    
    def init_net(self, device):        
        config = self.config
        self.net = MLNRNet(config, device=device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                          betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))        
        if config.NRNET_PRETRAIN:
            if config.NRNET_PRETRAIN_PATH.endswith('tar'):
                self.LoadSegNetFromTar(device, config.NRNET_PRETRAIN_PATH)
            else:
                pretrained_dir = config.NRNET_PRETRAIN_PATH
                self.LoadSegNetCheckpoint(device, pretrained_dir)

    def preprocess(self, data, device):
        """
        put data onto the right device
        'color00', 'nox00', 'linkseg', 'pnnocs00'
        data input(is a dict):
            keys include:
            # for segnet
            1. color00
            2. linkseg
            3. nox00
            4. pnnocs00
            5. joint map
            6. pose

            # for if-net
            7 grid_coords
            8. occupancies
            9. translation
            10. scale
        """
        no_compute_item = ['mesh', 'iso_mesh', 'crr-idx-mtx', 'crr-mask-mtx']
        input_items = ['color00', 'grid_coords', 'translation', 'scale']
        target_items = ['nox00', 'pnnocs00', 'joint_map', 'linkseg', 'occupancies', 'pose']
        
        inputs = {}
        targets = {}

        for k in data:
            # print(k)
            if k in no_compute_item:
                targets[k] = data[k][0] # data['mesh] = [('p1','p2')]
            else:
                ondevice_data = [item.to(device=device) for item in data[k]]
                if k in input_items:
                    inputs[k] = ondevice_data
                if k in target_items:
                    targets[k] = ondevice_data
        return inputs, targets


    def validate(self, val_dataloader, objective, device):

        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

        self.setup_checkpoint(device)
        self.net.eval()

        num_test_sample = 30000

        grid_coords = self.net.grid_coords
        grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)

        # Make sure we are not tracking running stats
        # Otherwise it perform bad
        for child in self.net.IFNet.children():
            if type(child) == nn.BatchNorm3d:
                child.track_running_stats = False

        num_samples = min(num_test_sample, len(val_dataloader))
        print('Testing on ' + str(num_samples) + ' samples')

        epoch_losses = []
        pose_diff = []
        loc_diff = []

        for i, data in enumerate(val_dataloader, 0):  # Get each batch
            if i > (num_samples-1): 
                break

            # first pass generate loss
            net_input, target = self.preprocess(data, device)
            output_recon = self.net(net_input)
            loss = objective(output_recon, target)
            epoch_losses.append(loss.item())


            if not self.config.STAGE_ONE:
                self.gen_mesh(grid_points_split, data, i)

            for view_id in range(self.view_num):

                segnet_output = output_recon[0]
                self.save_img(net_input['color00'][view_id], segnet_output[view_id][:, 0:4, :, :], target['nox00'][view_id][:, 0:4, :, :], i)

                self.gen_NOCS_pc(segnet_output[view_id][:, 0:3, :, :], target['nox00'][view_id][:, 0:3, :, :], \
                    target['nox00'][view_id][:, 3, :, :], "nocs", i)

                # if self.config.SKIN_LOSS or self.config.TASK == "lbs_seg":
                #     self.save_mask(segnet_output, target['maps'], i)
                # if self.config.LOC_LOSS or self.config.LOC_MAP_LOSS:
                #     cur_loc_diff = self.save_joint(segnet_output, target, i, self.config.LOC_LOSS)
                #     loc_diff.append(cur_loc_diff)
                #     self.visualize_joint_prediction(segnet_output, target, i)
                # if self.config.POSE_LOSS or self.config.POSE_MAP_LOSS:
                #     cur_pose_diff = self.save_joint(segnet_output, target, i, use_score=self.config.POSE_LOSS, loc=False)
                #     pose_diff.append(cur_pose_diff)
                #     self.visualize_joint_prediction(segnet_output, target, i, loc=False)

                # self.visualize_confidence(segnet_output, i)

                # if segnet_output.shape[1] > 64 + 4+self.config.BONE_NUM*8+2:
                #     pred_pnnocs = segnet_output[:, -3:, :, :]
                #     mask = target['maps'][:, 3, :, :]
                #     tar_pnnocs = target['maps'][:, -3:, :, :]
                    
                #     self.save_single_img(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i)

                #     self.gt_debug(target, "reposed_debug", i)
                #     if self.config.TRANSFORM:
                #         transform = {'translation': net_input['translation'],
                #         'scale':net_input['scale']}
                #     else:
                #         transform = None

                #     self.gen_NOCS_pc(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i, transform =transform)

                # str_loc_diff = "avg loc diff is {:.6f} ".format(np.mean(np.asarray(loc_diff)))
                # str_angle_diff = "avg diff is {:.6f} degree ".format(np.degrees(np.mean(np.asarray(pose_diff))))
                str_loss = "avg validation loss is {:6f}".format(np.mean(np.asarray(epoch_losses)))
                # sys.stdout.write("\r[ VAL ] {}th data ".format(i) + str_loc_diff + str_angle_diff + str_loss)
                sys.stdout.write("\r[ VAL ] {}th data ".format(i) + str_loss)
                sys.stdout.flush()
            print("\n")