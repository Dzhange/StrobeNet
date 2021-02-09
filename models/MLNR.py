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
from models.loss import *

from pykdtree.kdtree import KDTree
# from eval.evaluator import chamfer_dis

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
        no_compute_item = ['mesh', 'iso_mesh']
        input_items = ['color00', 'grid_coords', 'translation', 'scale']
        target_items = ['nox00', 'pnnocs00', 'joint_map', 'linkseg', 'occupancies', 'pose']
        crr_items = ['crr-idx-mtx', 'crr-mask-mtx']

        inputs = {}
        targets = {}

        for k in data:            
            if k in no_compute_item:
                targets[k] = data[k][0] # data['mesh] = [('p1','p2')]
            else:                
                if k in crr_items:
                    # targets[k] = [[ii.float().permute(0, 2, 1).unsqueeze(3).to(device) for ii in item] for item in data[k]]
                    targets[k] = [[ii.permute(0, 2, 1).to(device) for ii in item] for item in data[k]]                    
                else:
                    ondevice_data = [item.to(device=device) for item in data[k]]
                    if k in input_items:
                        inputs[k] = ondevice_data
                    if k in target_items:
                        targets[k] = ondevice_data

        return inputs, targets


    def validate(self, val_dataloader, objective, device):

        # self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        self.output_dir = os.path.join(self.expt_dir_path, self.config.VAL_DIR)
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

        # self.setup_checkpoint(device)
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
        
        nocs_diff = [] # record the estimation error of nocs map
        masked_l2_loss = JiahuiL2Loss()
        
        nocs_accuracy = []
        nocs_completeness = []

        for i, data in enumerate(val_dataloader, 0):  # Get each batch            
            # if i < 319:
            #     continue

            if i > (num_samples-1): 
                break

            # first pass generate loss
            net_input, target = self.preprocess(data, device)
            output_recon = self.net(net_input)
            loss = objective(output_recon, target)
            epoch_losses.append(loss.item())

            if not self.config.STAGE_ONE:
                self.gen_mesh(grid_points_split, data, i)

            whole_pn_pc = []

            for view_id in range(self.view_num):
                segnet_output = output_recon[0]
                self.save_img(net_input['color00'][view_id], segnet_output[view_id][:, 0:4, :, :], target['nox00'][view_id][:, 0:4, :, :], i, view_id=view_id)

                if self.config.VAL_DIR != "ValResults":
                    continue

                self.gen_NOCS_pc(segnet_output[view_id][:, 0:3, :, :], target['nox00'][view_id][:, 0:3, :, :], \
                    target['nox00'][view_id][:, 3, :, :], "nocs", i)

                mask = target['nox00'][view_id][:, 3, :, :].cpu()
                pred_mask = segnet_output[view_id][:, 3, :, :].sigmoid().cpu()

                bone_num = self.bone_num

                if self.config.SKIN_LOSS or self.config.TASK == "lbs_seg":
                    tar_seg = target['linkseg'][view_id].squeeze(0)
                    self.save_mask(segnet_output[view_id], tar_seg, mask, i, view_id=view_id)
                if self.config.LOC_LOSS or self.config.LOC_MAP_LOSS:
                    tar_loc = target['pose'][view_id][0, :, 0:3]
                    cur_loc_diff = self.save_joint(segnet_output[view_id], tar_loc, mask, i, self.config.LOC_LOSS, view_id=view_id)
                    loc_diff.append(cur_loc_diff)

                    gt_joint_map = target["joint_map"][view_id][:,:bone_num*3]
                    self.visualize_joint_prediction(segnet_output[view_id], gt_joint_map, mask, i, view_id=view_id)

                if self.config.POSE_LOSS or self.config.POSE_MAP_LOSS:
                    tar_pose = target['pose'][view_id][0, :, 3:6]
                    cur_pose_diff = self.save_joint(segnet_output[view_id], tar_pose, mask, i, \
                        use_score=self.config.POSE_LOSS, loc=False, view_id=view_id)
                    pose_diff.append(cur_pose_diff)

                    gt_joint_map = target["joint_map"][view_id][:,bone_num*3:]
                    self.visualize_joint_prediction(segnet_output[view_id], gt_joint_map, mask, i, loc=False, view_id=view_id)

                self.visualize_confidence(segnet_output[view_id], i, view_id=view_id)

                if self.config.REPOSE:
                    pred_pnnocs = segnet_output[view_id][:, -3:, :, :]
                    tar_pnnocs = target['pnnocs00'][view_id]
                    self.save_single_img(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i, view_id=view_id)

                    tar_pose = target['pose'][view_id]
                    tar_nocs = target['nox00'][view_id][:, 0:3, :, :].squeeze()
                    tar_mask = target['nox00'][view_id][:, 3, :, :].squeeze()
                    tar_skin_seg = target['linkseg'][view_id]
                    # self.gt_debug(target, "reposed_debug", i)
                    self.gt_debug(tar_pose, tar_nocs, tar_mask, tar_skin_seg, "reposed_debug", i)
                    if self.config.TRANSFORM:
                        transform = {'translation': net_input['translation'],
                        'scale':net_input['scale']}
                    else:
                        transform = None
                                                            
                    pnnocs_error = masked_l2_loss(pred_pnnocs.cpu(), tar_pnnocs.cpu(), pred_mask).item()                    
                    nocs_diff.append(pnnocs_error)

                    pn_pc = self.gen_NOCS_pc(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i, transform =transform, view_id=view_id)
                    whole_pn_pc.append(pn_pc)
                    # print(pn_pc.shape)
            

            ####### calculate chamfer distance ######
            whole_pn_pc = np.concatenate(whole_pn_pc, axis=0)
            iso_mesh = trimesh.load(data['iso_mesh'][0][0], process=False)
            gt_pc = iso_mesh.vertices.astype('float32')
            # Completeness: how far are the points of the target point cloud
            # from thre predicted point cloud
            # print(gt_pc.dtype, whole_pn_pc.dtype)
            kdtree = KDTree(whole_pn_pc)
            complete, _ = kdtree.query(gt_pc)            
            nocs_completeness.append(complete.mean())

            # Accuracy: how far are th points of the predicted pointcloud
            # from the target pointcloud                        
            kdtree = KDTree(gt_pc)
            accuracy, _ = kdtree.query(whole_pn_pc)            
            nocs_accuracy.append(accuracy.mean())
            

            str_nocs_diff = "avg nocs diff is {:.6f} ".format(np.sqrt(np.mean(np.asarray(nocs_diff))))
            str_chamfer_diff = "compl/acc is {:.6f}/{:.6f} ".format(np.mean(np.asarray(nocs_completeness)), np.mean(np.asarray(nocs_accuracy)))
            str_joint_diff = "avg loc/angle diff is {:.6f}/{:.6f} ".format(np.mean(np.asarray(loc_diff)), np.degrees(np.mean(np.asarray(pose_diff))))
            # str_angle_diff = "avg diff is {:.6f} degree ".format(np.degrees(np.mean(np.asarray(pose_diff))))
            str_loss = "avg validation loss is {:6f}".format(np.mean(np.asarray(epoch_losses)))
            str_item_loss = "{}th data loss {:6f} ".format(i, loss.item())
            sys.stdout.write("\r[ VAL ]"  + str_nocs_diff + str_chamfer_diff + str_joint_diff)
            # sys.stdout.write("\r[ VAL ] {}th data ".format(i) + str_loss)
            sys.stdout.flush()
        
        print("\n")