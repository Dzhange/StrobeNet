"""
This model is the Linear blend skinning version
We first predict NOCS map
then predict pose and joint location, along with segmentation map
We then pose the NOCS back into PNNOCS
We feed the features with PNNOCS, then use IF-Net for final reconstruction
"""

import os, sys
import shutil
import glob
import torch
import torch.nn as nn
from models.networks.LNRNet import LNRNet
from models.loss import MixLoss
# from models.LBS import ModelLBSNOCS
from models.SegLBS import ModelSegLBS
from utils.DataUtils import *
from utils.lbs import *
import trimesh, mcubes
from collections import OrderedDict

class ModelLNRNET(ModelSegLBS):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lr = config.LR
        # device = torch.device(config.GPU)
        # self.init_net(device)        
        
        self.loss_history = []
        self.val_loss_history = []
        self.start_epoch = 0
        self.expt_dir_path = os.path.join(expandTilde(self.config.OUTPUT_DIR), self.config.EXPT_NAME)        
        if os.path.exists(self.expt_dir_path) == False:
            os.makedirs(self.expt_dir_path)
                
        # generation conifgs        
        self.batch_points = 100000

    def init_net(self, device):
        config = self.config
        self.net = LNRNet(config, device=device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                          betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))
        if config.NRNET_PRETRAIN:
            if config.NRNET_PRETRAIN_PATH.endswith('tar'):
                self.LoadSegNetFromTar(device, config.NRNET_PRETRAIN_PATH)
            else:
                pretrained_dir = config.NRNET_PRETRAIN_PATH
                self.LoadSegNetCheckpoint(device, pretrained_dir)


    # @staticmethod
    def preprocess(self, data, device):
        """
        put data onto the right device
        data input:
            0: color image
            1: loaded maps for supervision:
                1) segmap
                2) joint map
            2: Pose in np format
            3: occupancies
            4: GT mesh path
        """
        data_to_device = []
        for item in data:
            tuple_or_tensor = item
            tuple_or_tensor_td = item
            if isinstance(tuple_or_tensor_td, list):
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

        inputs = data_to_device[0]
        # TODO: uncomment these when pipeline finished
        inputs = {}
        inputs['color00'] = data_to_device[0]
        inputs['grid_coords'] = data_to_device[3]['grid_coords']
        if self.config.TRANSFORM:
            inputs['translation'] = data_to_device[3]['translation']
            inputs['scale'] = data_to_device[3]['scale']

        targets = {}
        targets['maps'] = data_to_device[1]
        targets['pose'] = data_to_device[2]
        targets['occupancies'] = data_to_device[3]['occupancies']
        # targets['mesh'] = data[4]
        targets['iso_mesh'] = data[3]["iso_mesh"]
        
        return inputs, targets

    def LoadSegNetCheckpoint(self, train_device, TargetPath):
        all_checkpoints = glob.glob(os.path.join(TargetPath, '*.tar'))
        if len(all_checkpoints) > 0:
            latest_checkpoint_dict = loadLatestPyTorchCheckpoint(TargetPath, map_location=train_device)
            print('[ INFO ]: Temp Use, Loading from pretrained model in {}'.format(TargetPath))

        if latest_checkpoint_dict is not None:
            # Make sure experiment names match
            self.net.load_state_dict(latest_checkpoint_dict['ModelStateDict'])
    
    def LoadSegNetFromTar(self, train_device, TargetPath):
        check_point_dict = loadPyTorchCheckpoint(TargetPath, train_device)
        if check_point_dict is not None:
            # Make sure experiment names match
            keys = check_point_dict['ModelStateDict'].keys()
            # model_dict = check_point_dict['ModelStateDict'].copy()
            model_dict = OrderedDict()
            
            for k in keys:
                if "IFNet" in k:
                    continue
                elif "SegNet" in k:
                    new_k = k.replace("SegNet.", "")
                    model_dict[new_k] = check_point_dict['ModelStateDict'][k]            
            print('[ INFO ]: Temp Use, Loading from pretrained model in {}'.format(TargetPath))
            self.net.SegNet.load_state_dict(model_dict)

    def validate(self, val_dataloader, objective, device):

        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

        self.setup_checkpoint(device)
        self.net.eval()

        num_test_sample = 30000

        grid_coords = self.net.grid_coords

        grid_coords = self.net.init_grids(256)
        # print(grid_coords.shape)
        # grid_path = os.path.join(self.output_dir, 'grid_coords.xyz')
        # write_off(grid_path, grid_coords[0].cpu().numpy())
        
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


            segnet_output = output_recon[0]
            self.save_img(net_input['color00'], segnet_output[:, 0:4, :, :], target['maps'][:, 0:4, :, :], i)
            self.gen_NOCS_pc(segnet_output[:, 0:3, :, :], target['maps'][:, 0:3, :, :], target['maps'][:, 3, :, :], "nocs", i)
            
            mask = target['maps'][:, 3, :, :].cpu().detach()
            bone_num = self.bone_num
            if self.config.SKIN_LOSS or self.config.TASK == "lbs_seg":
                tar_seg = target['maps'][:, 4+bone_num*6, :, :].cpu()                
                self.save_mask(segnet_output, tar_seg, mask, i)
            if self.config.LOC_LOSS or self.config.LOC_MAP_LOSS:
                tar_loc = target['pose'][0, :, 0:3]
                cur_loc_diff = self.save_joint(segnet_output, tar_loc, mask, i, self.config.LOC_LOSS)
                loc_diff.append(cur_loc_diff)
                
                gt_joint_map = target['maps'][:, 4:4+bone_num*3, :, :].cpu().detach()
                self.visualize_joint_prediction(segnet_output, gt_joint_map, mask, i)
            
            if self.config.POSE_LOSS or self.config.POSE_MAP_LOSS:
                tar_pose = target['pose'][0, :, 3:6]
                cur_pose_diff = self.save_joint(segnet_output, tar_pose, mask, i, use_score=self.config.POSE_LOSS, loc=False)
                pose_diff.append(cur_pose_diff)
                
                gt_joint_map = target['maps'][:, 4+bone_num*3:4+bone_num*6, :, :].cpu().detach()                
                self.visualize_joint_prediction(segnet_output, gt_joint_map, mask, i, loc=False)

            self.visualize_confidence(segnet_output, i)

            if self.config.REPOSE:
                pred_pnnocs = segnet_output[:, -3:, :, :]                
                tar_pnnocs = target['maps'][:, -3:, :, :]                
                self.save_single_img(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i)


                tar_pose = target['pose']        
                tar_nocs = target['maps'][:, 0:3, :, :].squeeze() # nocs        
                tar_mask = target['maps'][:, 3, :, :].squeeze() # mask
                tar_skin_seg = target['maps'][:, 4+self.bone_num*6:4+self.bone_num*6+1, :, :]

                self.gt_debug(tar_pose, tar_nocs, tar_mask, tar_skin_seg, "reposed_debug", i)
                if self.config.TRANSFORM:
                    transform = {'translation': net_input['translation'],
                     'scale':net_input['scale']}
                else:
                    transform = None

                self.gen_NOCS_pc(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i, transform =transform)

            str_loc_diff = "avg loc diff is {:.6f} ".format(np.mean(np.asarray(loc_diff)))
            str_angle_diff = "avg diff is {:.6f} degree ".format(np.degrees(np.mean(np.asarray(pose_diff))))
            str_loss = "avg validation loss is {:6f}".format(np.mean(np.asarray(epoch_losses)))
            sys.stdout.write("\r[ VAL ] {}th data ".format(i) + str_loc_diff + str_angle_diff + str_loss)
            sys.stdout.flush()
        print("\n")

    def save_single_img(self, output, target, mask, target_str, i, view_id=0):
        mask = sendToDevice(mask.detach(), 'cpu')
        output = sendToDevice(output.detach(), 'cpu')
        target = sendToDevice(target, 'cpu')

        # We use gt mask
        mask_prob = torch2np(mask.squeeze())
        output = torch2np(torch.squeeze(output)) * 255
        target = torch2np(torch.squeeze(target)) * 255
        target[mask_prob <= 0.7] = 255
        
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_view_{}_{}_00gt.png').format(str(i).zfill(3), view_id, target_str),
                    cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_view_{}_{}_01pred.png').format(str(i).zfill(3), view_id, target_str),
                    cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    def gen_mesh(self, grid_points_split, data, i):
        
        device = self.net.device
        logits_list = []
        for points in grid_points_split:
            with torch.no_grad():
                net_input, target = self.preprocess(data, device)
                net_input['grid_coords'] = points.to(device)
                output = self.net(net_input)
                # self.save_img(net_input['RGB'], output[0], target['NOCS'], i)
                logits_list.append(output[1].squeeze(0).detach().cpu())

        # generate predicted mesh from occupancy and save
        logits = torch.cat(logits_list, dim=0).numpy()
        mesh = self.mesh_from_logits(logits, self.net.resolution)
        export_pred_path = os.path.join(self.output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
        mesh.export(export_pred_path)

        # Copy ground truth in the val results
        export_gt_path = os.path.join(self.output_dir, "frame_{}_gt.off".format(str(i).zfill(3)))
        # print(target['mesh'][0])
        shutil.copyfile(target['iso_mesh'][0], export_gt_path)

        if self.config.TRANSFORM:
            # Get the transformation into val results
            export_trans_path = os.path.join(self.output_dir, "frame_{}_trans.npz".format(str(i).zfill(3)))
            trans_path = target['iso_mesh'][0].replace("isosurf_scaled.off", "transform.npz")
            shutil.copyfile(trans_path, export_trans_path)

    def gen_NOCS_pc(self, pred_nocs_map, tar_nocs_map, mask, target_str, i, view_id=0, transform=None):
                
        mask = sendToDevice(mask.detach(), 'cpu')
        pred_nocs_map = sendToDevice(pred_nocs_map.detach(), 'cpu')
        tar_nocs_map = sendToDevice(tar_nocs_map.detach(), 'cpu')
        
        # We use gt mask
        mask_prob = torch2np(mask.squeeze())
        pred_nocs_map = torch2np(torch.squeeze(pred_nocs_map))
        tar_nocs_map = torch2np(torch.squeeze(tar_nocs_map))

        pred_nocs = pred_nocs_map[mask_prob > 0.7]
        tar_nocs = tar_nocs_map[mask_prob > 0.7]

        tar_nocs_path = os.path.join(self.output_dir, 'frame_{}_view_{}_{}_00gt.xyz').format(str(i).zfill(3), view_id, target_str)
        pred_nocs_path = os.path.join(self.output_dir, 'frame_{}_view_{}_{}_01pred.xyz').format(str(i).zfill(3), view_id, target_str)
        self.write(tar_nocs_path, tar_nocs)
        self.write(pred_nocs_path, pred_nocs)


        if transform is not None:
            # print("HERE")
            pred_nocs +=  transform['translation'][0].detach().cpu().numpy()
            pred_nocs *= transform['scale'][0].detach().cpu().numpy()
            tar_nocs +=  transform['translation'][0].detach().cpu().numpy()
            tar_nocs *= transform['scale'][0].detach().cpu().numpy()

            tar_nocs_path = os.path.join(self.output_dir, 'frame_{}_view_{}_{}_trs_00gt.xyz').format(str(i).zfill(3), view_id, target_str)
            pred_nocs_path = os.path.join(self.output_dir, 'frame_{}_view_{}_{}_trs_01pred.xyz').format(str(i).zfill(3), view_id, target_str)
            self.write(tar_nocs_path, tar_nocs)
            self.write(pred_nocs_path, pred_nocs)
        
    # @staticmethod
    def mesh_from_logits(self, logits, resolution):
        logits = np.reshape(logits, (resolution,) * 3)
        initThreshold = 0.5
        if self.config.TRANSFORM:
            pmax = 0.5
            pmin = -0.5
        else:
            pmax = 1
            pmin = 0
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

    def gt_debug(self, tar_pose, tar_nocs, tar_mask, tar_skin_seg, target_str, i):
        # tar_pose = target['pose']
        
        # tar_nocs = target['maps'][:, 0:3, :, :].squeeze() # nocs        
        # tar_mask = target['maps'][:, 3, :, :].squeeze() # mask
        tar_loc = tar_pose[:, :, 0:3].squeeze(0)
        tar_rot = tar_pose[:, :, 3:6].squeeze(0)
        # tar_loc[:, 1] = 0
        # tar_skin_seg = target['maps'][:, 4+self.bone_num*6:4+self.bone_num*6+1, :, :]
        tar_skin_seg = self.label2map(tar_skin_seg.squeeze(0))
        # print("HEREHERE")
        pnnocs_pc, _ = self.net.repose_pm_core(tar_nocs, tar_loc, tar_rot, tar_skin_seg, tar_mask, self.bone_num)
        # pnnocs_pc, _ = self.net.repose_pm_fast(tar_nocs, tar_loc, tar_rot, tar_skin_seg, tar_mask, self.bone_num, True)
    
        pn_path = os.path.join(self.output_dir, 'frame_{}_{}_00gt.xyz').format(str(i).zfill(3), target_str)
        nox_path = os.path.join(self.output_dir, 'frame_{}_{}_01orig.xyz').format(str(i).zfill(3), target_str)

        nocs_pc = tar_nocs[:,  tar_mask > 0.7].transpose(0, 1)

        write_off(pn_path, pnnocs_pc[0])
        # write_off(pn_path, pnnocs_pc)
        write_off(nox_path, nocs_pc)

    def visualize_confidence(self, output, frame_id, view_id=0):
        
        # bone_num = self.bone_num
        # H, W        
        conf_maps = output[:, self.net.skin_end:self.net.conf_end, :, :].sigmoid().cpu().detach().squeeze().numpy()                
        for i in range(self.bone_num):
            conf_map = conf_maps[i] # squeezed
            scale = conf_map.max() - conf_map.min()
            # print(conf_map.max(), conf_map.min())
            if scale != 0:
                conf_map /= scale
            conf_map *= 255            
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_view_{}_conf.png').format(str(frame_id).zfill(3), view_id), conf_map)
