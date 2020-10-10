"""
This model is the baseline version
We directly take the input from the pnnocs
and feed it to the IFNet
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
import trimesh, mcubes


class ModelLNRNET(ModelSegLBS):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lr = config.LR
        device = torch.device(config.GPU)
        # self.net = NRNet(config, device=device)
        self.net = LNRNet(config, device=device)
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

        # generation conifgs
        self.resolution = 128
        self.batch_points = 100000
        
    @staticmethod
    def preprocess(data, device):
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
        inputs['RGB'] = data_to_device[0]
        inputs['grid_coords'] = data_to_device[3]['grid_coords']
        inputs['translation'] = data_to_device[3]['translation']
        inputs['scale'] = data_to_device[3]['scale']

        targets = {}
        targets['maps'] = data_to_device[1]
        targets['pose'] = data_to_device[2]
        targets['occupancies'] = data_to_device[3]['occupancies']
        targets['mesh'] = data[4]
        return inputs, targets

    def LoadSegNetCheckpoint(self, train_device, TargetPath):
        all_checkpoints = glob.glob(os.path.join(TargetPath, '*.tar'))
        if len(all_checkpoints) > 0:
            latest_checkpoint_dict = loadLatestPyTorchCheckpoint(TargetPath, map_location=train_device)
            print('[ INFO ]: Temp Use, Loading from pretrained model in {}'.format(TargetPath))

        if latest_checkpoint_dict is not None:
            # Make sure experiment names match
            self.net.SegNet.load_state_dict(latest_checkpoint_dict['ModelStateDict'])

    def validate(self, val_dataloader, objective, device):

        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)
        
        self.setup_checkpoint(device)
        self.net.eval()

        num_test_sample = 30

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
            output = self.net(net_input)
            loss = objective(output, target)
            epoch_losses.append(loss.item())
            
            # self.gen_mesh(grid_points_split, data, i)
            self.save_img(net_input['RGB'], output[:, 0:4, :, :], target['maps'][:, 0:4, :, :], i)
            if self.config.SKIN_LOSS or self.config.TASK == "lbs_seg":
                self.save_mask(output, target['maps'], i)
            if self.config.LOC_LOSS or self.config.LOC_MAP_LOSS:
                cur_loc_diff = self.save_joint(output, target, i, self.config.LOC_LOSS)
                loc_diff.append(cur_loc_diff)
                self.visualize_joint_prediction(output, target, i)
            if self.config.POSE_LOSS or self.config.POSE_MAP_LOSS:
                cur_pose_diff = self.save_joint(output, target, i, use_score=self.config.POSE_LOSS, loc=False)
                pose_diff.append(cur_pose_diff)
                self.visualize_joint_prediction(output, target, i, loc=False)
            

            if output.shape[1] > 64 + 4+self.config.BONE_NUM*8+2:
                pred_pnnocs = output[:, -3:, :, :]
                tar_pnnocs = target['maps'][:, -3:, :, :]
                self.save_single_img(pred_pnnocs, tar_pnnocs, "reposed_pn", i)

            ## save nocs point cloud
            # nocs_pc = self.gen_NOCS_pc(output[:, 0:3, :, :], output[:, 0:3, :, :])
            # pnnocs_pc = self.gen_NOCS_pc(output[:, -3:, :, :], output[:, 0:3, :, :])


            str_loc_diff = "avg loc diff is {:.6f}".format(  np.mean(np.asarray(loc_diff)))
            str_angle_diff = "avg diff is {:.6f} degree".format(  np.degrees(np.mean(np.asarray(pose_diff))))
            str_loss = "avg validation loss is {}".format( np.mean(np.asarray(epoch_losses)))
            sys.stdout.write("\r[ VAL ] " + str_loc_diff + str_angle_diff + str_loss) 
            sys.stdout.flush()

    def save_single_img(self, output, target, target_str, i):
        
        output = sendToDevice(output.detach(), 'cpu')
        target = sendToDevice(target, 'cpu')
        output_img = torch2np(output.squeeze()).squeeze() * 255
        target_img = torch2np(target.squeeze()).squeeze() * 255

        
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_00gt.png').format(str(i).zfill(3), target_str),
                    cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_01pred.png').format(str(i).zfill(3), target_str),
                    cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))


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
        mesh = self.mesh_from_logits(logits, self.resolution)
        export_pred_path = os.path.join(self.output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
        mesh.export(export_pred_path)

        # Copy ground truth in the val results
        export_gt_path = os.path.join(self.output_dir, "frame_{}_gt.off".format(str(i).zfill(3)))
        shutil.copyfile(target['mesh'][0], export_gt_path)

        # Get the transformation into val results
        export_trans_path = os.path.join(self.output_dir, "frame_{}_trans.npz".format(str(i).zfill(3)))
        trans_path = target['mesh'][0].replace("isosurf_scaled.off", "transform.npz")
        shutil.copyfile(trans_path, export_trans_path)

    # @staticmethod
    # def gen_NOCS_pc(nocs_map, mask):
        
    #     nocs_pc = nocs_map[i, :, index[0], index[1]].transpose(0, 1).unsqueeze(0)
    #     print(nocs_pc.shape)
    #     nocs_pc = nocs_pc.detach().cpu().numpy()
    #     print(nocs_pc.shape)
        

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