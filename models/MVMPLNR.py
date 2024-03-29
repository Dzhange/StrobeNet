"""
This is the multi-view version of LNRNOCS
We use Multi-View SegNet for the first stage
and then use the network to predict it back.
"""
import os, sys, gc
import shutil
import glob
import torch
import torch.nn as nn
from models.networks.MVMPLNRNet import MVMPLNRNet
from models.LNR import ModelLNRNET
from models.SegLBS import ModelSegLBS
from utils.DataUtils import *
from utils.lbs import *
from utils.tools.voxels import VoxelGrid
import trimesh, mcubes

class ModelMVMPLNRNet(ModelLNRNET):

    def __init__(self, config):
        super().__init__(config)
        self.view_num = config.VIEW_NUM
    
    def init_net(self, device):        
        config = self.config
        self.net = MVMPLNRNet(config, device=device)
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
        no_compute_item = ['mesh', 'iso_mesh', 'cano_iso_mesh']

        input_items = ['color00', 'grid_coords', 'translation', 'scale', \
                        'cano_grid_coords', 'cano_translation', 'cano_scale']

        target_items = ['nox00', 'pnnocs00', 'joint_map', 'linkseg', 'pose',\
                        'occupancies', 'cano_occupancies']
        crr_items = ['crr-idx-mtx', 'crr-mask-mtx']

        inputs = {}
        targets = {}

        for k in data:
            # print(k)
            if k in no_compute_item:
                targets[k] = data[k][0] # data['mesh] = [('p1','p2')]
            else:
                if k in crr_items:
                    # targets[k] = [[ii.float().permute(0, 2, 1).unsqueeze(3).to(device) for ii in item] for item in data[k]]
                    targets[k] = [[ii.permute(0, 2, 1).to(device) for ii in item] for item in data[k]]                    
                else:
                    if isinstance(data[k],list):
                        ondevice_data = [item.to(device=device) for item in data[k]]
                    else:
                        ondevice_data = data[k].to(device=device)
                    if k in input_items:
                        inputs[k] = ondevice_data
                    if k in target_items:
                        targets[k] = ondevice_data
        return inputs, targets


    def validate(self, val_dataloader, objective, device):
        # pass
        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        
        if self.config.ANIM_MODEL:
            self.output_dir = os.path.join(self.expt_dir_path, "animation_frame_{}_start_{}_step{}".\
            format(self.config.ANIM_MODEL_ID, self.config.ANGLE_START, self.config.ANGLE_STEP))

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
            # if i < 190:
            #     continue
            # first pass generate loss
            net_input, target = self.preprocess(data, device)

            if self.config.ANIM_MODEL or self.config.FIX_HACK:                    
                angle = self.config.ANGLE_START + self.config.ANGLE_STEP * i
                output_recon = self.net(net_input, angle)          
            else:
                output_recon = self.net(net_input)


            if not self.config.VIEW_NUM > self.config.VIEW_RECON:                
                loss = objective(output_recon, target)
                epoch_losses.append(loss.item())

            if not self.config.STAGE_ONE:
                self.gen_mesh(grid_points_split, data, i)


            for view_id in range(self.view_num):

                segnet_output = output_recon[0]
                self.save_img(net_input['color00'][view_id], segnet_output[view_id][:, 0:4, :, :], target['nox00'][view_id][:, 0:4, :, :], i, view_id=view_id)

                # if self.config.VIEW_NUM > self.config.VIEW_RECON:
                #     continue

                self.gen_NOCS_pc(segnet_output[view_id][:, 0:3, :, :], target['nox00'][view_id][:, 0:3, :, :], \
                    target['nox00'][view_id][:, 3, :, :], "nocs", i)

                mask = target['nox00'][view_id][:, 3, :, :].cpu()
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

                    self.gen_NOCS_pc(pred_pnnocs, tar_pnnocs, mask, "reposed_pn", i, transform =transform, view_id=view_id)

                str_loc_diff = "avg loc diff is {:.6f} ".format(np.mean(np.asarray(loc_diff)))
                str_angle_diff = "avg diff is {:.6f} degree ".format(np.degrees(np.mean(np.asarray(pose_diff))))
                str_loss = "avg validation loss is {:6f}".format(np.mean(np.asarray(epoch_losses)))
                sys.stdout.write("\r[ VAL ] {}th data ".format(i) + str_loc_diff + str_angle_diff + str_loss)
                # sys.stdout.write("\r[ VAL ] {}th data ".format(i) + str_loss)
                sys.stdout.flush()
            print("\n")

    def gen_mesh(self, grid_points_split, data, frame_id):

        device = self.net.device
        cano_logits_list = []
        posed_logits_list = []
        for i, points in enumerate(grid_points_split):
            with torch.no_grad():
                net_input, target = self.preprocess(data, device)
                net_input['grid_coords'] = [points.to(device), ] * self.config.VIEW_NUM
                net_input['cano_grid_coords'] = points.to(device)                
                if self.config.ANIM_MODEL or self.config.FIX_HACK:                    
                    angle = self.config.ANGLE_START + self.config.ANGLE_STEP * frame_id
                    output = self.net(net_input, angle)
                else:
                    output = self.net(net_input)

                # cano
                cano_logits_list.append(output[1].squeeze(0).detach().cpu())

                # posed
                split_posed_logits = [occ.squeeze(0).detach().cpu() for occ in output[2]]
                posed_logits_list.append(split_posed_logits)

        # pos_fix = "off"
        pos_fix = "obj"

        for j in range(self.config.VIEW_NUM):
            if j >= self.config.VIEW_RECON:
                print("[ skipping view {} ]".format(j))
                continue

            mesh_logits = []
            for i in range(len(grid_points_split)):
                mesh_logits.append(posed_logits_list[i][j])
            
            logits = torch.cat(mesh_logits, dim=0).numpy()
            mx = logits.max()
            mn = logits.min()            
                                
            # posed_mesh = self.mesh_from_logits(logits, self.net.resolution)
            # posed_mesh = self.mesh_from_logits(logits, self.net.resolution, initThreshold=0.45)
            posed_mesh = self.mesh_from_logits(logits, self.net.resolution, initThreshold=0.5)

            if self.config.ANIM_MODEL or self.config.FIX_HACK:
                export_pred_path = os.path.join(self.output_dir, "frame_{}_angle_{}_recon.{}".format(str(frame_id).zfill(3), self.config.ANGLE_START + self.config.ANGLE_STEP * frame_id, pos_fix))
            else:
                export_pred_path = os.path.join(self.output_dir, "frame_{}_view_{}_recon.{}".format(str(frame_id).zfill(3), j, pos_fix))

            posed_mesh.export(export_pred_path)
            
            # print("gt is ", data['iso_mesh'][j])
            iso_orig = os.path.basename(str(data['iso_mesh'][j][0])).split('.')[0]

            export_gt_path = os.path.join(self.output_dir, "frame_{}_view_{}_gt.{}".format(str(frame_id).zfill(3), j, pos_fix))
            # print("gt is ", data['iso_mesh'][j][0])
            if "off" in pos_fix:                
                shutil.copyfile(str(data['iso_mesh'][j][0]), export_gt_path)
            else:
                gt_mesh = trimesh.load(str(data['iso_mesh'][j][0]))
                gt_mesh.export(export_gt_path)
            gc.collect()
            torch.cuda.empty_cache()
        
        if not self.config.ANIM_MODEL:
            # generate predicted mesh from occupancy and save
            logits = torch.cat(cano_logits_list, dim=0).numpy()
            mx = logits.max()
            mn = logits.min()
            mesh = self.mesh_from_logits(logits, self.net.resolution)
            export_pred_path = os.path.join(self.output_dir, "frame_{}_recon.{}".format(str(frame_id).zfill(3), pos_fix))        
            mesh.export(export_pred_path)

        # Copy ground truth in the val results
        export_gt_path = os.path.join(self.output_dir, "frame_{}_gt.{}".format(str(frame_id).zfill(3), pos_fix))
        if "off" in pos_fix:
            shutil.copyfile(target['cano_iso_mesh'], export_gt_path)
        else:
            gt_mesh = trimesh.load(target['cano_iso_mesh'])
            gt_mesh.export(export_gt_path)

        if self.config.TRANSFORM:
            # Get the transformation into val results
            export_trans_path = os.path.join(self.output_dir, "frame_{}_trans.npz".format(str(frame_id).zfill(3)))
            trans_path = target['iso_mesh'][0].replace("isosurf_scaled.off", "transform.npz")
            shutil.copyfile(trans_path, export_trans_path)