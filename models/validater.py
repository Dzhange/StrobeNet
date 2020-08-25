"""
Generate some validation results for visualization
"""
import os
import traceback
import torch
from utils.DataUtils import *
from loaders.HandDataset import *
import trimesh, mcubes
import shutil
import torch.nn as nn

class Validater:

    def __init__(self, config, model, val_dataloader, objective, device):
        self.config = config
        self.model = model
        self.objective = objective
        self.device = device
        self.val_dataloader = val_dataloader

        self.num_test_sample = 3000
        self.model.net.to(device)
        self.output_dir = os.path.join(self.model.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

    def validate(self):
        if self.config.TASK in ["nocs", "pretrain"]:
            self.validate_nocs()
        if self.config.TASK == "lbs":
            self.validate_lbs()
        elif self.config.TASK == "occupancy":
            self.validate_occ()

    def validate_nocs(self):
        self.model.setup_checkpoint(self.device)
        self.model.net.eval()        
        
        epoch_losses = []
        for i, data in enumerate(self.val_dataloader, 0):  # Get each batch        
            if i >= self.num_test_sample:
                break
            net_input, target = self.model.preprocess(data, self.device)
            output = self.model.net(net_input)
            loss = self.objective(output, target)
            epoch_losses.append(loss.item())
            print("validating on the {}th data, loss is {}".format(i, loss))
            print("average validation loss is ",np.mean(np.asarray(epoch_losses)))            
            self.save_img(net_input, output, target, i)            
        print("average validation loss is ", np.mean(np.asarray(epoch_losses)))

    def validate_lbs(self):
        self.model.setup_checkpoint(self.device)
        self.model.net.eval()

        epoch_losses = []
        for i, data in enumerate(self.val_dataloader, 0):  # Get each batch        
            if i >= self.num_test_sample:
                break
            net_input, target = self.model.preprocess(data, self.device)
            output = self.model.net(net_input)
            loss = self.objective(output, target)
            epoch_losses.append(loss.item())
            print("validating on the {}th data, loss is {}".format(i, loss))
            print("average validation loss is ", np.mean(np.asarray(epoch_losses)))            
            self.save_img(net_input, output[:, 0:4, :, :], target['maps'][:, 0:4, :, :], i)
            # self.save_mask(output, ta`rget['maps'], i)
            self.save_joint_location(output, target, i)
        print("average validation loss is ", np.mean(np.asarray(epoch_losses)))

    def validate_occ(self):
        self.model.setup_checkpoint(self.device)
        self.model.net.eval()

        if self.config.TASK == "occupancy":            
            for child in self.model.net.IFNet.children():                        
                if type(child) == nn.BatchNorm3d:
                    child.track_running_stats = False
                    
        output_dir = self.output_dir
        resolution = 128
        test_net = self.model.net

        num_samples = min(self.num_test_sample, len(self.val_dataloader))
        print('Testing on ' + str(num_samples) + ' samples')

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)

        grid_coords = test_net.initGrids(resolution)
        batch_points = 100000
        grid_points_split = torch.split(grid_coords, batch_points, dim=1)
        for i, data in enumerate(self.val_dataloader, 0):  # Get each batch
            if i > (num_samples-1): break                 
            logits_list = []
            target = None
            net_input, target = self.model.preprocess(data, self.device)
            output = self.model.net(net_input)
            loss = self.objective(output, target)
            print(loss)
            # logits = output[1].squeeze(0).detach().cpu()
            for points in grid_points_split:
                with torch.no_grad():
                    net_input, target = self.model.preprocess(data, self.device)
                    # print(data)
                    net_input['grid_coords'] = points.to(self.device)
                    output = test_net(net_input)
                    self.save_img(net_input['RGB'], output[0], target['NOCS'], i)
                    logits_list.append(output[1].squeeze(0).detach().cpu())

            logits = torch.cat(logits_list, dim=0).numpy()
            print(logits.shape)
            mesh = self.mesh_from_logits(logits, resolution)
            export_pred_path = os.path.join(output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
            export_gt_path = os.path.join(output_dir, "frame_{}_gt.off".format(str(i).zfill(3)))
            export_trans_path = os.path.join(output_dir, "frame_{}_trans.npz".format(str(i).zfill(3)))

            print(export_pred_path)
            mesh.export(export_pred_path)

            shutil.copyfile(target['mesh'][0], export_gt_path)

            trans_path = target['mesh'][0].replace("isosurf_scaled.off", "transform.npz")
            shutil.copyfile(trans_path, export_trans_path)

    def save_img(self, net_input, output, target, i):
        input_img, gt_out_tuple_rgb, gt_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(target, 'cpu'))
        _, pred_out_tuple_rgb, pred_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(output.detach(), 'cpu'), isMaskNOX=True)
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_color00.png').format(str(i).zfill(3)),  cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        out_target_str = [self.config.TARGETS]
        if 'color00' in out_target_str:
            out_target_str.remove('color00')
        if 'camera' in out_target_str:
            out_target_str.remove('camera')

        for target_str, gt, pred, gt_mask, pred_mask in zip(out_target_str, gt_out_tuple_rgb, pred_out_tuple_rgb, gt_out_tuple_mask, pred_out_tuple_mask):
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_00gt.png').format(str(i).zfill(3), target_str),
                        cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_01pred.png').format(str(i).zfill(3), target_str),
                        cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_02gtmask.png').format(str(i).zfill(3), target_str),
                        gt_mask)
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_03predmask.png').format(str(i).zfill(3), target_str),
                        pred_mask)

    def save_mask(self, output, target, i):        
        bone_num = 16
        mask = target[:, 3, :, :]
        print(mask.max())
        sigmoid = torch.nn.Sigmoid()
        zero_map = torch.zeros(mask.size(), device=mask.device)    
        pred_bw_index =  4+bone_num*7
        tar_bw_index =  4
        for b_id in range(16):
            pred_bw = sigmoid(output[:, pred_bw_index + b_id, :, :])*255
            pred_bw = torch.where(mask > 0.7, pred_bw, zero_map)
            pred_bw = pred_bw.squeeze().cpu().detach().numpy()

            tar_bw = target[:, tar_bw_index + b_id, :, :]*255
            tar_bw = torch.where(mask > 0.7, tar_bw, zero_map)
            tar_bw = tar_bw.squeeze().cpu().detach().numpy()
            # print(tar_bw.shape)

            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_BW{}_00gt.png').format(str(i).zfill(3), b_id), tar_bw)
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_BW{}_01pred.png').format(str(i).zfill(3), b_id), pred_bw)
    
    def save_joint_location(self, output, target, i, use_score=False):
        """
        input: 
            predicted joint map
            joint score
            target joints
        
            get the predicted joint position from 
        """
        bone_num = 16
        n_batch = output.shape[0]
        pred_joint_map = output[:, 4+bone_num*1:4+bone_num*4, :, :]
        
         # get final prediction: score map summarize
        pred_joint_map = pred_joint_map.reshape(n_batch, bone_num, 3, pred_joint_map.shape[2],
                                                pred_joint_map.shape[3])  # B,bone_num,3,R,R    
        out_mask = target['maps'][:, 3, :, :]
        pred_joint_map = pred_joint_map * out_mask.unsqueeze(1).unsqueeze(1)                        
        if use_score:
            joint_loc_scores = output[:, 4:4+bone_num*1, :, :]
            pred_joint_score = sigmoid(pred_joint_score) * out_mask.unsqueeze(1)                          
            pred_score_map = pred_joint_score / (torch.sum(pred_joint_score.reshape(n_batch, bone_num, -1),
                                                    dim=2, keepdim=True).unsqueeze(3) + 1e-5)
            pred_joint_map = pred_joint_map.detach() * pred_score_map.unsqueeze(2)
            pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
        else:
            pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
            pred_joint /= out_mask.nonzero().shape[0]
        
        pred_joint = pred_joint[0] # retrive the first one from batch
        pred_path = os.path.join(self.output_dir, 'frame_{}_pred_loc.xyz').format(str(i).zfill(3))

        # f = open(pred_path, "a")
        # for i in range(pred_joint.shape[0]):
        #     p = pred_joint[i]
        #     f.write("{} {} {}\n".format(p[0], p[1], p[2]))
        # f.close()

        gt_joint = target['pose'][0, :, 0:3]
        gt_path = os.path.join(self.output_dir, 'frame_{}_gt_loc.xyz').format(str(i).zfill(3))
        print(gt_joint.shape)
        fgt = open(gt_path, "a")
        for i in range(gt_joint.shape[0]):
            p = gt_joint[i]
            fgt.write("{} {} {}\n".format(p[0], p[1], p[2]))
        fgt.close()
        
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