"""
This model is the FIRST stage of out proposed LBS pipeline
Here we predict the skinning weights, the pose and the confident score
"""
import os, shutil
import glob
import torch
import torch.nn as nn
import trimesh
from models.networks.SegNet import SegNet as old_SegNet
from models.networks.say4n_SegNet import SegNet as new_SegNet
from models.loss import LBSLoss
from utils.DataUtils import *
import time

class ModelLBSNOCS(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR # set learning rate
        # self.net = new_SegNet(input_channels=3, output_channels=config.OUT_CHANNELS)
        self.net = old_SegNet(output_channels=config.OUT_CHANNELS, bn=False)
        # self.objective = LBSLoss()
        # self.objective = LBSLoss()
        self.bone_num = 16
        self.loss_history = []
        self.val_loss_history = []
        self.start_epoch = 0
        self.expt_dir_path = os.path.join(expandTilde(self.config.OUTPUT_DIR), self.config.EXPT_NAME)
        if os.path.exists(self.expt_dir_path) == False:
            os.makedirs(self.expt_dir_path)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                          betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))

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

        inputs = {}
        inputs = data_to_device[0]

        targets = {}
        targets['maps'] = data_to_device[1]
        targets['pose'] = data_to_device[2]
        targets['mesh'] = data[3]

        return inputs, targets

    def validate(self, val_dataloader, objective, device):

        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

        self.setup_checkpoint(device)
        self.net.eval()

        num_test_sample = 300

        epoch_losses = []
        for i, data in enumerate(val_dataloader, 0):  # Get each batch
            if i >= num_test_sample: break
            net_input, target = self.preprocess(data, device)
            output = self.net(net_input)
            loss = objective(output, target)
            epoch_losses.append(loss.item())
            print("validating on the {}th data, loss is {}".format(i, loss))
            print("average validation loss is ", np.mean(np.asarray(epoch_losses)))            
            self.save_img(net_input, output[:, 0:4, :, :], target['maps'][:, 0:4, :, :], i)
            if self.config.SKIN_LOSS or self.config.TASK == "lbs_seg":
                self.save_mask(output, target['maps'], i)
            if self.config.LOC_LOSS or self.config.LOC_MAP_LOSS:
                self.save_joint_location(output, target, i, self.config.LOC_LOSS)
                self.visualize_joint_prediction(output, target, i)
                gt_path = os.path.join(self.output_dir, 'frame_{}_gt.obj').format(str(i).zfill(3))
                mesh = trimesh.load(target['mesh'][0])
                trimesh.repair.fix_inversion(mesh)
                mesh.export(gt_path)
                # shutil.copyfile(target['mesh'][0], gt_path)

        print("average validation loss is ", np.mean(np.asarray(epoch_losses)))
    
    def save_img(self, net_input, output, target, i):
        input_img, gt_out_tuple_rgb, gt_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(target, 'cpu'))
        _, pred_out_tuple_rgb, pred_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(output.detach(), 'cpu'), isMaskNOX=True)
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_color00.png').format(str(i).zfill(3)), cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

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

    def save_mask(self, output, target, i):
        bone_num = self.bone_num
        mask = target[:, 3, :, :]
        print(mask.max())
        sigmoid = torch.nn.Sigmoid()
        zero_map = torch.zeros(mask.size(), device=mask.device)
        pred_bw_index = 4+bone_num*6
        tar_bw_index = 4+bone_num*6

        for b_id in range(bone_num):
            pred_bw = sigmoid(output[:, pred_bw_index + b_id, :, :])*255
            pred_bw = torch.where(mask > 0.7, pred_bw, zero_map)
            pred_bw = pred_bw.squeeze().cpu().detach().numpy()

            tar_bw = target[:, tar_bw_index + b_id, :, :]*255
            tar_bw = torch.where(mask > 0.7, tar_bw, zero_map)
            tar_bw = tar_bw.squeeze().cpu().detach().numpy()
            

            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_BW{}_00gt.png').format(str(i).zfill(3), b_id), tar_bw)
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_BW{}_01pred.png').format(str(i).zfill(3), b_id), pred_bw)

    def save_joint_location(self, output, target, i, use_score=False):
        """
        input:
            predicted joint map
            joint score
            target joints

            get the predicted joint position from prediction
        """
        bone_num = self.bone_num
        n_batch = output.shape[0]
        pred_joint_map = output[:, 4:4+bone_num*3, :, :]
        # get final prediction: score map summarize
        pred_joint_map = pred_joint_map.reshape(n_batch, bone_num, 3, pred_joint_map.shape[2],
                                                pred_joint_map.shape[3])  # B,bone_num,3,R,R
        out_mask = target['maps'][:, 3, :, :]
        pred_joint_map = pred_joint_map * out_mask.unsqueeze(1).unsqueeze(1)

        # gt
        gt_joint = target['pose'][0, :, 0:3]
        gt_path = os.path.join(self.output_dir, 'frame_{}_gt_loc.xyz').format(str(i).zfill(3))
        self.write(gt_path, gt_joint)

        # Mean results
        gt_joint_map = target['maps'][:, 4:4+bone_num*3, :, :]
        gt_joint_map = gt_joint_map.reshape(n_batch, bone_num, 3, gt_joint_map.shape[2],
                                                gt_joint_map.shape[3])  # B,bone_num,3,R,R
        gt_joint_map = gt_joint_map * out_mask.unsqueeze(1).unsqueeze(1)

        mean_gt_joint = gt_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
        mean_gt_joint /= out_mask.nonzero().shape[0]
        mean_gt_joint = mean_gt_joint[0]
        mean_gt_path = os.path.join(self.output_dir, 'frame_{}_gt_mean_loc.xyz').format(str(i).zfill(3))
        self.write(mean_gt_path, mean_gt_joint)

        if use_score:
            # # Vote results
            sigmoid = nn.Sigmoid()
            pred_joint_score = output[:, 4+bone_num*7:4+bone_num*8, :, :]
            pred_joint_score = sigmoid(pred_joint_score) * out_mask.unsqueeze(1)
            pred_score_map = pred_joint_score / (torch.sum(pred_joint_score.reshape(n_batch, bone_num, -1),
                                                    dim=2, keepdim=True).unsqueeze(3) + 1e-5)

            pred_joint_map = pred_joint_map.detach() * pred_score_map.unsqueeze(2)
            pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3

            pred_joint = pred_joint[0] # retrive the first one from batch
            pred_path = os.path.join(self.output_dir, 'frame_{}_pred_loc.xyz').format(str(i).zfill(3))
            self.write(pred_path, pred_joint)
        else:
            # Mean results
            mean_pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
            mean_pred_joint /= out_mask.nonzero().shape[0]
            mean_pred_joint = mean_pred_joint[0]
            mean_pred_path = os.path.join(self.output_dir, 'frame_{}_mean_pred_loc.xyz').format(str(i).zfill(3))
            self.write(mean_pred_path, mean_pred_joint)

        # tell difference
        if use_score:
            joint_diff = torch.sum((pred_joint - gt_joint) ** 2, dim=1)  # B,22
            joint_loc_loss = joint_diff.sum() / (n_batch * bone_num)
            print("[ DIFF ] vote diff is {:5f}".format(joint_loc_loss))
        else:
            mean_joint_diff = torch.sum((mean_pred_joint - gt_joint) ** 2, dim=1)  # B,22
            mean_joint_loc_loss = mean_joint_diff.sum() / (n_batch * bone_num)
            print("[ DIFF ] mean diff is {:5f}".format(mean_joint_loc_loss))

    def visualize_joint_prediction(self, output, target, frame_id):
        """
        save the inter results of joint predication as RGB image
        """
        bone_num = self.bone_num
        mask = target['maps'][:, 3, :, :].cpu().detach()
        pred_joint_map = output[:, 4:4+bone_num*3, :, :].cpu().detach()
        gt_joint_map = target['maps'][:, 4:4+bone_num*3, :, :].cpu().detach()
        
        zero_map = torch.zeros(3, pred_joint_map.shape[2], pred_joint_map.shape[3])
        
        to_cat = ()
        for i in range(bone_num):
            cur_pred = pred_joint_map[0, i*3:i*3+3, :, :]
            gt = gt_joint_map[0, i*3:i*3+3, :, :]
            
            masked_map = torch.where(mask > 0.7, cur_pred, zero_map)
            joint_map = torch2np(masked_map) * 255

            gt = torch.where(mask > 0.7, gt, zero_map)
            gt = torch2np(gt) * 255
            
            diff = np.abs(gt-joint_map)
            comb = np.concatenate((gt, joint_map, diff), axis=1)

            to_cat = to_cat + (comb, )
        
        big_img = np.concatenate(to_cat, axis=0)
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_pred_joint.png').format(str(frame_id).zfill(3)),
                        cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB))


    def write(self, path, joint):
        f = open(path, "a")
        for i in range(joint.shape[0]):
            p = joint[i]
            f.write("{} {} {}\n".format(p[0], p[1], p[2]))
        f.close()