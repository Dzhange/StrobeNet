"""
The model to train, validate and test NOCS
"""
import os
import glob
import torch
import torch.nn as nn 
from models.networks.SegNet import SegNet as old_SegNet
from models.networks.say4n_SegNet import SegNet as new_SegNet
from models.loss import L2MaskLoss
from utils.DataUtils import *

class ModelNOCS(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR # set learning rate
        # self.net = new_SegNet(input_channels=3, output_channels=config.OUT_CHANNELS)
        if config.TASK == "pretrain":
            self.net = old_SegNet(output_channels=config.OUT_CHANNELS + config.FEATURE_CHANNELS, bn=config.BN) # 3 + 1 + 16 * 1 = 116
        else:
            self.net = old_SegNet(output_channels=config.OUT_CHANNELS)

        self.objective = L2MaskLoss()

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

    def preprocess(self, data, device):
        data_todevice = []
        for item in data:
            tuple_or_tensor = item
            tuple_or_tensor_td = item
            if not isinstance(tuple_or_tensor_td, (tuple, list)):
                tuple_or_tensor_td = tuple_or_tensor.to(device)
            else:
                for ctr in range(len(tuple_or_tensor)):
                    if isinstance(tuple_or_tensor[ctr], torch.Tensor):
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr].to(device)
                    else:
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr]

            data_todevice.append(tuple_or_tensor_td)
        return data_todevice

    def validate(self, val_dataloader, objective, device):

        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)
        self.setup_checkpoint(device)
        self.net.eval()
        # self.net.train()
        
        # for child in self.net.children():
        #     if type(child) == nn.BatchNorm2d:
        #         child.track_running_stats = False

        num_test_sample = 30
        epoch_losses = []
        for i, data in enumerate(val_dataloader, 0):  # Get each batch        
            if i >= num_test_sample:
                break
            net_input, target = self.preprocess(data, device)
            output = self.net(net_input)
            loss = objective(output, target)
            epoch_losses.append(loss.item())
            
            print("validating on the {}th data, loss is {}".format(i, loss))
            print("average validation loss is ",np.mean(np.asarray(epoch_losses)))            
            self.save_img(net_input, output[:, :4, :, :], target, i)
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