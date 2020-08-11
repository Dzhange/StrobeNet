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

class Validater:

    def __init__(self, config, model, val_dataloader, objective, device):
        self.config = config
        self.model = model
        self.objective = objective
        self.device = device
        self.val_dataloader = val_dataloader                

        self.num_test_sample = 30
        self.model.net.to(device)
        self.output_dir = os.path.join(self.model.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)


    def validate(self):
        if self.config.TASK == "nocs":
            self.validate_nocs()
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

    
    def validate_occ(self):
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
            for points in grid_points_split:
                with torch.no_grad():
                    net_input, target = self.model.preprocess(data, self.device)
                    # print(data)
                    net_input['grid_coords'] = points.to(self.device)
                    output = test_net(net_input)
                    self.save_img(net_input['RGB'], output[0], target['NOCS'], i)
                    logits_list.append(output[1].squeeze(0).detach().cpu())

            logits = torch.cat(logits_list, dim=0).numpy()

            mesh = self.mesh_from_logits(logits, resolution)
            export_pred_path = os.path.join(output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
            export_gt_path = os.path.join(output_dir, "frame_{}_gt.off".format(str(i).zfill(3)))

            print(export_pred_path)
            mesh.export(export_pred_path)
            
            shutil.copyfile(target['mesh'][0], export_gt_path)

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