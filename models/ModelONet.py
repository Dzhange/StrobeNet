"""
This model is the FIRST stage of out proposed LBS pipeline
Here we predict the skinning weights, the pose and the confident score
"""
import os, sys, shutil
import glob
import torch
import torch.nn as nn
import trimesh
from models.networks.OccNet import OccupancyNetwork

from models.loss import LBSLoss

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))

from expt.im2mesh.common import *
from expt.im2mesh.utils import libmcubes
import utils.tools.implicit_waterproofing as iw
from utils.DataUtils import *
import time
from torch import distributions as dist
import trimesh, mcubes

    
class ModelONet(object):

    def __init__(self, config):
        self.config = config
        self.lr = config.LR # set learning rate

        self.loss_history = []
        self.val_loss_history = []
        self.start_epoch = 0
        self.expt_dir_path = os.path.join(expandTilde(self.config.OUTPUT_DIR), self.config.EXPT_NAME)
        self.output_dir = os.path.join(self.expt_dir_path, "ValResults")
        
        if os.path.exists(self.expt_dir_path) == False:
            os.makedirs(self.expt_dir_path)
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

        device = torch.device(config.GPU)
        self.init_net(device=device)

        self.device = device
        self.threshold = 0.5
        self.eval_sample = False
        self.init_grids(128)
        self.view_num = self.config.VIEW_NUM

    def init_net(self, device=None):
        config = self.config
        self.net = OccupancyNetwork(view_num=config.VIEW_NUM, device=device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr)

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
        """
        put data onto the right device
        """

        if self.view_num > 1:
            inputs = [img.to(device=device) for img in data['color00']]
            points = data['grid_coords'][0].to(device=device)
            occ = data['occupancies'][0].to(device=device)
        else:
            inputs = data[0].to(device=device)        
            points = data[1]['grid_coords'].to(device=device)
            occ = data[1]['occupancies'].to(device=device)

        data = {}
        data['inputs'] = inputs
        data['points'] = points
        data['occupancies'] = occ

        return data

    def train_step(self, data):
        
        data = self.preprocess(data, self.device)

        self.net.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        
        return loss


    def compute_loss(self, data):

        device = self.device
        p = data.get('points').to(device)
        occ = data.get('occupancies').to(device)
        
                
        if self.view_num == 1:
            inputs = data.get('inputs').to(device)
            c = self.net.encode_inputs(inputs)
        else:
            inputs = data.get('inputs')
            con_list = []
            for img in inputs:
                c = self.net.encode_inputs(img)
                con_list.append(c)
            c = torch.cat(tuple(con_list), dim=1)

        # q_z = self.net.infer_z(p, occ, c)
        # z = q_z.rsample()
        z = None
        # KL-divergence
        # kl = dist.kl_divergence(q_z, self.net.p0_z).sum(dim=-1)
        # loss = kl.mean()

        # General points
        
        logits = self.net.decode(p, z, c).logits
        # print(logits.sum(), occ.sum())
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        # loss = loss + loss_i.sum(-1).mean()
        loss = loss_i.sum(-1).mean()

        # print(loss_i.sum(-1).mean() / loss_i.shape[1])
        return loss


    def val_step(self, data):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
        '''
        self.net.eval()        
        data = self.preprocess(data, self.device)
        loss = self.compute_loss(data)
        return loss


    def init_grids(self, resolution):
        
        bb_min = -0.5
        bb_max = 0.5
       
        # self.GridPoints = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()        
        
        a = bb_max + bb_min # 0, 1
        b = bb_max - bb_min # 1, 1
        grid_coords = 2 * grid_points - a # 
        grid_coords = grid_coords / b


        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_coords = grid_coords

    @staticmethod
    def make_3d_grid(bb_min, bb_max, shape):
        ''' Makes a 3D grid.

        Args:
            bb_min (tuple): bounding box minimum
            bb_max (tuple): bounding box maximum
            shape (tuple): output shape
        '''
        size = shape[0] * shape[1] * shape[2]

        pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
        pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
        pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

        pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
        pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
        pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
        p = torch.stack([pxs, pys, pzs], dim=1)

        return p


    def eval_step(self, batch_data, i):

        box_size = 1.1
        self.batch_points = 2097152
        nx = 128
        # grid_points_split = torch.split(self.grid_coords, self.batch_points, dim=1)
        pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
                    
        device = self.device
        logits_list = []
        
        with torch.no_grad():
            data = self.preprocess(batch_data, device)

            p = pointsf.to(device).unsqueeze(0)
            # p = data.get('points').to(device) # DEBUG
            # p = pointsf.to(device).unsqueeze(0)
            # print(p.shape)
            
            occ = data.get('occupancies').to(device)
                        
            # inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
                        
            if self.view_num == 1:                
                inputs = data.get('inputs').to(device) 
                c = self.net.encode_inputs(inputs)
            else:
                inputs = data.get('inputs')
                con_list = []
                for img in inputs:
                    c = self.net.encode_inputs(img)
                    con_list.append(c)
                c = torch.cat(tuple(con_list), dim=1)

            # q_z = self.net.infer_z(p, occ, c)
            # z = q_z.rsample()
            z = None
            # General points
            p_split = torch.split(p, 100000)
            # occ_hats = []

            for pi in p_split:
                logits = self.net.decode(pi, z, c).logits
                logits_list.append(logits.squeeze(0).detach().cpu())
                # print(logits.shape)
                # p_occ = p[0][logits[0] > 0]
                # write_off("/workspace/text.xyz", p_occ)
                # exit()
                

            # loss_i = F.binary_cross_entropy_with_logits(
            # logits, occ, reduction='none')                                
        # generate predicted mesh from occupancy and save
        logits = torch.cat(logits_list, dim=0).numpy()
        # print(logits)
        # mesh = self.mesh_from_logits(logits, 128)
        logits = logits.reshape((nx,)*3)
        mesh = self.extract_mesh(logits)
        export_pred_path = os.path.join(self.output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
        mesh.export(export_pred_path)

        # self.save_img(net_input['RGB'], output[0], target['NOCS'], i)/
        # Copy ground truth in the val results
        export_gt_path = os.path.join(self.output_dir, "frame_{}_gt.off".format(str(i).zfill(3)))
        # print(target['mesh'][0])
        if self.config.VIEW_NUM == 1:
            shutil.copyfile(batch_data[1]['iso_mesh'][0], export_gt_path)        
        else:
            print(batch_data['iso_mesh'][0])
            shutil.copyfile(batch_data['iso_mesh'][0][0], export_gt_path)

    def mesh_from_logits(self, logits, resolution):
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

    def _eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.net.eval()
        data = self.preprocess(data, self.device)
        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('occupancies').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.net.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.net(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.net(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels
        
        print(eval_dict)
        return eval_dict

    def extract_mesh(self, occ_hat):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        import time
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1.1
        threshold = np.log(0.2) - np.log(1. - 0.2)
        # Make sure that mesh is watertight
        
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        # if self.with_normals and not vertices.shape[0] == 0:        
        #     normals = self.estimate_normals(vertices, z, c)    
        # else:
        #     normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               process=False)

        # Directly return if mesh is empty
        # if vertices.shape[0] == 0:
        #     return mesh

        # TODO: normals are lost here
        # if self.simplify_nfaces is not None:
        #     t0 = time.time()
        #     mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
        #     stats_dict['time (simplify)'] = time.time() - t0

        # # Refine mesh
        # if self.refinement_step > 0:
        #     t0 = time.time()
        #     self.refine_mesh(mesh, occ_hat, z, c)
        #     stats_dict['time (refine)'] = time.time() - t0

        return mesh