"""
Loader for HandRigDataSet
"""
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby

import numpy as np
import torch
import torch.utils.data
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
from utils.DataUtils import *
from models.loss import L2MaskLoss, L2Loss, LBSLoss
import trimesh
import utils.tools.implicit_waterproofing as iw
import json

class OccNetDataset(torch.utils.data.Dataset):
    """
    A basic dataloader for HandOccDataset
    """
    def __init__(self, config, train=True, required='color00'):
        
        self.config = config
        root = config.DATASET_ROOT
        limit = config.DATA_LIMIT if train else config.VAL_DATA_LIMIT
        img_size = config.IMAGE_SIZE
        self.num_cameras = 10
        self.frame_load_str = ['color00']

        self.init(root, train, img_size, limit, self.frame_load_str, required)
        self.load_data()

    def init(self, root, train=True,
             img_size=(320, 240), limit=100, frame_load_str=None, required='VertexColors'):
        self.dataset_dir = root
        config_path = os.path.join(root,'config.json')
        # pose num let us know how can we find the occupancies
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
                self.dataset_config = configs
                self.pose_num = configs['pose_num']
        else:
                self.pose_num = 1
        self.is_train_data = train        
        self.img_size = img_size
        self.required = required
        self.frame_load_str = frame_load_str

        # self.occ_load_str = ['boundary_0.1_samples', 'boundary_0.01_samples']
        self.occ_load_str = 'boundary_0.01_samples.npz'
        self.num_sample_points = 2048        
        # self.num_sample_points = 100000

        self.shuffle_in_limit = True
        # print(limit)
        if limit <= 0.0 or limit > 100.0:
            raise RuntimeError('Data limit percent has to be >0% and <=100%')
        self.data_limit = limit
        if self.required not in self.frame_load_str:
            raise RuntimeError('frame_load_str should contain {}.'.format(self.required))
        if os.path.exists(self.dataset_dir) == False:
            print("Dataset {} doesn't exist".format(self.dataset_dir))
            exit()

    def __len__(self):
        return len(self.frame_files[self.frame_load_str[0]])

    def __getitem__(self, idx):

        required_path = self.frame_files[self.required][idx]
        # print("path is ", required_path)
        RGB = self.load_images(idx)
        
        # if self.is_train_data:
        occ_data = self.load_occupancies(required_path)
        return RGB, occ_data

        #occupancy related ground truth is not needed in validation
        # return RGB, (load_imgs)

    def load_data(self):
        """
        Get the index of files
        """
        self.frame_files = {}

        if self.is_train_data:
            file_path = os.path.join(self.dataset_dir, 'train/')
        else:
            file_path = os.path.join(self.dataset_dir, 'val/')

        # Load index for data 
        camera_idx_str = '*'
        glob_prepend = '_'.join(str(i) for i in self.frame_load_str)
        glob_cache = os.path.join(file_path, 'all_glob_' + glob_prepend + '.cache')
        if os.path.exists(glob_cache):
            # use pre-cached index
            print('[ INFO ]: Loading from glob cache:', glob_cache)
            with open(glob_cache, 'rb') as fp:
                for string in self.frame_load_str:
                    self.frame_files[string] = pickle.load(fp)
        else:
            # glob and save cache
            print('[ INFO ]: Saving to glob cache:', glob_cache)
            for string in self.frame_load_str:
                self.frame_files[string] = glob.glob(file_path + '/**/frame_*_view_' + camera_idx_str + '_' + string + '.*')
                self.frame_files[string].sort()
            with open(glob_cache, 'wb') as fp:
                for string in self.frame_load_str:
                    pickle.dump(self.frame_files[string], fp)

        frame_files_lengths = []
        for k, cur_frame_files in self.frame_files.items():
            if not cur_frame_files:
                raise RuntimeError('[ ERR ]: None data for', k)
            if len(cur_frame_files) == 0:
                raise RuntimeError('[ ERR ]: No files found during data loading for', k)
            frame_files_lengths.append(len(cur_frame_files))

        if len(set(frame_files_lengths)) != 1:
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match', frame_files_lengths)
        
        
        if self.shuffle_in_limit:
            total_size = len(self)
            dataset_length = math.ceil((self.data_limit / 100) * total_size)
            # dataset_length = 64
            step = int(np.floor(100 / self.data_limit))        
            sample_index = []
            cur_idx = 0
            while True:
                if len(sample_index) >= dataset_length or cur_idx >= total_size:
                    break
                sample_index.append(cur_idx)
                cur_idx += step                                    
            print('[ INFO ]: Loading {} / {} items.'.format(len(sample_index), total_size))
                        
            for K in self.frame_files:
                self.frame_files[K] = [self.frame_files[K][i] for i in sample_index]
                self.frame_files[K].sort()
        else:                
            total_size = len(self)
            dataset_length = math.ceil((self.data_limit / 100) * total_size)
            print('[ INFO ]: Loading {} / {} items.'.format(dataset_length, total_size))

            # dataset_length = 1
            for k in self.frame_files:
                self.frame_files[k] = self.frame_files[k][:dataset_length]

    def load_images(self, idx):
        """
        actual implementation of __getitem__
        """
        frame = {}
        for k in self.frame_files:
            # print(self.frame_files[k][idx])
            if 0:
                frame[k] = imread_rgb_torch("/workspace/Data/SAPIEN/eyeglasses/aug_scale_mpf_uni/train/0000/frame_00000000_view_00_color00.png", Size=self.img_size).type(torch.FloatTensor)
            else:
                frame[k] = imread_rgb_torch(self.frame_files[k][idx], Size=self.img_size).type(torch.FloatTensor)                        
            # frame[k] = imread_rgb_torch('/workspace/Data/shapenet_sample/img_choy2016/000.jpg', Size=self.img_size).type(torch.FloatTensor)                        
            frame[k] /= 255.0
        

        return frame['color00']

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
    
    def load_occupancies(self, required_path):
        """
        load the occupancy data for the 2nd part(IF-Net)
            coords: position to be queried
            occupancies: ground truth for position to be queried
        """
        data_dir = os.path.dirname(required_path)
        file_name = os.path.basename(required_path)
        index_of_frame = find_frame_num(file_name)
        
        

        points = []
        coords = []
        occupancies = []



        if 0:
            # path = "/workspace/Data/shapenet_sample/points.npz"
            path = "/workspace/Data/SAPIEN/eyeglasses/aug_scale_mpf_uni/train/0000/frame_00000000_view_03_uni_sampled.npz"
            pointsf = np.load(path)['points']

            # write_off("/workspace/z.xyz", pointsf)
            # exit()
            grid_occ = np.load(path)['occupancies']
            # grid_occ = np.unpackbits(np.load(path)['occupancies'])
            # print(pointsf.shape)
            sample_indices = np.random.randint(0, len(pointsf), self.num_sample_points)

            return {
                'grid_coords':pointsf.astype(np.float32)[sample_indices],
                'occupancies':np.array(grid_occ).astype(np.float32)[sample_indices],
                'iso_mesh': gt_mesh_path
            }
        elif 0:
            # boundary_samples_path = "/workspace/Data/SAPIEN/eyeglasses/aug_scale_mpf_uni/train/0000/frame_00000000_boundary_0.01_samples.npz"
            grid_sample = "/workspace/Data/debug/frame_00000000_grid_0.01_samples.npz"
            if os.path.exists(grid_sample):
                pointsf = np.load(grid_sample)['points']
                grid_occ = np.load(grid_sample)['occupancies']
            else:
                nx = 128
                pointsf = 1.1 * self.make_3d_grid(
                    (-0.5,)*3, (0.5,)*3, (nx,)*3
                ).numpy()
                # generate online
                mesh = trimesh.load(gt_mesh_path)            
                grid_occ = iw.implicit_waterproofing(mesh, pointsf)[0]
                np.savez(grid_sample, points=pointsf,
                        occupancies=grid_occ)
            
            p_occ = pointsf[grid_occ]
            write_off("/workspace/in_load.xyz", p_occ)
            # exit()

            sample_indices = np.random.randint(0, len(pointsf), self.num_sample_points)
            return {
                'grid_coords':pointsf[sample_indices],
                'occupancies':grid_occ.astype(np.float32)[sample_indices],
                'iso_mesh': gt_mesh_path
            }

        else:
            if not self.config.ONET_CANO:
                boundary_samples_path = required_path.replace('color00.png', self.occ_load_str)
            else:
                index_of_frame = str(int(index_of_frame) // self.pose_num * self.pose_num).zfill(8)
                boundary_samples_path = os.path.join(data_dir, "frame_{}_{}".format(index_of_frame, self.occ_load_str))
        
        # boundary_samples_npz = np.load(boundary_samples_path)
        # boundary_sample_points = boundary_samples_npz['points']        
        # boundary_sample_occupancies = boundary_samples_npz['occupancies']
        # gt_mesh_path = required_path.replace('color00.png', 'isosurf_scaled.off')
        uniform_sample_path = boundary_samples_path.replace(self.occ_load_str, "uni_sampled.npz")
        gt_mesh_path = boundary_samples_path.replace(self.occ_load_str, 'isosurf_scaled.off')
        # print(uniform_sample_path)
        if os.path.exists(uniform_sample_path):
            uniform_points = np.load(uniform_sample_path)['points']
            uniform_occ = np.load(uniform_sample_path)['occupancies']
        else:
            # generate online
            mesh = trimesh.load(gt_mesh_path)
            uniform_points = np.random.rand(100000, 3)
            uniform_points = 1.1 * (uniform_points - 0.5)
            uniform_occ = iw.implicit_waterproofing(mesh, uniform_points)[0]
            np.savez(uniform_sample_path, points=uniform_points,
                     occupancies=uniform_occ)
                
        sample_indices = np.random.randint(0, len(uniform_points), self.num_sample_points)
                
        # points.extend(boundary_sample_points)
        points.extend(uniform_points[sample_indices])

        # occupancies.extend(boundary_sample_occupancies)
        occupancies.extend(uniform_occ[sample_indices])

        # points = points[sample_indices]
        # occupancies = occupancies[sample_indices]

        # assert len(points) == self.num_sample_points
        # assert len(occupancies) == self.num_sample_points
        # assert len(coords) == self.num_sample_points
        
        
        # None of the if-data would be needed if in validation mode
        if_data = {            
            'grid_coords':np.array(points, dtype=np.float32),
            'occupancies': np.array(occupancies, dtype=np.float32),
            'iso_mesh': gt_mesh_path
            }

        return if_data



if __name__ == '__main__':
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HandRigDatasetV2', required=True)
    
    Args, _ = Parser.parse_known_args()

    Data = OccNetDataset(root=Args.data_dir, train=True, frame_load_str=["color00"])
    
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=64, shuffle=True, num_workers=16)
    # loss = LBSLoss()
    for i, data in enumerate(DataLoader, 0):  # Get each batch
        # DataTD = ptUtils.sendToDevice(Targets, 'cpu')
        # print('Data size:', Data.size())
        print(data[1]['occupancies'].shape)
        # l = loss(Targets[0], Targets)
        # print(l)