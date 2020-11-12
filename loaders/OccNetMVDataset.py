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
import time 

class OccNetMVDataset(torch.utils.data.Dataset):
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
        self.view_num = config.VIEW_NUM
        self.rng = np.random.RandomState(0)
        
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

    
    def get_sv_data(self, frame_path, view):
        data = self.load_images(frame_path, view)
        occ = self.load_occupancies(frame_path)
        data.update(occ)
        return data


    def __getitem__(self, idx):
        
        data_list = []
        batch = {}
        frame_base_path = self.frame_ids[idx]

        if self.config.RANDOM_VIEW:
            if self.is_train_data:
                views = np.random.choice(self.config.TOTAL_VIEW, self.view_num, replace=False)
            else:
                views = self.rng.choice(self.config.TOTAL_VIEW, self.view_num, replace=False)
        else:
            views = list(range(self.view_num))
        for view in views:
            data = self.get_sv_data(frame_base_path, view)
            data_list.append(data)
        for k in data_list[0].keys():
            batch[k] = [item[k] for item in data_list]
        return batch

    def load_data(self):
        # we only load valid frameIDS, we do not record 
        # file names as the view num is not specified
        self.frame_ids = []

        if self.is_train_data:
            file_path = os.path.join(self.dataset_dir, 'train/')
        else:
            file_path = os.path.join(self.dataset_dir, 'val/')

        glob_cache = os.path.join(file_path, 'all_glob_frames.cache')
        if os.path.exists(glob_cache):
            # use pre-cached index
            print('[ INFO ]: Loading from glob cache:', glob_cache)
            with open(glob_cache, 'rb') as fp:
                self.frame_ids = pickle.load(fp)
        else:
            # glob and save cache
            print('[ INFO ]: Saving to glob cache:', glob_cache)
            all_color_imgs = glob.glob(os.path.join(file_path, '**/frame_*_color00.*'))
            all_frames = [os.path.join(os.path.dirname(p), "frame_" + find_frame_num(p)) for p in all_color_imgs]
            all_frames = list(dict.fromkeys(all_frames)) # discard duplicated frame ids
            all_frames.sort()
            self.frame_ids = all_frames
            with open(glob_cache, 'wb') as fp:
                for string in self.frame_load_str:
                    pickle.dump(self.frame_ids, fp)

        if self.shuffle_in_limit:
            total_size = len(self)
            dataset_length = math.ceil((self.data_limit / 100) * total_size)
            # dataset_length = 20
            step = int(np.floor(100 / self.data_limit))
            sample_index = []
            cur_idx = 0
            while True:
                if len(sample_index) >= dataset_length or cur_idx >= total_size:
                    break
                sample_index.append(cur_idx)
                cur_idx += step                                    
            print('[ INFO ]: Loading {} / {} items.'.format(len(sample_index), total_size))

            
            self.frame_ids = [self.frame_ids[i] for i in sample_index]
            self.frame_ids.sort()
        else:
            total_size = len(self)
            dataset_length = math.ceil((self.data_limit / 100) * total_size)
            print('[ INFO ]: Loading {} / {} items.'.format(dataset_length, total_size))

        
            if self.data_limit > 10:
                self.frame_ids = self.frame_ids[:dataset_length]
            else:
                # offset only works for overfitting task
                self.frame_ids = self.frame_ids[self.data_offset:self.data_offset+dataset_length]

    def __len__(self):
        return len(self.frame_ids)

    def load_images(self, frame_path, view_id):
        """
        actual implementation of __getitem__
        """
        frame = {}
        for k in self.frame_load_str:            
            item_path = get_path_by_frame(frame_path, view_id, k, 'png')                        
            frame[k] = imread_rgb_torch(item_path, Size=self.img_size).type(torch.FloatTensor)                        
            # frame[k] = imread_rgb_torch('/workspace/Data/shapenet_sample/img_choy2016/000.jpg', Size=self.img_size).type(torch.FloatTensor)                        
            frame[k] /= 255.0
        

        return frame

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
    
    def load_occupancies(self, frame_path):
        """
        load the occupancy data for the 2nd part(IF-Net)
            coords: position to be queried
            occupancies: ground truth for position to be queried
        """
        data_dir = os.path.dirname(frame_path)
        file_name = os.path.basename(frame_path)
        index_of_frame = find_frame_num(file_name)
        index_of_frame = str(int(index_of_frame) // self.pose_num * self.pose_num).zfill(8)

        gt_mesh_path = os.path.join(data_dir, "frame_" + index_of_frame + '_' +\
                                                    "isosurf_scaled.off")

        points = []
        coords = []
        occupancies = []
                
        # uniform_sample_path = required_path.replace('color00.png', "uni_sampled.npz")
        uniform_sample_path = os.path.join(data_dir, "frame_" + index_of_frame + '_' + "uni_sampled.npz")
        if os.path.exists(uniform_sample_path):
            uniform_points = np.load(uniform_sample_path)['points']
            uniform_occ = np.load(uniform_sample_path)['occupancies']

            # pointsf = uniform_points[uniform_occ]
            # write_off("/workspace/z.xyz", pointsf)
            # write_off("/workspace/check/z_{}.xyz".format(time.time()), pointsf)
            # exit()
        else:
            # generate online
            # print("MESH IS ", gt_mesh_path)            
            mesh = trimesh.load(gt_mesh_path)
            uniform_points = np.random.rand(100000, 3)
            uniform_points = 1.1 * (uniform_points - 0.5)
            uniform_occ = iw.implicit_waterproofing(mesh, uniform_points)[0]
            
            # pointsf = uniform_points[uniform_occ]
            # write_off("/workspace/check/z_{}.xyz".format(time.time()), pointsf)
            # exit()
            np.savez(uniform_sample_path, points=uniform_points,
                     occupancies=uniform_occ)

        sample_indices = np.random.randint(0, len(uniform_points), self.num_sample_points)
                        
        points.extend(uniform_points[sample_indices])
        occupancies.extend(uniform_occ[sample_indices])

        
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