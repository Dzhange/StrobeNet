import os, sys, argparse, zipfile, glob, random, pickle, math
import numpy as np
import torch
from itertools import groupby
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
from models.SAPIENDataset import SAPIENDataset
from utils.DataUtils import *


class MVSPDataset(SAPIENDataset):

    def __init__(self,config):
        super().__init__(config)
        self.view_num = config.VIEW_NUM # number of cameras per frame

        
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
            all_frames = [os.path.join(os.path.dirname(p), "frame_" + self.findFrameNum(p)) for p in all_color_imgs]
            all_frames = list(dict.fromkeys(all_frames)) # discard duplicated frame ids
            all_frames.sort()
            self.frame_ids = all_frames
            with open(glob_cache, 'wb') as fp:
                for string in self.frame_load_str:
                    pickle.dump(self.frame_ids, fp)
    
    def __getitem__(self, idx):
        
        data_list = []
        batch = {}
        frame_base_path = self.frame_ids[idx]

        for view in range(self.view_num):
            data = self.get_sv_data(frame_base_path, view)                        
            data_list.append(data)            
        for k in data_list[0].keys():
            batch[k] = [item[k] for item in data_list]

        return batch

    def get_sv_data(self, frame_path, view):
        data = self.load_images(frame_path, view)
        occ = self.load_occupancies(frame_path)        
        data.update(occ)
        return data

    def load_images(self, frame_path, view_id):
        """
        actual implementation of __getitem__
        """                
        frame = {}
        has_mask = False
        for k in self.frame_load_str:
            item_path = get_path_by_frame(frame_path, view_id, k, 'png')
            if "linkseg" in k:                
                frame[k] = imread_gray_torch(item_path, Size=self.img_size)\
                    .type(torch.FloatTensor).unsqueeze(0) # other wise would all be zero # 1,W,H
                continue # no need to be divided by 255
            else:
                frame[k] = imread_rgb_torch(item_path, Size=self.img_size).type(torch.FloatTensor)
            # we only create one mask create 
            if (k == "nox00" or k == "pnnocs00") and not has_mask:
                frame[k] = torch.cat((frame[k], createMask(frame[k])), 0).type(torch.FloatTensor)
                has_mask = True
            frame[k] /= 255.0                
        ##################################################################
        # deal with poses        
        cur_pose_path = get_path_by_frame(frame_path, view_id, "pose", "txt")
        cur_pose = torch.Tensor(np.loadtxt(cur_pose_path))
        # extend dimension in case of single joint case
        if len(cur_pose.shape) == 1:
            cur_pose = cur_pose.unsqueeze(0)
        # project the joint location to eliminate unnecessary joint variation
        if self.projection:
            if "laptop" in self.dataset_dir:
                cur_pose[:, 1] = 0                
        # Create map for pixel-wise supervisoin of pose        
        joint_map = self.gen_joint_map(cur_pose, self.img_size)
        
        frame['joint_map'] = joint_map
        frame['pose'] = cur_pose
        
        mesh_path = frame_path + '_wt_mesh.obj'
        frame['mesh'] = mesh_path
        
        return frame
        
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

        points = []
        coords = []
        occupancies = []
        for i, num in enumerate(self.num_samples):
            
            boundary_samples_path = os.path.join(data_dir, "frame_" + index_of_frame + '_' +\
                                                    self.occ_load_str[i] + '.npz')            
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        gt_mesh_path = os.path.join(data_dir, "frame_" + index_of_frame + '_' +\
                                                    "isosurf_scaled.off")
                        
        # None of the if-data would be needed if in validation mode
        if_data = {
            'grid_coords':np.array(coords, dtype=np.float32),
            'occupancies': np.array(occupancies, dtype=np.float32),            
            'mesh': gt_mesh_path
            }

        if self.config.TRANSFORM:
            transform_path = os.path.join(data_dir, "frame_" + index_of_frame + '_transform.npz')        
            nocs_transform = {}
            nocs_transform['translation'] = np.load(transform_path)['translation']
            nocs_transform['scale'] = np.load(transform_path)['scale']
            
            if_data['translation'] = nocs_transform['translation']
            if_data['scale'] = nocs_transform['scale']

        return if_data


