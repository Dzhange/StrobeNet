"""
Loader for HandRigDataSet
"""
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby
import re
import time
import numpy as np
import torch
import torch.utils.data
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
from models.loss import L2MaskLoss, L2Loss, LBSLoss
from utils.DataUtils import *



class SAPIENDataset(torch.utils.data.Dataset):
    """
    A Dataloader for SAPIEN dataset:
    Items to load:
        1. Color Images
        2. NOCS
        3. PNNOCS
        4. POSE
        5. Segmentation
        5. Occupancy
    """
    def __init__(self, root, train=True, transform=None,
                 img_size=(320, 240), limit=10, frame_load_str=None, required='color00', rel=False):
        
        self.num_cameras = 10        
        self.frame_load_str = ['color00', 'nocs00', 'pnnocs00', 'linkseg'] \
            if frame_load_str is None else frame_load_str

        self.init(root, train, transform, img_size, limit, self.frame_load_str, required, rel=rel)
        self.load_data()

    def init(self, root, train=True, transform=None,
             img_size=(320, 240), limit=100, frame_load_str=None, required='VertexColors', rel=False):
        
        self.dataset_dir = root
        self.is_train_data = train
        self.transform = transform
        self.img_size = img_size
        self.required = required
        self.frame_load_str = frame_load_str
        
        # For occupancies
        self.occ_load_str = ['boundary_0.1_samples', 'boundary_0.01_samples']
        self.sample_distribt = np.array([0.5, 0.5])
        self.num_sample_points = 50000
        self.num_samples = np.rint(self.sample_distribt * self.num_sample_points).astype(np.uint32)

        self.data_offset = 0
        # self.data_offset = 2000

        if limit <= 0.0 or limit > 100.0:
            raise RuntimeError('Data limit percent has to be >0% and <=100%')
        self.data_limit = limit
        if self.required not in self.frame_load_str:
            raise RuntimeError('frame_load_str should contain {}.'.format(self.required))

        ######### Add BoneWeights #########
        self.bone_num = 16

        ###### Change Boneweight into Segmentation map ######        
        # self.as_seg = True
        self.as_seg = False
        
        ###### Uniformly Sample Dataset? ######
        # self.shuffle_in_limit = True
        self.shuffle_in_limit = False
                    
        if os.path.exists(self.dataset_dir) == False:
            print("Dataset {} doesn't exist".format(self.dataset_dir))
            exit()

    def __len__(self):
        return len(self.frame_files[self.frame_load_str[0]])

    def __getitem__(self, idx):
        required_path = self.frame_files[self.required][idx]
        RGB, target_imgs, pose, mesh_path = self.load_images(idx)        
        load_imgs = torch.cat(target_imgs, 0)
        occ_data = self.load_occupancies(required_path)

        return RGB, load_imgs, pose, occ_data, mesh_path    

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
        # camera_idx_str = '00'
        prepend_list = []
        for i in self.frame_load_str:
            prepend_list.append(i)

        prepend_list.append("sapien")
        glob_prepend = '_'.join(prepend_list)
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
            # DatasetLength = 10
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

            for k in self.frame_files:
                if self.data_limit > 10:
                    self.frame_files[k] = self.frame_files[k][:dataset_length]
                else:
                    # offset only works for overfitting task
                    self.frame_files[k] = self.frame_files[k][self.data_offset:self.data_offset+dataset_length]

    def load_images(self, idx):
        """
        actual implementation of __getitem__
        """
        typical_path = self.frame_files['color00'][idx]

        curdir = os.path.dirname(typical_path)
        file_name = os.path.basename(typical_path)
        idx_of_frame = find_frame_num(file_name)
                
        cur_pose_path = os.path.join(curdir, "frame_" + idx_of_frame + '_curr_pose.txt')
        cano_pose_path = os.path.join(curdir, "frame_" + idx_of_frame + '_cano_pose.txt')        
        
        # angles in poses are in radians format
        cur_pose = torch.Tensor(np.loadtxt(cur_pose_path))
        cano_pose = torch.Tensor(np.loadtxt(cano_pose_path))
        frame = {}
        for k in self.frame_files:
            if "BoneWeight" in k:
                frame[k] = imread_gray_torch(self.frame_files[k][idx], Size=self.img_size)\
                    .type(torch.FloatTensor).unsqueeze(0) # 1,W,H
            else:
                frame[k] = imread_rgb_torch(self.frame_files[k][idx], Size=self.img_size).type(torch.FloatTensor)
            if k == "nox00":
                frame[k] = torch.cat((frame[k], createMask(frame[k])), 0).type(torch.FloatTensor)            
            frame[k] /= 255.0

        grouped_frame_str = [list(i) for j, i in groupby(self.frame_load_str,\
                                lambda a: ''.join([i for i in a if not i.isdigit()]))]

        load_tuple = ()
        # Concatenate any peeled outputs
        for group in grouped_frame_str:
            concated = ()
            for frame_str in group:
                if 'color00' in frame_str: # Append manually
                    continue
                concated = concated + (frame[frame_str],)
            if len(concated) > 0:
                load_tuple = load_tuple + (torch.cat(concated, 0), )

        # faster!
        img_shape = load_tuple[0].shape
        joint_map = torch.Tensor(cur_pose.shape[0]*6, img_shape[1], img_shape[2])

        cnt = 0
        for i in range(cur_pose.shape[0]):
            for j in range(3):                
                joint_map[cnt] = joint_map[cnt].fill_(cur_pose[i, j])
                cnt += 1
        for i in range(cur_pose.shape[0]):
            for j in range(3, 6):                
                joint_map[cnt] = joint_map[cnt].fill_(cur_pose[i, j])
                cnt += 1

        load_tuple = load_tuple + (joint_map, )
        load_tuple = (load_tuple[0], load_tuple[2], load_tuple[1])

        mesh_path = os.path.join(curdir, "frame_" + idx_of_frame + '_wt_mesh.obj')

        return frame['color00'], load_tuple, cur_pose, mesh_path

    def load_occupancies(self, required_path):
        """
        load the occupancy data for the 2nd part(IF-Net)
            coords: position to be queried
            occupancies: ground truth for position to be queried
        """
        data_dir = os.path.dirname(required_path)
        file_name = os.path.basename(required_path)
        index_of_frame = find_frame_num(file_name)
        transform_path = os.path.join(data_dir, "frame_" + index_of_frame + '_transform.npz')

        nocs_transform = {}
        nocs_transform['translation'] = np.load(transform_path)['translation']
        nocs_transform['scale'] = np.load(transform_path)['scale']

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
            'translation': nocs_transform['translation'],
            'scale': nocs_transform['scale'],
            'mesh': gt_mesh_path
            }
        return if_data    

    # def check_map(self, tar_joint_map, out_mask, tar_joints):
    #     n_batch = tar_joint_map.shape[0]
    #     bone_num = 16
    #     tar_joint_map = tar_joint_map.reshape(n_batch, bone_num, 3, tar_joint_map.shape[2],
    #                                             tar_joint_map.shape[3])  # B,bone_num,3,R,R
    #     tar_joint_map = tar_joint_map * out_mask.unsqueeze(1).unsqueeze(1)
    #     pred_joint = tar_joint_map.reshape(n_batch, bone_num, 3, -1).mean(dim=3)  # B,22,3
    #     # print(pred_joint.shape, tar_joints.shape)
    #     joint_diff = torch.sum((pred_joint - tar_joints) ** 2, dim=2)  # B,22
    #     joint_loc_loss = joint_diff.sum() / (n_batch * tar_joint_map.shape[1])
    #     return joint_loc_loss

if __name__ == '__main__':
    import argparse
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HandRigDatasetV2', required=True)

    Args, _ = Parser.parse_known_args()

    Data = SAPIENDataset(root=Args.data_dir, train=True, frame_load_str=["color00", "nox00"])
    # Data.saveItem(random.randint(0, len(Data)))
    # Data.visualizeRandom(10, nColsPerSam=len(Data.FrameLoadStr)-1) # Ignore Pose
    # exit()
    # LossUnitTest = GenericImageDataset.L2MaskLoss(0.7)
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=4, shuffle=True, num_workers=0)
    # loss = LBSLoss()
    for i, Data in enumerate(DataLoader, 0):  # Get each batch
        target_img = Data[1]
        