"""
Loader for HandRigDataSet
"""
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby

import numpy as np
import torch
import torch.utils.data
from utils.DataUtils import *
from models.loss import L2MaskLoss, L2Loss, LBSLoss
import trimesh

class HandOccDataset(torch.utils.data.Dataset):
    """
    A basic dataloader for HandOccDataset
    """
    def __init__(self, root, train=True, transform=None,
                 img_size=(320, 240), limit=100, frame_load_str=None, required='color00'):
        self.num_cameras = 10
        self.frame_load_str = ['color00', 'color01', 'normals00', 'normals01',\
                        'nox00', 'nox01', 'pnnocs00', 'pnnocs01',\
                        'uv00', 'uv01'] if frame_load_str is None else frame_load_str

        self.init(root, train, transform, img_size, limit, self.frame_load_str, required)
        self.load_data()

    def init(self, root, train=True, transform=None,
             img_size=(320, 240), limit=100, frame_load_str=None, required='VertexColors'):
        self.dataset_dir = root
        self.is_train_data = train
        self.transform = transform
        self.img_size = img_size
        self.required = required
        self.frame_load_str = frame_load_str

        self.occ_load_str = ['boundary_0.1_samples','boundary_0.01_samples']
        self.sample_distribt = np.array([0.5, 0.5])
        self.num_sample_points = 50000
        self.num_samples = np.rint(self.sample_distribt * self.num_sample_points).astype(np.uint32)

        self.shuffle_in_limit = True

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

        RGB, load_tup = self.load_images(idx)
        load_imgs = torch.cat(load_tup, 0)

        # if self.is_train_data:
        occ_data = self.load_occupancies(required_path)
        return RGB, load_imgs, occ_data

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
            self.frame_files[k] = self.frame_files[k][:dataset_length]

    def load_images(self, idx):
        """
        actual implementation of __getitem__
        """
        frame = {}
        for k in self.frame_files:
            frame[k] = imread_rgb_torch(self.frame_files[k][idx], Size=self.img_size).type(torch.FloatTensor)
            if k != self.required:
                frame[k] = torch.cat((frame[k], createMask(frame[k])), 0).type(torch.FloatTensor)
            if self.transform is not None:
                frame[k] = self.transform(frame[k])
            # Convert range to 0.0 - 1.0
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

        return frame['color00'], load_tuple

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


if __name__ == '__main__':
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HandRigDatasetV2', required=True)
    
    Args, _ = Parser.parse_known_args()

    Data = HandOccDataset(root=Args.data_dir, train=True, frame_load_str=["color00", "nox00"])
    
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=4, shuffle=True, num_workers=0)
    loss = LBSLoss()
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        # DataTD = ptUtils.sendToDevice(Targets, 'cpu')
        # print('Data size:', Data.size())
        print(Targets[1].shape)
        
        l = loss(Targets[0], Targets)
        print(l)
        