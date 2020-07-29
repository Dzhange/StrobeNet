"""
Loader for HandRigDataSet
"""
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby

import numpy as np
import torch
import torch.utils.data.Dataset
from utils.DataUtils import *

class HandDataset(torch.utils.data.Dataset):
    """
    A basic dataloader for HandRigDatasetv3
    Most codes copied from HandRigDatasetV3 from Srinath
    """
    def __init__(self, root, Train=True, Transform=None,
                 ImgSize=(320, 240), Limit=100, frame_load_str=None, required='color00'):        
        self.num_cameras = 10
        self.file_name = 'hand_rig_dataset_v3.zip'
        self.frame_load_str = ['color00', 'color01', 'normals00', 'normals01',\
                        'nox00', 'nox01', 'pnnocs00', 'pnnocs01',\
                        'uv00', 'uv01'] if frame_load_str is None else frame_load_str

        self.init(root, Train, Transform, ImgSize, Limit, self.frame_load_str, required)
        self.load_data()

    def init(self, root, train=True, transform=None,
             img_size=(320, 240), limit=100, frame_load_str=None, required='VertexColors'):
        self.dataset_dir = root
        self.is_train_data = train
        self.transform = transform
        self.image_size = img_size
        self.required = required
        self.frame_load_str = frame_load_str
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
        RGB, load_tup = self.loadImages(idx)
        load_imgs = torch.cat(load_tup, 0)
        # print(RGB,RGB.shape)
        return RGB, load_imgs

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

        total_size = len(self)
        dataset_length = math.ceil((self.data_limit / 100) * total_size)
        print('[ INFO ]: Loading {} / {} items.'.format(dataset_length, total_size))

        for k in self.frame_files:
            self.frame_files[k] = self.frame_files[k][:dataset_length]

    def loadImages(self, idx):
        frame = {}

        for k in self.frame_files:
            frame[k] = imread_rgb_torch(self.frame_files[k][idx], Size=self.image_size).type(torch.FloatTensor)
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
