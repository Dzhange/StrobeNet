"""
Loader for HandRigDataSet
"""
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby
import re
import numpy as np
import torch
import torch.utils.data
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
from models.loss import L2MaskLoss,L2Loss,LBSLoss
from utils.DataUtils import *

class HandDatasetLBS(torch.utils.data.Dataset):
    """
    A basic dataloader for HandRigDatasetv3
    Most codes copied from HandRigDatasetV3 from Srinath
    """
    def __init__(self, root, train=True, transform=None,
                 img_size=(320, 240), limit=10, frame_load_str=None, required='color00'):        
        self.num_cameras = 10
        self.file_name = 'hand_rig_dataset_v3.zip'
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
        if limit <= 0.0 or limit > 100.0:
            raise RuntimeError('Data limit percent has to be >0% and <=100%')
        self.data_limit = limit
        if self.required not in self.frame_load_str:
            raise RuntimeError('frame_load_str should contain {}.'.format(self.required))
        
        ######### Add BoneWeights #########
        self.bone_num = 16
        for i in range(self.bone_num):
            self.frame_load_str.append("BoneWeight00bone_" + str(i))            
            # self.frame_load_str.append("BoneWeight01bone_" + str(i))
            
        if os.path.exists(self.dataset_dir) == False:
            print("Dataset {} doesn't exist".format(self.dataset_dir))
            exit()

    def __len__(self):
        return len(self.frame_files[self.frame_load_str[0]])

    def __getitem__(self, idx):
        RGB, load_tup = self.load_images(idx)
        load_imgs = torch.cat(load_tup[0], 0)
        # print(RGB,RGB.shape)
        return RGB, (load_imgs, load_tup[1])

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
        prepend_list = []
        for i in self.frame_load_str:
            if "Bone" not in i:
                prepend_list.append(i)
        prepend_list.append("LBS")
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

        total_size = len(self)
        dataset_length = math.ceil((self.data_limit / 100) * total_size)
        print('[ INFO ]: Loading {} / {} items.'.format(dataset_length, total_size))

        for k in self.frame_files:
            self.frame_files[k] = self.frame_files[k][:dataset_length]

    def load_images(self, idx):
        """
        actual implementation of __getitem__
        """
        typical_path = self.frame_files['color00'][idx]
        # print('Typical is ',TypicalPath)
        dir = os.path.dirname(typical_path)
        file_name = os.path.basename(typical_path)
        idx_of_frame = find_frame_num(file_name)
        rel_pose_path = os.path.join(dir, "frame_" + idx_of_frame + '_hpose_rel.txt')
        
        pose = torch.Tensor(np.loadtxt(rel_pose_path))
        
        frame = {}
        for k in self.frame_files:            
            if "BoneWeight" in k:
                frame[k] = imread_gray_torch(self.frame_files[k][idx], Size=self.img_size).type(torch.FloatTensor).unsqueeze(0)
            else:
                frame[k] = imread_rgb_torch(self.frame_files[k][idx], Size=self.img_size).type(torch.FloatTensor)
            
            if k == "nox00":                 
                frame[k] = torch.cat((frame[k], createMask(frame[k])), 0).type(torch.FloatTensor)
            # if self.transform is not None:
            #     frame[k] = self.transform(frame[k])
            # # Convert range to 0.0 - 1.0
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
        
        return frame['color00'], (load_tuple, pose)

if __name__ == '__main__':
    import argparse
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-d', '--data-dir', help='Specify the location of the directory to download and store HandRigDatasetV2', required=True)
    
    Args, _ = Parser.parse_known_args()

    Data = HandDatasetLBS(root=Args.data_dir, train=True, frame_load_str=["color00", "nox00"])
    # Data.saveItem(random.randint(0, len(Data)))
    # Data.visualizeRandom(10, nColsPerSam=len(Data.FrameLoadStr)-1) # Ignore Pose
    # exit()
    # LossUnitTest = GenericImageDataset.L2MaskLoss(0.7)
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=4, shuffle=True, num_workers=0)
    loss = LBSLoss()
    for i, (Data, Targets) in enumerate(DataLoader, 0):  # Get each batch
        # DataTD = ptUtils.sendToDevice(Targets, 'cpu')
        # print('Data size:', Data.size())
        print(Targets[1].shape)
        # print('Targets size:', len(Targets))
        # Loss = LossUnitTest(Targets[0], Targets)
        l = loss(Targets[0], Targets)
        print(l)
        # print('Loss:', Loss.item())
        # Loss.backward()