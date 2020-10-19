import os, sys, argparse, zipfile, glob, random, pickle, math
import numpy as np
import torch
from SAPIENDataset import SAPIENDataset


class MVSPDataset(SAPIENDataset):

    def __init__(self,config):
        super().__init__(config)
        self.view_num = config.VIEW_NUM # number of cameras per frame

        
    def load_data(self):
        # load path for all the data needed
        self.frame_files = []

        if self.is_train_data:
            file_path = os.path.join(self.dataset_dir, 'train/')
        else:
            file_path = os.path.join(self.dataset_dir, 'val/')

        glob_cache = os.path.join(file_path, 'all_glob_frames.cache')
        
        if os.path.exists(glob_cache):
            # use pre-cached index
            print('[ INFO ]: Loading from glob cache:', glob_cache)
            with open(glob_cache, 'rb') as fp:                
                self.frame_files[string] = pickle.load(fp)
        else:
            # glob and save cache
            print('[ INFO ]: Saving to glob cache:', glob_cache)
            for string in self.frame_load_str:
                self.frame_files[string] = glob.glob(file_path + '/**/frame_*_view_*_' + string + '.*')
                self.frame_files[string].sort()
            with open(glob_cache, 'wb') as fp:
                for string in self.frame_load_str:
                    pickle.dump(self.frame_files[string], fp)

        # load frame number 
        # get views by frame number 
        

