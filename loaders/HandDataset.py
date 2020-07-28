"""
Loader for HandRigDataSet
"""
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby

import numpy as np
import json
import matplotlib.pyplot as plt

import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from utils.DataUtils import *

class HandDataset(torch.utils.data.Dataset):
    """
    A basic dataloader for HandRigDatasetv3
    Most codes copied from HandRigDatasetV3 from Srinath
    """
    def __init__(self, Root, Train=True, Transform=None,
                 ImgSize=(320, 240), Limit=100, FrameLoadStr=None, Required='color00'):        
        self.NCameras = 10
        self.FileName = 'hand_rig_dataset_v3.zip'        
        self.FrameLoadStr = ['color00', 'color01', 'normals00', 'normals01',\
                        'nox00', 'nox01', 'pnnocs00', 'pnnocs01',\
                        'uv00', 'uv01'] if FrameLoadStr is None else FrameLoadStr
        
        self.init(Root, Train, Transform, ImgSize, Limit, self.FrameLoadStr, Required)
        self.LoadData()

    def init(self, Root, Train=True, Transform=None,
             ImgSize=(320, 240), Limit=100, FrameLoadStr=None, Required='VertexColors'):
        self.DataRoot = Root
        self.DatasetDir = self.DataRoot
        self.isTrainData = Train
        self.Transform = Transform
        self.ImageSize = ImgSize
        self.Required = Required
        self.FrameLoadStr = FrameLoadStr
        if Limit <= 0.0 or Limit > 100.0:
            raise RuntimeError('Data limit percent has to be >0% and <=100%')
        self.DataLimit = Limit

        if self.Required not in self.FrameLoadStr:
            raise RuntimeError('FrameLoadStr should contain {}.'.format(self.Required))                
        if os.path.exists(self.DatasetDir) == False:
            print("Dataset {} doesn't exist".format(self.DatasetDir))
            exit()

    def __len__(self):
        return len(self.FrameFiles[self.FrameLoadStr[0]])

    def __getitem__(self, idx):
        RGB, LoadTup = self.loadImages(idx)
        LoadIms = torch.cat(LoadTup, 0)
        # print(RGB,RGB.shape)
        return RGB, LoadIms

    def LoadData(self):
        self.FrameFiles = {}

        if self.isTrainData:
            FilesPath = os.path.join(self.DatasetDir, 'train/')
        else:
            FilesPath = os.path.join(self.DatasetDir, 'val/')

        # Load index for data 
        CameraIdxStr = '*'                
        GlobPrepend = '_'.join(str(i) for i in self.FrameLoadStr)
        GlobCache = os.path.join(FilesPath, 'all_glob_' + GlobPrepend + '.cache')
        if os.path.exists(GlobCache):
            # use pre-cached index
            print('[ INFO ]: Loading from glob cache:', GlobCache)
            with open(GlobCache, 'rb') as fp:
                for Str in self.FrameLoadStr:
                    self.FrameFiles[Str] = pickle.load(fp)
        else:
            # glob and save cache
            print('[ INFO ]: Saving to glob cache:', GlobCache)
            for Str in self.FrameLoadStr:
                self.FrameFiles[Str] = glob.glob(FilesPath + '/**/frame_*_view_' + CameraIdxStr + '_' + Str + '.*')
                self.FrameFiles[Str].sort()
            with open(GlobCache, 'wb') as fp:
                for Str in self.FrameLoadStr:
                    pickle.dump(self.FrameFiles[Str], fp)

        FrameFilesLengths = []
        for K, CurrFrameFiles in self.FrameFiles.items():
            if not CurrFrameFiles:
                raise RuntimeError('[ ERR ]: None data for', K)
            if len(CurrFrameFiles) == 0:
                raise RuntimeError('[ ERR ]: No files found during data loading for', K)
            FrameFilesLengths.append(len(CurrFrameFiles))

        if len(set(FrameFilesLengths)) != 1:
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match', FrameFilesLengths)

        TotSize = len(self)
        DatasetLength = math.ceil((self.DataLimit / 100) * TotSize)
        print('[ INFO ]: Loading {} / {} items.'.format(DatasetLength, TotSize))

        for K in self.FrameFiles:
            self.FrameFiles[K] = self.FrameFiles[K][:DatasetLength]

    def loadImages(self, idx):
        Frame = {}

        for K in self.FrameFiles:                   
            Frame[K] = imread_rgb_torch(self.FrameFiles[K][idx], Size=self.ImageSize).type(torch.FloatTensor)
            if K != self.Required:
                Frame[K] = torch.cat((Frame[K], createMask(Frame[K])), 0).type(torch.FloatTensor)
            if self.Transform is not None:
                Frame[K] = self.Transform(Frame[K])
            # Convert range to 0.0 - 1.0
            Frame[K] /= 255.0

        GroupedFrameStr = [list(i) for j, i in groupby(self.FrameLoadStr, lambda a: ''.join([i for i in a if not i.isdigit()]))]

        LoadTup = ()
        # Concatenate any peeled outputs
        for Group in GroupedFrameStr:            
            Concated = ()
            for FrameStr in Group:                
                if 'color00' in FrameStr: # Append manually
                    continue
                Concated = Concated + (Frame[FrameStr],)
            if len(Concated) > 0:
                LoadTup = LoadTup + (torch.cat(Concated, 0), )        

        return Frame['color00'], LoadTup
