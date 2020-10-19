import os
import sys
import numpy as np
import torch
from SAPIENDataset import SAPIENDataset


class MVSPDataset(SAPIENDataset):

    def __init__(self,config):
        super().__init__(config)
        self.view_num = config.VIEW_NUM # number of cameras per frame

        

