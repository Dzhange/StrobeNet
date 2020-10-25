import cv2
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import re
import os

cache_path = '/workspace/Data/SAPIEN/laptop/mv_laptop_500_4_IF/val/all_glob_frames.cache'
cnt = 0


def remove(frame_path):
    print(frame_path)    
    os.remove(frame_path)

# def move_unwanted_frames(path):
#     frame_num = re.findall(r'%s(\d+)' % "frame_", path)[0]
#     if int(frame_nunm) > 1000:
#         os.remove(path)
if __name__ == "__main__":    
    # with open(cache_path, 'rb') as fp:
        # frames = pickle.load(fp)
    import glob
    frames = glob.glob("/workspace/Data/SAPIEN/oven/oven_mv1/train/0008/*")      
    p = Pool(mp.cpu_count() >> 2)
    # p.map(recolor, frames)
    print(frames)
    p.map(remove, frames)