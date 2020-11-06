import cv2
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import re
import os
import argparse

cache_path = '/workspace/Data/SAPIEN/laptop/mv_laptop_500_4_IF/val/all_glob_frames.cache'
cnt = 0


def remove(path):
    # print(path)    
    os.remove(path)

# def move_unwanted_frames(path):
#     frame_num = re.findall(r'%s(\d+)' % "frame_", path)[0]
#     if int(frame_nunm) > 1000:
#         os.remove(path)
if __name__ == "__main__":    
    # with open(cache_path, 'rb') as fp:
        # frames = pickle.load(fp)
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target-dir', help='Specify the location of the directory to download and store HandRigDatasetV2', required=True)
    
    Args, _ = parser.parse_known_args()

    paths = glob.glob(os.path.join(Args.target_dir, "*/*"))
    p = Pool(mp.cpu_count() >> 1)
    # p.map(recolor, frames)
    print(len(paths))
    p.map(remove, paths)