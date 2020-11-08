import cv2
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import re
import os, shutil
import argparse

cache_path = '/workspace/Data/SAPIEN/laptop/mv_laptop_500_4_IF/val/all_glob_frames.cache'
cnt = 0


def remove(path):
    # print(path)
    os.remove(path)

# def subdir_glob(dir):
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

    paths = []
    all_subdirs = glob.glob(os.path.join(
                    Args.target_dir, '*'))
    
    print(all_subdirs)
    if len(all_subdirs) < 10:
        all_subdirs = glob.glob(os.path.join(Args.target_dir, "*/*"))
    all_subdirs.sort()
    paths = []
    for subdir in all_subdirs:
        print("globing on ", subdir)
        cur_items = glob.glob(os.path.join(
            subdir, "*"))
        cur_items.extend(glob.glob(os.path.join(
            subdir, "*/*")))
        paths.extend(cur_items)
        print(len(paths))


    # paths.extend(glob.glob(os.path.join(Args.target_dir, "*")))
    # paths.extend(glob.glob(os.path.join(Args.target_dir, "*/*")))
    # paths.extend(glob.glob(os.path.join(Args.target_dir, "*/*/*")))
    
    
    p = Pool(mp.cpu_count()*4)
    # p.map(recolor, frames)
    
    p.map(remove, paths)
    shutil.rmtree(Args.target_dir)