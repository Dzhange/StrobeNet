import cv2
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import re
import os

cache_path = '/workspace/Data/SAPIEN/laptop/mv_laptop_500_4_IF/val/all_glob_frames.cache'
cnt = 0

# for f in frames:
#     for view in ['00', '01', '02','03']:
#         item_path = f + '_view_' + view +  '_linkseg.png'
#         print(item_path)
#         img = cv2.imread(item_path, -1)
#         if img.shape[-1] == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = np.where(img==2, 1, img)
#             img = np.where(img==3, 2, img)
#         cv2.imwrite(item_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def recolor(frame_path):    
    # for view in ['00', '01', '02', '03']:
    # for view in ['00']:
        item_path = frame_path
        # print(item_path)
        img = cv2.imread(item_path, -1)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.where(img==2, 1, img)
            img = np.where(img==3, 2, img)
        cv2.imwrite(item_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def move_unwanted_frames(path):
    frame_num = re.findall(r'%s(\d+)' % "frame_", path)[0]
    if int(frame_nunm) > 1000:
        os.remove(path)
if __name__ == "__main__":    
    # with open(cache_path, 'rb') as fp:
        # frames = pickle.load(fp)
    import glob
    frames = glob.glob("/workspace/Data/SAPIEN/laptop/laptop_all_200_IF/*/*/*linkseg.png")      
    p = Pool(mp.cpu_count())
    p.map(recolor, frames)