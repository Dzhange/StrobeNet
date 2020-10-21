import cv2
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool

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
    for view in ['00', '01', '02', '03']:
        item_path = frame_path + '_view_' + view +  '_linkseg.png'
        # print(item_path)
        img = cv2.imread(item_path, -1)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.where(img==2, 1, img)
            img = np.where(img==3, 2, img)
        cv2.imwrite(item_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


if __name__ == "__main__":    
    with open(cache_path, 'rb') as fp:
        frames = pickle.load(fp)

    p = Pool(mp.cpu_count())
    p.map(recolor, frames)