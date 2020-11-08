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


# def recolor(frame_path):    
#     # for view in ['00', '01', '02', '03']:
#     # for view in ['00']:
#         item_path = frame_path
#         # print(item_path)
#         img = cv2.imread(item_path, -1)
#         if img.shape[-1] == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = np.where(img==2, 1, img)
#             img = np.where(img==3, 2, img)
#         cv2.imwrite(item_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


# def move_unwanted_frames(path):
#     frame_num = re.findall(r'%s(\d+)' % "frame_", path)[0]
#     if int(frame_nunm) > 1000:
#         os.remove(path)

# def simplify(path):
#     if "split" not in path:
#         # command = "{} {} {} 5000".format("/workspace/Manifol/build/manifold", path, path)
#         mt = "meshlabserver -i {} -o {} -m wt".format(path, path)
#         command = "xvfb-run -a -s '-screen 0 1000x1000x24' meshlabserver -i {} -o {} -s {}".format(path, path, './utils/meshlab_scripts/simplify.mlx')
#         os.system(mt)
#         os.system(command)

def crt(frame):
    v = np.loadtxt(frame)
    if v.shape[0] == 0:
        os.system("rm {}".format(frame))
        os.system("cp {} {}".format(frame.replace('_uni', ''), frame ))

if __name__ == "__main__":
    # with open(cache_path, 'rb') as fp:
        # frames = pickle.load(fp)
    import glob
    frames = glob.glob("/workspace/Data/SAPIEN/benchmark/glasses/mv8_fix_scale_mpf_nobg_uni/*/*/*pose*")
    p = Pool(mp.cpu_count())
    # p.map(simplify, frames)
    for i in range(len(frames)):
        # print(frames[i])
        # simplify(frames[i])
        crt(frames[i])