import glob
import shutil

root = "/workspace/Data/SAPIEN/eyeglasses/pose_aa"

all_nocs = glob.glob(root + "/*/*/*final_pose*")
print(len(all_nocs))
for nocs in all_nocs:
    new = nocs.replace("final_pose", "pose")
    shutil.move(nocs,new)
