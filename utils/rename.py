import glob
import shutil

root = "/workspace/Data/SAPIEN/nonrigid_eyeglasses_IF/"

all_nocs = glob.glob(root + "*/*/*pnnocs.png")
print(all_nocs)
for nocs in all_nocs:
    new = nocs.replace("pnnocs.png", "pnnocs00.png")
    shutil.move(nocs,new)
