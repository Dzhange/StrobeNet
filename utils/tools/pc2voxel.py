import neuralnet_pytorch as nnt
import torch as T
from torch_scatter import scatter_add


def pointcloud2voxel_fast(pc: T.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    b, n, _ = pc.shape
    # b, _, n = pc.shape
    half_size = grid_size / 2.
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = T.all(valid, 2)
    pc_grid = (pc + half_size) * (voxel_size - 1.)
    indices_floor = T.floor(pc_grid)
    indices = indices_floor.long()
    batch_indices = T.arange(b).to(pc.device)
    batch_indices = nnt.utils.shape_padright(batch_indices)
    
    
    # pip install "neuralnet-pytorch[option] @ git+git://github.com/justanhduc/neuralnet-pytorch.git@master"
    batch_indices = nnt.utils.tensor_utils.tile(batch_indices, (1, n))
    batch_indices = nnt.utils.shape_padright(batch_indices)
    # batch_indices = batch_indices.view(b,1,n)f

    # export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    indices = T.cat((batch_indices, indices), 2)
    indices = T.reshape(indices, (-1, 4))

    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = T.tensor([[0] + pos]).to(pc.device)
        indices_loc = indices + indices_shift
        out_shape = (b,) + (voxel_size,) * 3
        out = T.zeros(*out_shape).to(pc.device).flatten()
        voxels = scatter_add(updates, nnt.utils.ravel_index(indices_loc.t(), out_shape), out=out).view(*out_shape)
        
        # voxels.requires_grad = True
        return voxels

    voxels = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
    voxels = sum(voxels)
    voxels = T.clamp(voxels, 0., 1.)
    return voxels


def voxelize(pc, vox_size=32):
    vox = pointcloud2voxel_fast(pc, vox_size)
    
    vox = T.clamp(vox, 0., 1.)
    vox = T.squeeze(vox)
    return vox


def iou(pred, gt, th=.5):
    pred = pred > th
    gt = gt > th
    intersect = T.sum(pred & gt).float()
    union = T.sum(pred | gt).float()
    iou_score = intersect / (union + 1e-8)
    return iou_score


def batch_iou(bpc1, bpc2, voxsize=32, thres=.4):
    def _iou(pc1, pc2):
        pc1 = pc1 - T.mean(pc1, -2, keepdim=True)
        pc1 = voxelize(pc1[None], voxsize)

        pc2 = pc2 - T.mean(pc2, -2, keepdim=True)
        pc2 = voxelize(pc2[None], voxsize)
        return iou(pc1, pc2, thres)

    total = map(_iou, bpc1, bpc2)
    return sum(total) / len(bpc1)