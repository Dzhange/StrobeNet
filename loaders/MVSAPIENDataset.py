import os, sys, argparse, zipfile, glob, random, pickle, math
import numpy as np
import torch
from itertools import groupby
from sklearn.neighbors import NearestNeighbors

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
from loaders.SAPIENDataset import SAPIENDataset
from utils.DataUtils import *
import time

class MVSPDataset(SAPIENDataset):

    def __init__(self, config, train):
        super().__init__(config, train)
        self.view_num = config.VIEW_NUM # number of cameras per frame
        
        self.rng = np.random.RandomState(0)
        # same set up as from jiahui's code
        self.supervision_cap = 9102 # maximum correspondence, bigger than most cases

    def load_data(self):    
        # we only load valid frameIDS, we do not record 
        # file names as the view num is not specified
        self.frame_ids = []

        if self.is_train_data:
            file_path = os.path.join(self.dataset_dir, 'train/')
        else:
            file_path = os.path.join(self.dataset_dir, 'val/')

        glob_cache = os.path.join(file_path, 'all_glob_frames.cache')
        if os.path.exists(glob_cache):
            # use pre-cached index
            print('[ INFO ]: Loading from glob cache:', glob_cache)
            with open(glob_cache, 'rb') as fp:
                self.frame_ids = pickle.load(fp)
        else:
            # glob and save cache
            print('[ INFO ]: Saving to glob cache:', glob_cache)
            all_color_imgs = glob.glob(os.path.join(file_path, '**/frame_*_color00.*'))
            all_frames = [os.path.join(os.path.dirname(p), "frame_" + find_frame_num(p)) for p in all_color_imgs]
            all_frames = list(dict.fromkeys(all_frames)) # discard duplicated frame ids
            all_frames.sort()
            self.frame_ids = all_frames
            with open(glob_cache, 'wb') as fp:
                for string in self.frame_load_str:
                    pickle.dump(self.frame_ids, fp)

        if self.shuffle_in_limit:
            total_size = len(self)
            dataset_length = math.ceil((self.data_limit / 100) * total_size)
            # dataset_length = 20
            step = int(np.floor(100 / self.data_limit))
            sample_index = []
            cur_idx = 0
            while True:
                if len(sample_index) >= dataset_length or cur_idx >= total_size:
                    break
                sample_index.append(cur_idx)
                cur_idx += step                                    
            print('[ INFO ]: Loading {} / {} items.'.format(len(sample_index), total_size))

            
            self.frame_ids = [self.frame_ids[i] for i in sample_index]
            self.frame_ids.sort()
        else:
            total_size = len(self)
            dataset_length = math.ceil((self.data_limit / 100) * total_size)
            print('[ INFO ]: Loading {} / {} items.'.format(dataset_length, total_size))

        
            if self.data_limit > 10:
                self.frame_ids = self.frame_ids[:dataset_length]
            else:
                # offset only works for overfitting task
                self.frame_ids = self.frame_ids[self.data_offset:self.data_offset+dataset_length]

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):

        data_list = []
        batch = {}


        frame_base_path = self.frame_ids[idx]        
        if self.config.RANDOM_VIEW:
            if self.is_train_data:
                views = np.random.choice(self.config.TOTAL_VIEW, self.view_num, replace=False)
            else:
                views = self.rng.choice(self.config.TOTAL_VIEW, self.view_num, replace=False)            
        else:
            views = list(range(self.view_num))


        
        for view in views:
            data = self.get_sv_data(frame_base_path, view)                        
            data_list.append(data)
        for k in data_list[0].keys():
            batch[k] = [item[k] for item in data_list]                
        # TODO: also include pair-wise consistency data
        
        crr = self.get_crr(batch)
        if self.config.CONSISTENCY != 0:            
            batch.update(crr)

        return batch

    def get_sv_data(self, frame_path, view):
        data = self.load_images(frame_path, view)
        occ = self.load_occupancies(frame_path)

        # pointsf = occ['grid_coords'][occ['occupancies'].astype(np.bool)]
        # write_off("/workspace/check/z_{}.xyz".format(time.time()), pointsf)

        data.update(occ)
        return data

    def load_images(self, frame_path, view_id):
        """
        actual implementation of __getitem__
        """                
        frame = {}
        has_mask = False
        for k in self.frame_load_str:
            item_path = get_path_by_frame(frame_path, view_id, k, 'png')
            # print(item_path)
            # print(item_path)
            if "linkseg" in k:
                frame[k] = imread_gray_torch(item_path, Size=self.img_size)\
                    .type(torch.FloatTensor).unsqueeze(0) # other wise would all be zero # 1,W,H
                continue # no need to be divided by 255
            else:                
                frame[k] = imread_rgb_torch(item_path, Size=self.img_size).type(torch.FloatTensor)
            # we only create one mask create
            if (k == "nox00" or k == "pnnocs00") and not has_mask:
                frame[k] = torch.cat((frame[k], createMask(frame[k])), 0).type(torch.FloatTensor)
                has_mask = True
            frame[k] /= 255.0                
        ##################################################################
        # deal with poses        
        cur_pose_path = get_path_by_frame(frame_path, view_id, "pose", "txt")
        if not os.path.exists(cur_pose_path):
            cur_pose_path = frame_path + '_pose.txt'
        cur_pose = torch.Tensor(np.loadtxt(cur_pose_path))
        # extend dimension in case of single joint case
        if len(cur_pose.shape) == 1:
            cur_pose = cur_pose.unsqueeze(0)

        # project the joint location to eliminate unnecessary joint variation
        if self.projection:
            if "laptop" in self.dataset_dir:
                cur_pose[:, 1] = 0

        # Create map for pixel-wise supervisoin of pose
        joint_map = self.gen_joint_map(cur_pose, self.img_size)

        frame['joint_map'] = joint_map
        frame['pose'] = cur_pose

        mesh_path = frame_path + '_wt_mesh.obj'

        if not self.is_train_data:        
            frame['mesh'] = mesh_path

        return frame
        
    def load_occupancies(self, frame_path, view_id=None):
        """
        load the occupancy data for the 2nd part(IF-Net)
            coords: position to be queried
            occupancies: ground truth for position to be queried
        """        
        data_dir = os.path.dirname(frame_path)
        file_name = os.path.basename(frame_path)
        index_of_frame = find_frame_num(file_name)
        # we only use canonical as supervisioin
        if view_id is None:
            index_of_frame = str(int(index_of_frame) // self.pose_num * self.pose_num).zfill(8)

        points = []
        coords = []
        occupancies = []
        for i, num in enumerate(self.num_samples):
            if view_id is None:
                boundary_samples_path = os.path.join(data_dir, "frame_" + index_of_frame + '_' +\
                                                    self.occ_load_str[i] + '.npz')
            else:
                boundary_samples_path = os.path.join(data_dir, "frame_{}_view_{}_{}.npz".format(index_of_frame, str(view_id).zfill(2), self.occ_load_str[i]))
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points
        
        if view_id is None:
            gt_mesh_path = os.path.join(data_dir, "frame_" + index_of_frame + '_' +\
                                                    "isosurf_scaled.off")
        else:
            gt_mesh_path = os.path.join(data_dir, "frame_{}_view_{}_isosurf_scaled.off".format(index_of_frame, str(view_id).zfill(2)))
        # None of the if-data would be needed if in validation mode
        # print("loaded", gt_mesh_path)

        if_data = {
            'grid_coords':np.array(coords, dtype=np.float32),
            'occupancies': np.array(occupancies, dtype=np.float32),                        
            }
        # if not self.is_train_data:
        if_data['iso_mesh'] = gt_mesh_path

        if self.config.TRANSFORM:
            if view_id is None:
                transform_path = os.path.join(data_dir, "frame_" + index_of_frame + '_transform.npz')        
            else:
                transform_path = os.path.join(data_dir, "frame_{}_view_{}_transform.npz".format(index_of_frame, str(view_id).zfill(2)))                
            nocs_transform = {}
            nocs_transform['translation'] = np.load(transform_path)['translation']
            nocs_transform['scale'] = np.load(transform_path)['scale']
            
            if_data['translation'] = nocs_transform['translation']
            if_data['scale'] = nocs_transform['scale']

        return if_data

    def get_crr(self, batch):
        # ADDITIONAL OPERATION: FIND CORRESPONDENCE BETWEEN NEAR VIEWS
        # crr-idx-mtx is a triangular matrix RIGHT, each row corresponds to the base
        # (which pc the index saved in this row indicate) and each column corresponds to query (aligned with which pc)
        # WARNING! THE CORRESPONDENCE ONLY WORKS FOR NOCS-V
        # batch = {}
        crr_idx_mtx, crr_mask_mtx, crr_min_d_mtx = list(), list(), list()
        pair_count = 0
        for view_id in range(self.view_num - 1):
            query_nox_list = batch['nox00'][view_id + 1:]
            query_mask_list = [nox[3] for nox in query_nox_list]
            idx_list, mask_list, mind_list = self.find_correspondence_list(
                # query_pc_list=batch['uv-xyz-v'][view_id + 1:], query_mask_list=batch['uv-mask-v'][view_id + 1:],
                query_pc_list=batch['pnnocs00'][view_id + 1:], query_mask_list=query_mask_list,
                base_pc=batch['pnnocs00'][view_id], base_mask=batch['nox00'][view_id][3]
            )
            crr_idx_mtx.append(idx_list)
            crr_mask_mtx.append(mask_list)
            crr_min_d_mtx.append(mind_list)
            for mask in mask_list:
                pair_count += mask.sum()
                # print(pair_count)
        ave_crr_per_view = int(pair_count / self.view_num)
        batch['crr-idx-mtx'] = crr_idx_mtx
        batch['crr-mask-mtx'] = crr_mask_mtx

        return batch

    def find_correspondence_list(self, query_pc_list, base_pc, query_mask_list, base_mask, th=1e-3):
        """
        For each pc in query_pc_list, find correspondence point in base pc,
        if no correspondent point in base, mask this position with 0 in mask
        :param query_pc_list: the returned element is aligned with this list [N1*d, N2*d, ...]
        :param base_pc: the search base
        :param query_mask_list: whether each point is really valid in this sample
        :param base_mask:
        :param th: it the minimal distance is smaller than this, correspondence is found for the pair
        :return:
            index_list: aligned with query_pc_list, for each element in the list, the values at each position indicate
                        which position in the base pc that this position point in query pc corresponds to;
                        i.e values of index means the index in base pc
            mask_list: aligned with index_list, whether on this position, a pair is found
            mind_list: the min_distance of
        """

        # cur_mask = out_mask[i].squeeze()
        # masked = cur_mask > threshold
        # point_cloud = pred_nocs[i, :, masked]
        crr_len = self.supervision_cap

        q_pc_list = [query_view[:, query_mask.squeeze() > 0].permute(1, 0).numpy()
                     for query_view, query_mask in zip(query_pc_list, query_mask_list)]

        b_pc = base_pc[:, base_mask.squeeze() > 0].permute(1, 0).numpy()

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(b_pc)
        index_list, mask_list, min_dis_list = list(), list(), list()
        # fix_len = True
        fix_len = False

        for q_pc in q_pc_list:
            assert q_pc.shape[1] == b_pc.shape[1]
            distance, indices = neigh.kneighbors(q_pc, return_distance=True)
            _min_d = distance.ravel()
            _idx = indices.ravel()  
            _mask = (_min_d < th).astype(np.float)
            # make sure the output is in the same size

            if fix_len:
                index = np.zeros((8192, 1))
                mask = np.zeros((8192, 1)).astype(np.float)
                min_d = np.zeros((8192, 1)).astype(np.float)
            else:
                index = np.zeros((len(q_pc), 1))
                mask = np.zeros((len(q_pc), 1)).astype(np.float)
                min_d = np.zeros((len(q_pc), 1)).astype(np.float)

            # crr_idx = np.where(_min_d < th)[0] # here crr_idx stores the index

            # if len(q_pc) > self.supervision_cap:
                # samples = np.random.choice(crr_idx.shape[0], self.supervision_cap, replace=False)
                
            # write_off("/workspace/crr/gt_crr_1.xyz", q_pc)
            # write_off("/workspace/crr/gt_crr_0.xyz", b_pc[_idx])
            # crr_idx = crr_idx[samples]
            
            # num_crr = len(crr_idx)

            # # here we store no more than self.cap pairs
            # _idx = _idx[crr_idx]
            # _mask = _mask[crr_idx]
            # _min_d = _min_d[crr_idx]
            
            # # for current use we choose uniform sample
            # sampled_idx = crr_idx[random_index]
            index[:len(q_pc), 0] = _idx
            mask[:len(q_pc), 0] = _mask
            min_d[:len(q_pc), 0] = _min_d
            # index = _idx
            # mask = _mask
            # min_d = _min_d

            index_list.append(index.astype(np.int))
            mask_list.append(mask.astype(np.float))
            min_dis_list.append(min_d.astype(np.float))

        return index_list, mask_list, min_dis_list

if __name__ == '__main__':
    import argparse
    from config import get_cfg
    # preparer configuration
    cfg = get_cfg()    
    f_str = None
    Data = MVSPDataset(cfg, train=True)
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=1, shuffle=True, num_workers=4)
    
    cnt = 0
    mean = 0
    std = 0

    for i, Data in enumerate(DataLoader, 0):  # Get each batch
        img = Data['color00'][0][0]
        # print(img.shape)
        cur_mean = torch.mean(img, axis=(1,2))
        cur_std = torch.std(img, axis=(1,2))
        
        mean += cur_mean
        std += cur_std
        cnt += 1

        sys.stdout.write("\r mean {} std {}".format(mean.numpy() / cnt, std.numpy() / cnt ).ljust(40) )  
        sys.stdout.flush()
        pass
        
