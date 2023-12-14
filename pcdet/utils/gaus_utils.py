#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from os import makedirs, path
from errno import EEXIST
from typing import NamedTuple
from datetime import datetime

import torch
from torch.nn import functional as F
import random
import numpy as np
from ..ops.voxel import Voxelization

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def load_instance(instance_path, select_num = None):

    all_points = np.loadtxt(instance_path)[:,:3]
    if select_num is not None:
        random_index = random.randint(0, points.shape[0])
        points = all_points[random_index, :]
    point_class = BasicPointCloud(points=points, colors=None, normals=None)
    return all_points, point_class


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / (norm[:, None] + 1e-5)

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0] + 1e-5
    L[:,1,1] = s[:,1] + 1e-5
    L[:,2,2] = s[:,2] + 1e-5

    L = R @ L
    return L

def strip_lowerdiag(L):

    # L[:, 0, 0] = 1 / (L[:, 0, 0] + 1e-5)
    # L[:, 1, 1] = 1 / (L[:, 1, 1] + 1e-5)
    # L[:, 2, 2] = 1 / (L[:, 2, 2] + 1e-5)

    uncertainty = torch.zeros((L.shape[0], 3, 3), dtype=torch.float, device = 'cuda')

    uncertainty[:, 0, 0] = L[:, 0, 0]
    uncertainty[:, 0, 1] = L[:, 0, 1]
    uncertainty[:, 0, 2] = L[:, 0, 2]
    uncertainty[:, 1, 0] = L[:, 0, 1]
    uncertainty[:, 1, 1] = L[:, 1, 1]
    uncertainty[:, 1, 2] = L[:, 1, 2]
    uncertainty[:, 2, 0] = L[:, 0, 2]
    uncertainty[:, 2, 1] = L[:, 1, 2]
    uncertainty[:, 2, 2] = L[:, 2, 2]

    L = torch.inverse(uncertainty)

    return L

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_covariance(s, r):

    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    # R = build_rotation(r)
    R = torch.eye(3, device='cuda').unsqueeze(dim=0).repeat(s.shape[0],1,1)

    L[:,0,0] = 1 / (s[:,0] + 1e-5)
    L[:,1,1] = 1 / (s[:,1] + 1e-5)
    L[:,2,2] = 1 / (s[:,2] + 1e-5)

    inv_C = R @ L @ L @ R.transpose(1,2)

    det_C = torch.prod(s, dim=-1)

    return inv_C, det_C

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


@torch.no_grad()
def voxelize(points, voxelize_cfg, points_seg_label = None):
    pts_voxel_layer = Voxelization(**voxelize_cfg)
    feats, coords, sizes = [], [], []
    for k, res in enumerate(points):
        if points_seg_label is not None:
            res = torch.cat([res, points_seg_label[k].view(-1, 1 ).to(torch.int64)], dim=-1)
        ret = pts_voxel_layer(res)
        if len(ret) == 3:
            f, c, n = ret # [1108, 100, 5]
        else:
            assert len(ret) == 2
            f, c = ret
            n = None
        feats.append(f)
        coords.append(F.pad(c, (1, 0), mode='constant', value=k))
        if n is not None:
            sizes.append(n)

    coords = torch.cat(coords, dim=0)

    sizes = torch.cat(sizes, dim=0)
    feats = torch.cat(feats, dim=0)
    feats = feats.sum(
                dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
    feats = feats.contiguous()
    return feats, coords, sizes

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


if __name__ == "__main__":

    r = torch.randn(5,4) + 1

    s = torch.randint(1,5,(5,3))

    inv_C, det_C = build_covariance(s, r)
