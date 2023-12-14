import torch
import numpy as np

KITTI_ARGS = dict(
        voxelsize = [0.05, 0.05, 0.1],
        point_cloud_range = [0, -40, 0, 70.4, 40, 0.4],
        ignore_labels = [0],
        vis_voxel_size = 0.4,
    )

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        x, y, z = x.squeeze(), y.squeeze(), z.squeeze()
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def gen_stan_inds(d):
    assert d % 2 != 0
    s = (d - 1) / 2
    before_mesh = torch.arange(d).float()
    before_mesh -= s

    x, y = torch.meshgrid(before_mesh, before_mesh, indexing='ij')
    x = x.reshape(-1)
    y = y.reshape(-1)

    expand_grid_inds = torch.stack([x,y])

    return expand_grid_inds.permute(1,0).unsqueeze(dim=0)
    
def voxel_inds2points(voxel_inds):

    point_cloud_range = KITTI_ARGS['point_cloud_range']
    voxel_size = KITTI_ARGS['voxelsize']
    voxel_inds[:,0] = voxel_inds[:,0]*voxel_size[0] + point_cloud_range[0]
    voxel_inds[:,1] = voxel_inds[:,1]*voxel_size[1] + point_cloud_range[1]

def expand_grid(ins_points, diameter, is_voxel=False):

    if not is_voxel:
        points = ins_points
        point_cloud_range = KITTI_ARGS['point_cloud_range']
        voxel_size = KITTI_ARGS['voxelsize']

        x, y = points[:, 0], points[:, 1]
        coord_x = (x - point_cloud_range[0]) / voxel_size[0] / 8
        coord_y = (y - point_cloud_range[1]) / voxel_size[1] / 8

        coord_x = torch.clamp(coord_x, min=0, max=176 - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=200 - 0.5)  #

        ins_points = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)

    coord_int = ins_points.int().unsqueeze(dim = 1)
    
    coord_int = coord_int.repeat([1, diameter**2, 1])
    expend_inds = gen_stan_inds(diameter).int().to(coord_int.device)
    coord_int = coord_int + expend_inds # 256 * 25 * 2
    coord_int_float = coord_int.float().reshape(-1,2)
    # return coord_int_float[val_mask]
    return coord_int_float

def gaussian_2d(shape, sigma: float = 1):

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_2d_v2(gaus_features, diameter):

    voxel_inds = gen_stan_inds(diameter).to(gaus_features.device) # 1x25x2
    voxel_dis = voxel_inds[...,0]**2 + voxel_inds[...,1]**2
    voxel_dis = voxel_dis.repeat([gaus_features.shape[0], 1])
    gaus_coeff = torch.exp(-0.4*voxel_dis*gaus_features)
    return gaus_coeff

def get_sh_coeff(sh_features, diameter):

    assert torch.isnan(sh_features).sum() == 0
    voxel_inds = gen_stan_inds(diameter).to(sh_features.device) # 1x25x2
    dir_pp = torch.zeros([voxel_inds.shape[1], 3]).to(sh_features.device)
    dir_pp[:,:2] = -voxel_inds.squeeze()
    dir_pp_normalized = dir_pp/(dir_pp.norm(dim=1, keepdim=True) + 1e-5) # [x, 3]
    dir_pp_normalized = dir_pp_normalized.unsqueeze(dim = 0).repeat([sh_features.shape[0],1,1]) # [2, 24, 3]
    sh_features = eval_sh(3, sh_features.repeat([1,dir_pp_normalized.shape[1],1]), dir_pp_normalized)
    sh_features = sh_features + 0.5
    sh_features[:, int((diameter**2-1)/2)] = 1
    sh_features = torch.clamp_min(sh_features, 0.0)
    assert torch.isnan(sh_features).sum() == 0

    return sh_features

def get_gaus_coeff(gaussian_radius):
    return gaussian_2d((gaussian_radius,gaussian_radius), sigma=2).reshape(-1)

def get_sparse_voxel_feature(model_config, points_coord, pw_features, sh_features):

    gaussian_radius = model_config.GAUS_RADIUS
    diameter = model_config.DIAMETER
    pw_feature_dim = pw_features.shape[-1]
    sh_feature_dim = sh_features.shape[-1]

    # get expand voxel_indices
    points = points_coord[...,1:].reshape(-1,3)
    coord_int_float = expand_grid(points, diameter) # [51200, 2]
    batch_inds = points_coord[...,0].view(-1,1)
    batch_inds = batch_inds.repeat([1,diameter**2])
    batch_inds = batch_inds.reshape(-1,1)
    coord_int_float = torch.cat([batch_inds, coord_int_float], dim=-1)

    # get sh coeff
    sh_features = sh_features.reshape(-1,1,sh_feature_dim)
    sh_coeff = get_sh_coeff(sh_features, diameter)

    # get gaussian coeff
    gaus_coeff = get_gaus_coeff(gaussian_radius) # [2048, 25]
    gaus_coeff = torch.from_numpy(gaus_coeff).to(sh_coeff.device, torch.float32)

    # add coeff
    coeff_all = sh_coeff + gaus_coeff# 2048 * 25
    coeff_all = coeff_all.unsqueeze(dim = -1)

    # merge same indices
    pw_features = pw_features.reshape(-1, 1, pw_feature_dim)
    pw_features = pw_features.repeat([1,diameter**2,1]) * coeff_all
    pw_features = pw_features.reshape(-1, pw_feature_dim)

    indices_unique, _inv = torch.unique(coord_int_float, dim=0,return_inverse=True)
    features_unique = pw_features.new_zeros((indices_unique.shape[0], pw_feature_dim))
    features_unique.index_add_(0, _inv, pw_features)

    x_mask = (indices_unique[:,1] >= 0) & (indices_unique[:,1] < 176)
    y_mask = (indices_unique[:,2] >= 0) & (indices_unique[:,2] < 200)
    val_mask = x_mask & y_mask

    return features_unique[val_mask], indices_unique[val_mask][:,[0,2,1]]

def pool_feature_in_same_voxel(points_coord, pw_feature):
    points_coord = points_coord.reshape(-1,4)
    points = points_coord[:,1:]
    point_cloud_range = KITTI_ARGS['point_cloud_range']
    voxel_size = KITTI_ARGS['voxelsize']

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    coord_x = (x - point_cloud_range[0]) / voxel_size[0] / 8
    coord_y = (y - point_cloud_range[1]) / voxel_size[1] / 8

    coord_x = torch.clamp(coord_x, min=0, max=176 - 0.5)  # bugfixed: 1e-6 does not work for center.int()
    coord_y = torch.clamp(coord_y, min=0, max=200 - 0.5)  #
    coord = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)

    points_coord[:,1:3] = coord.int()
    points_coord = points_coord[:,:3].float()

    indices_unique, _inv = torch.unique(points_coord, dim=0,return_inverse=True)
    features_unique = pw_feature.new_zeros((indices_unique.shape[0], pw_feature.shape[-1]))
    features_unique.index_add_(0, _inv, pw_feature)
    assert torch.isnan(pw_feature).sum() == 0

    return indices_unique, features_unique, _inv


# merge voxel before expand
def get_sparse_voxel_feature2(model_config, points_coord, pw_features, sh_features):

    gaussian_radius = model_config.GAUS_RADIUS
    diameter = model_config.DIAMETER
    pw_feature_dim = pw_features.shape[-1]
    sh_feature_dim = sh_features.shape[-1]

    # get expand voxel_indices
    points = points_coord[...,1:]
    coord_int_float = expand_grid(points, diameter, is_voxel=True) # [51200, 2]
    batch_inds = points_coord[...,0].view(-1,1)
    batch_inds = batch_inds.repeat([1,diameter**2])
    batch_inds = batch_inds.reshape(-1,1)
    coord_int_float = torch.cat([batch_inds, coord_int_float], dim=-1)

    # get sh coeff
    sh_features = sh_features.reshape(-1,1,sh_feature_dim)
    sh_coeff = get_sh_coeff(sh_features, diameter)

    # get gaussian coeff
    gaus_coeff = get_gaus_coeff(gaussian_radius) # [2048, 25]
    gaus_coeff = torch.from_numpy(gaus_coeff).to(sh_coeff.device, torch.float32)
    gaus_coeff = gaus_coeff.unsqueeze(dim=0).repeat([sh_coeff.shape[0], 1]).unsqueeze(dim=-1)

    # add coeff
    # coeff_all = sh_coeff + gaus_coeff# 2048 * 25
    # coeff_all = coeff_all.unsqueeze(dim = -1)
    sh_coeff = sh_coeff.unsqueeze(dim = -1)

    # merge same indices
    pw_features = pw_features.reshape(-1, 1, pw_feature_dim)
    pw_feature_dim = int(pw_feature_dim / 2)
    # pw_features = pw_features.repeat([1,diameter**2,1]) * coeff_all # [1890, 25, 256]
    pw_features = pw_features.repeat([1,diameter**2,1])
    pw_features = pw_features[..., :pw_feature_dim] * sh_coeff + pw_features[..., pw_feature_dim:] * gaus_coeff
    pw_features = pw_features.reshape(-1, pw_feature_dim)

    indices_unique, _inv = torch.unique(coord_int_float, dim=0,return_inverse=True)
    features_unique = pw_features.new_zeros((indices_unique.shape[0], pw_feature_dim))
    features_unique.index_add_(0, _inv, pw_features)

    x_mask = (indices_unique[:,1] >= 0) & (indices_unique[:,1] < 176)
    y_mask = (indices_unique[:,2] >= 0) & (indices_unique[:,2] < 200)
    val_mask = x_mask & y_mask

    return features_unique[val_mask], indices_unique[val_mask][:,[0,2,1]]
    
def get_sparse_voxel_feature3(model_config, points_coord, pw_features, sh_features, gaus_features):

    coffe_dict = {}
    gaussian_radius = model_config.GAUS_RADIUS
    diameter = model_config.DIAMETER
    pw_feature_dim = pw_features.shape[-1]
    sh_feature_dim = sh_features.shape[-1]

    # get expand voxel_indices
    points = points_coord[...,1:]
    coord_int_float = expand_grid(points, diameter, is_voxel=True) # [51200, 2]
    batch_inds = points_coord[...,0].view(-1,1)
    batch_inds = batch_inds.repeat([1,diameter**2])
    batch_inds = batch_inds.reshape(-1,1)
    coord_int_float = torch.cat([batch_inds, coord_int_float], dim=-1)

    # get sh coeff
    sh_features = sh_features.reshape(-1,1,sh_feature_dim)
    sh_coeff = get_sh_coeff(sh_features, diameter)

    # get gaussian coeff
    # gaus_coeff = get_gaus_coeff(gaussian_radius) # [2048, 25]
    gaus_coeff = gaussian_2d_v2(gaus_features, diameter)
    # gaus_coeff = torch.from_numpy(gaus_coeff).to(sh_coeff.device, torch.float32)
    gaus_coeff = gaus_coeff.unsqueeze(dim=-1)

    # add coeff
    # coeff_all = sh_coeff + gaus_coeff# 2048 * 25
    # coeff_all = coeff_all.unsqueeze(dim = -1)
    sh_coeff = sh_coeff.unsqueeze(dim = -1)

    coffe_dict.update({
        'sh_coeff': sh_coeff,
        'gaus_coeff': gaus_coeff
    })

    # merge same indices
    pw_features = pw_features.reshape(-1, 1, pw_feature_dim)
    pw_feature_dim = int(pw_feature_dim / 2)
    # pw_features = pw_features.repeat([1,diameter**2,1]) * coeff_all # [1890, 25, 256]
    pw_features = pw_features.repeat([1,diameter**2,1])
    pw_features = pw_features[..., :pw_feature_dim] * sh_coeff + pw_features[..., pw_feature_dim:] * gaus_coeff
    pw_features = pw_features.reshape(-1, pw_feature_dim)

    indices_unique, _inv = torch.unique(coord_int_float, dim=0,return_inverse=True)
    features_unique = pw_features.new_zeros((indices_unique.shape[0], pw_feature_dim))
    features_unique.index_add_(0, _inv, pw_features)

    x_mask = (indices_unique[:,1] >= 0) & (indices_unique[:,1] < 176)
    y_mask = (indices_unique[:,2] >= 0) & (indices_unique[:,2] < 200)
    val_mask = x_mask & y_mask

    return features_unique[val_mask], indices_unique[val_mask][:,[0,2,1]], coffe_dict


if __name__ == '__main__':

    # sh = torch.randn((2,1,16))
    # voxel_inds = gen_stan_inds(5) # 1x25x2
    # dir_pp = torch.zeros([voxel_inds.shape[1], 3])
    # dir_pp[:,:2] = -voxel_inds.squeeze()
    # dir_pp_normalized = dir_pp/(dir_pp.norm(dim=1, keepdim=True) + 1e-5) # [x, 3]
    # dir_pp_normalized = dir_pp_normalized.unsqueeze(dim = 0).repeat([2,1,1]) # [2, 24, 3]
    
    # sh_feature = eval_sh(3, sh.repeat([1,dir_pp_normalized.shape[1],1]), dir_pp_normalized)
    # sh_feature[:, 12] = 1
    # sh_feature = torch.clamp_min(sh_feature + 0.5, 0.0)

    # print(sh_feature.shape)

    # gaussian = gaussian_2d((5,5), sigma=2).reshape(-1)
    # print(gaussian)

    get_sh_coeff(torch.randn([2048,1,16]), 5)


            


