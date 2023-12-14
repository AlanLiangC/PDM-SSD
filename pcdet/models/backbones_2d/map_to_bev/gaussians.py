import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from functools import partial

from mmdet3d.models.layers import make_sparse_convmodule
from mmcv.ops.group_points import QueryAndGroup, grouping_operation
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from ....utils.gaus_utils import (build_covariance, voxelize)
from mmengine.model import BaseModule
from mmengine.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

class Gaussians(BaseModule):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            covas = build_covariance(scaling_modifier * scaling, rotation)
            return covas
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, 
                 model_cfg,
                 **kwards
                ) -> None:
        super().__init__()
        self.setup_functions()

        self.model_cfg = model_cfg.MODEL_CONFIG
        encoder_in_channels = self.model_cfg['encoder_in_channels']
        voxel_size = self.model_cfg['voxel_size']
        seg_output_channels = self.model_cfg['seg_output_channels']
        gaus_output_channels = self.model_cfg['gaus_output_channels']
        group_output_channels = self.model_cfg['group_output_channels']
        classificer_channels = self.model_cfg['classificer_channels']
        norm_cfg = dict(type='BN1d', eps=1e-3, momentum=0.01)
        point_cloud_range = self.model_cfg['point_cloud_range']
        sample_num = self.model_cfg['sample_num']
        class_num = self.model_cfg['class_num']
        top_num = 128
        vp_mode = self.model_cfg['vp_mode']
        ignore_index = None
        loss_ce = None
        loss_lovasz =  None
        frozen_parms = self.model_cfg['frozen_parms']

        self.seg_out = nn.ModuleList()
        self.gaus_out = nn.ModuleList()
        init_encoder_in_channels = encoder_in_channels
        self.point_cloud_range = point_cloud_range
        self.class_num = class_num
        self.sample_num = sample_num
        self.top_num = top_num
        self.vp_mode = vp_mode
        self.voxel_size = voxel_size
        self.frozen_parms = frozen_parms
        self.num_bev_features = 128

        for channel in seg_output_channels:
            conv_out = make_sparse_convmodule(
                encoder_in_channels,
                channel,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                norm_cfg=norm_cfg,
                padding=0,
                indice_key='spconv_down2',
                conv_type='SparseConv3d',)
            
            self.seg_out.append(conv_out)
            encoder_in_channels = channel
        
        encoder_in_channels = init_encoder_in_channels
        last_layer = False
        for i in range(len(gaus_output_channels)):
            channel = gaus_output_channels[i]
            conv_out = make_sparse_convmodule(
                encoder_in_channels,
                channel,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                norm_cfg=norm_cfg,
                padding=0,
                indice_key='spconv_down2',
                conv_type='SparseConv3d',
                order = ('conv', 'norm') if last_layer else ('conv', 'norm', 'act'))
            
            self.gaus_out.append(conv_out)
            encoder_in_channels = channel

            if i == len(gaus_output_channels) - 1:
                last_layer = True

        self.rot_regresser = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 4),
        )

        self.xyz_scales_regresser = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 3),
        )

        self.scales_regresser = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 3),
        )

        shared_mlps = []
        for k in range(len(group_output_channels) - 1):
            shared_mlps.extend([
                nn.Conv2d(group_output_channels[k], group_output_channels[k + 1],
                            kernel_size=1, bias=False),
                nn.BatchNorm2d(group_output_channels[k + 1]),
                nn.ReLU()
            ])
        self.grouped_mlps = nn.Sequential(*shared_mlps)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(128, 128, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(128, 128, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

        # shared_mlps = []
        # for k in range(len(classificer_channels) - 1):
        #     shared_mlps.extend([
        #         nn.Conv2d(classificer_channels[k], classificer_channels[k + 1],
        #                     kernel_size=1, bias=False),
        #         nn.BatchNorm2d(classificer_channels[k + 1]),
        #         nn.ReLU()
        #     ])
        # self.classifer_grouped_mlps = nn.Sequential(*shared_mlps)

        self.seg_pred = list()
        self.loss_lovasz = MODELS.build(loss_lovasz) if loss_lovasz is not None else None
        self.loss_ce = MODELS.build(loss_ce) if loss_ce is not None else None
        # self.ignore_index = torch.tensor(ignore_index, dtype=torch.int64) if ignore_index is not None else None
        self.ignore_index = ignore_index
        self.qag = QueryAndGroup(max_radius=1, sample_num=self.sample_num, use_xyz=True, return_grouped_idx=True)

    @property
    def get_xyz_scales(self):    
        return self.opacity_activation(self._xyz_scales)
    
    @property
    def get_scaling(self):
        ###############################
        self._scales = torch.clamp_max(self._scales, 5)
        self._scales = torch.clamp_min(self._scales, -5)
        ###############################
        return self.scaling_activation(self._scales)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def replace_feature(self, out: SparseConvTensor,
                    new_features: SparseConvTensor) -> SparseConvTensor:
        if 'replace_feature' in out.__dir__():
            # spconv 2.x behaviour
            return out.replace_feature(new_features)
        else:
            out.features = new_features
            return out 
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def rem_batch_size(self, batch_size):
        setattr(self, 'batch_size', batch_size)

    def get_gaus_from_parms(self):
        
        batch_size, _, D, H, W = self.voxel_pred_one_hot.dense().shape
        # voxel real size
        self.rem_batch_size(batch_size)

        # covariance matrix
        cov3D_precomp, det3D_precomp = self.get_covariance(scaling_modifier=1) # 
        # means
        voxel_indices = self.voxel_pred_one_hot.indices
        self.batch_voxel_center = []
        self.batch_voxel_seg = []
        self.batch_cov3D_precomp = []
        self.batch_det3D_precomp = []
        self.batch_voxel_feature = []
        # self.batch_voxel_as_points = /[]
        # self.invse_indices = []

        voxel_size = torch.tensor(self.voxel_size, device=cov3D_precomp.device).float() * 8
        voxel_size[-1] = 4

        for batch_idx in range(batch_size):
            batch_mask = voxel_indices[:,0] == batch_idx
            xyz = voxel_indices[batch_mask][:,[3, 2, 1]].float()
            # self.invse_indices.append(voxel_indices[batch_mask][:,[0,2,3]])
            xyz_scale = self.get_xyz_scales[batch_mask]

            xyz = (xyz + xyz_scale) * voxel_size + torch.tensor(self.point_cloud_range).float().to(xyz.device)[:3]
            # new_xyz = (xyz + 0.5) * voxel_size + torch.tensor(self.point_cloud_range).float().to(xyz.device)[:3]

            self.batch_voxel_center.append(xyz)
            # self.batch_voxel_as_points.append(new_xyz)
            self.batch_voxel_seg.append(self.voxel_pred_one_hot.features[batch_mask,...])
            self.batch_cov3D_precomp.append(cov3D_precomp[batch_mask,...])
            self.batch_det3D_precomp.append(det3D_precomp[batch_mask,...])
            self.batch_voxel_feature.append(self.identity[batch_mask,...])

        # self.invse_indices = torch.cat(self.invse_indices, dim=0)
    
    def voxel_as_points(self, batch_inputs_dict):

        x_min, y_min, z_min, x_max, y_max, z_max = torch.tensor(self.point_cloud_range).float()
        _, _, D, H, W, = self.voxel_pred_one_hot.dense().shape
        d, h, w = (z_max - z_min) / D, (y_max - y_min) / H, (x_max - x_min) / W

        voxelize_cfg=dict(
            max_num_points=50,
            point_cloud_range=self.point_cloud_range,
            voxel_size=[w, h, d],
            max_voxels=[16000, 40000])
        
        points_seg_label = None
        if hasattr(self, 'voxel_seg_label'):
            delattr(self, 'voxel_seg_label')

        points_list = []
        points = batch_inputs_dict['points']

        for idx in range(batch_inputs_dict['batch_size']):
            batch_mask = points[:,0] == idx
            points_list.append(points[batch_mask][:,1:].contiguous())

        feats, coords, sizes = voxelize(points = points_list,
                                    voxelize_cfg = voxelize_cfg,
                                    points_seg_label = points_seg_label)
        coords = coords.int()

        self.voxel_sp_tensor = SparseConvTensor(feats[:,-1].view(-1, 1), 
                                        coords[:,[0,2,1]],
                                        self.voxel_pred_one_hot.dense().shape[3:], 
                                        self.batch_size)
        self.batch_voxel_as_points = []

        for batch_idx in range(self.batch_size):
            batch_mask = coords[:,0] == batch_idx

            xyz = coords[batch_mask][:,1:].float()

            xyz[:,0] = (xyz[:,0] + 0.5) * h + x_min
            xyz[:,1] = (xyz[:,1] + 0.5) * w + y_min
            xyz[:,2] = (xyz[:,2] + 0.5) * d + z_min

            self.batch_voxel_as_points.append(xyz)

    def get_seg_pred(self, batch_inputs_dict):

        voxel_corr_feature = []
        for batch_idx in range(self.batch_size):
            means = self.batch_voxel_center[batch_idx].view(1,-1,3).contiguous()
            if self.vp_mode:
                points = self.batch_voxel_as_points[batch_idx].view(1,-1,3).contiguous()
            else:
                if batch_inputs_dict.get('key_points', None) is not None:
                    points = batch_inputs_dict['key_points'][batch_idx][:,:3].view(1,-1,3).contiguous()
                else:
                    points = batch_inputs_dict['points'][batch_idx][:,1:4].view(1,-1,3).contiguous()
            
            batch_one_hot = self.batch_voxel_seg[batch_idx].unsqueeze(dim = 0).permute(0, 2, 1).contiguous()
            batch_cov3D_precomp = self.batch_cov3D_precomp[batch_idx].reshape(-1,9).unsqueeze(dim = 0).permute(0, 2, 1).contiguous() # [1, 9, 11174]
            batch_det3D_precomp = self.batch_det3D_precomp[batch_idx].reshape(1,1,-1).contiguous()
            batch_voxel_feature = self.batch_voxel_feature[batch_idx].unsqueeze(dim = 0).permute(0, 2, 1).contiguous()

            grouped_xyz_feature, idx = self.qag(means, points) # [1, 3, 33500, 8]
            grouped_xyz_feature = -grouped_xyz_feature.squeeze().permute(1, 2, 0).unsqueeze(dim = 2) # [33706, 8, 1, 3]
            
            grouped_cova_feature = grouping_operation(batch_cov3D_precomp, idx) # [1, 9, 33500, 8]
            grouped_cova_feature = grouped_cova_feature.squeeze().permute(1, 2, 0).view(-1, self.sample_num, 3, 3) # [32738, 8, 3, 3]

            grouped_det_feature = grouping_operation(batch_det3D_precomp, idx) # [1, 1, 32738, 8]
            grouped_det_feature = grouped_det_feature.squeeze().unsqueeze(dim = -1) # [31740, 8, 1]
            
            exp_index = -(grouped_xyz_feature@grouped_cova_feature@grouped_xyz_feature.transpose(2,3))
            gaus_result = torch.exp(exp_index.squeeze()).unsqueeze(dim = -1) / grouped_det_feature # [30991, 8, 1]
            gaus_result = F.normalize(gaus_result, dim=1)
            grouped_voxel_feature = grouping_operation(batch_voxel_feature, idx).squeeze().permute(1, 2, 0) # [1612, 8, 128]

            grouped_voxel_feature = gaus_result * grouped_voxel_feature

            voxel_corr_feature.append(grouped_voxel_feature.unsqueeze(dim = 0))

        result_feature = []
        for feature in voxel_corr_feature:
            feature = feature.permute(0,3,1,2)
            feature = self.grouped_mlps(feature)
            new_features = F.max_pool2d(
                            feature, kernel_size=[1, feature.size(3)])
            result_feature.append(new_features.squeeze().permute(1,0))

        result_feature = torch.cat(result_feature, dim=0)
        result_featurel_sp_tensor = self.voxel_sp_tensor.replace_feature(result_feature)
        result_dict = dict(
            result_feature = [result_featurel_sp_tensor]
        )
        return result_dict
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append('rgb_{}'.format(i))
        l.append('opacity')
        for i in range(self._scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def merge_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices
        spatial_shape = x_conv.spatial_shape

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )

        x_out = self.shared_conv(self.conv_out(x_out))
        return x_out

    def merge_feature(self, feat_dict, indep_feature):

        ori_feature = feat_dict['result_feature'][0]
        ori_feature = ori_feature.replace_feature(torch.cat([torch.zeros_like(ori_feature.features), indep_feature]))
        ori_feature.indices = torch.cat([ori_feature.indices, self.voxel_pred_one_hot.indices[:,[0,2,3]]])

        new_feature = self.merge_out(ori_feature)

        feat_dict['result_feature'] = [new_feature]
        return feat_dict

    def extract_feature(self, batch_inputs_dict):

        self.get_gaus_from_parms()
        # pred 
        if self.vp_mode:
            self.voxel_as_points(batch_inputs_dict)
        result_dict = self.get_seg_pred(batch_inputs_dict)

        result_dict = self.merge_feature(result_dict, self.identity)

        return result_dict['result_feature']
    
    def dim_out(self, x_conv):
        features_cat = x_conv.features
        indices_unique = x_conv.indices
        spatial_shape = x_conv.spatial_shape

        new_indices = indices_unique.new_zeros([indices_unique.shape[0], 4])
        new_indices[:,[0,2,3]] = indices_unique

        new_spatial_shape = [1, spatial_shape[0], spatial_shape[1]]
        
        x_out = spconv.SparseConvTensor(
            features=features_cat,
            indices=new_indices,
            spatial_shape=new_spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def forward(self, batch_dict):

        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        x = self.dim_out(encoded_spconv_tensor)

        self.identity = x.features
        for sub_seg_model in self.seg_out:
            x = self.replace_feature(x, sub_seg_model(x).features)

        self.voxel_pred_one_hot = x.replace_feature(x.features) # 32

        x = self.replace_feature(x, self.identity)
        for sub_gaus_model in self.gaus_out:
            x = self.replace_feature(x, sub_gaus_model(x).features)

        self._rotation = self.rot_regresser(x.features)
        self._xyz_scales = self.xyz_scales_regresser(x.features)
        self._scales = self.scales_regresser(x.features)

        result_feature = self.extract_feature(batch_dict)

        batch_dict['encoded_spconv_tensor'] = result_feature[0]

        return batch_dict

        # self.identity = torch.cat([self.identity, self.voxel_pred_one_hot.features], dim=-1)

    def predict(self, feats, batch_inputs_dict):
        # parms learning
        self(feats)
        self.get_gaus_from_parms()
        if self.vp_mode:
            self.voxel_as_points(batch_inputs_dict, batch_data_samples = None)
        points_seg_pred, _ = self.get_seg_pred(batch_inputs_dict)

        return points_seg_pred

    def _freeze_stages(self):
        if self.frozen_parms:
            for name, child in self.named_children():
                child.eval()
                for param in child.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(Gaussians, self).train(mode)
        self._freeze_stages()
        if mode and self.frozen_parms:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()