import torch
import torch.nn as nn
from functools import partial
from ....utils import al_utils
from ....utils.spconv_utils import replace_feature, spconv


class PointExpandVoxel(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.sh_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.SH_FC,
            input_channels=256,
            output_channels=(self.model_cfg.MAX_SH + 1)**2
        )

        input_channels = self.model_cfg.get('SHARED_CONV_CHANNEL', 128)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(input_channels),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(True),
        )

        self.num_bev_features = input_channels

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, batch_dict):
        ins_features = batch_dict['encoder_features'][-3]
        ins_features = ins_features.permute(0, 2, 1).contiguous().view(-1, ins_features.shape[1])
        pred_sh = self.sh_layers(ins_features) # N x 16
        sp_features, sp_indices = al_utils.get_sparse_voxel_feature(self.model_cfg, 
                                                                 batch_dict['encoder_coords'][4].squeeze(), 
                                                                 pw_features=ins_features,
                                                                 sh_features=pred_sh)
        
        x_conv = spconv.SparseConvTensor(
            features=sp_features,
            indices=sp_indices.int(),
            spatial_shape=[200,176],
            batch_size=batch_dict['batch_size']
        )

        x_conv = self.shared_conv(self.conv_out(x_conv))

        batch_dict['encoded_spconv_tensor'] = x_conv

        return batch_dict
    
class PointExpandVoxel2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.sh_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.SH_FC,
            input_channels=256,
            output_channels=(self.model_cfg.MAX_SH + 1)**2
        )

        input_channels = self.model_cfg.get('SHARED_CONV_CHANNEL', 128)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(input_channels),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(True),
        )

        self.num_bev_features = 512

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, batch_dict):
        ins_features = batch_dict['encoder_features'][4]
        ins_features = ins_features.permute(0, 2, 1).contiguous().view(-1, ins_features.shape[1])
        
        assert torch.isnan(ins_features).sum() == 0
        points_coord, pw_features = al_utils.pool_feature_in_same_voxel(points_coord=batch_dict['encoder_coords'][4].squeeze(),
                                                                        pw_feature=ins_features)
        pred_sh = self.sh_layers(pw_features) # N x 16
        assert torch.isnan(pw_features).sum() == 0
        sp_features, sp_indices = al_utils.get_sparse_voxel_feature2(self.model_cfg, 
                                                                 points_coord, 
                                                                 pw_features=pw_features,
                                                                 sh_features=pred_sh)
        x_conv = spconv.SparseConvTensor(
            features=sp_features,
            indices=sp_indices.int(),
            spatial_shape=[200,176],
            batch_size=batch_dict['batch_size']
        )

        x_conv = self.shared_conv(self.conv_out(x_conv))

        batch_dict['encoded_spconv_tensor'] = x_conv

        return batch_dict

class PointExpandVoxel3(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.sh_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.SH_FC,
            input_channels=256,
            output_channels=(self.model_cfg.MAX_SH + 1)**2
        )

        self.gaus_layers = self.make_fc_layers(
            fc_cfg=[256,128],
            input_channels=256,
            output_channels=1
        )

        input_channels = self.model_cfg.get('SHARED_CONV_CHANNEL', 128)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(input_channels),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(input_channels, input_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(True),
        )

        self.num_bev_features = 512

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, batch_dict):
        ins_features = batch_dict['encoder_features'][4]
        ins_features = ins_features.permute(0, 2, 1).contiguous().view(-1, ins_features.shape[1])
        
        assert torch.isnan(ins_features).sum() == 0
        points_coord, pw_features, inv_ = al_utils.pool_feature_in_same_voxel(points_coord=batch_dict['encoder_coords'][4].squeeze(),
                                                                        pw_feature=ins_features)
        pred_sh = self.sh_layers(pw_features) # N x 16
        pred_gaus = self.gaus_layers(pw_features).sigmoid()
        assert torch.isnan(pw_features).sum() == 0
        sp_features, sp_indices, coffe_dict = al_utils.get_sparse_voxel_feature3(self.model_cfg, 
                                                                 points_coord, 
                                                                 pw_features=pw_features,
                                                                 sh_features=pred_sh,
                                                                 gaus_features=pred_gaus)
        x_conv = spconv.SparseConvTensor(
            features=sp_features,
            indices=sp_indices.int(),
            spatial_shape=[200,176],
            batch_size=batch_dict['batch_size']
        )

        x_conv = self.shared_conv(self.conv_out(x_conv))

        batch_dict['encoded_spconv_tensor'] = x_conv


        #################
        self.inv_ = inv_
        self.coffe_dict = coffe_dict
        #################

        return batch_dict