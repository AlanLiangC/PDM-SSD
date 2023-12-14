import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from ...utils.spconv_utils import spconv
from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from ..model_utils import centernet_utils

from .point_head_template import PointHeadTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features

        return ret_dict


class VoxelNeXtHeadPlus(PointHeadTemplate):
    """
    A simple point-based detect head, which are used for IA-SSD.
    """
    def __init__(self, model_cfg, input_channels, num_class,
                 predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        detector_dim = self.model_cfg.INPUT_DIM # for spec input_channel
        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=detector_dim,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=detector_dim,
            output_channels=self.box_coder.code_size
        )
        
        self.box_iou3d_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.IOU_FC,
            input_channels=detector_dim,
            output_channels=1
        ) if self.model_cfg.get('IOU_FC', None) is not None else None

        ##############################################
        ##******************************************##
        point_cloud_range = self.model_cfg.POINT_CLOUD_RANGE
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor([0.05,0.05,0.1]).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        class_names = self.model_cfg.CLASS_NAMES
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.gaussian_ratio = self.model_cfg.get('GAUSSIAN_RATIO', 1)
        self.gaussian_type = self.model_cfg.get('GAUSSIAN_TYPE', ['nearst', 'gt_center'])
        # The iou branch is only used for Waymo dataset
        self.iou_branch = self.model_cfg.get('IOU_BRANCH', False)

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.get('SHARED_CONV_CHANNEL', 128),
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                )
            )

        self.SA_modules = nn.ModuleList()
        sa_config = self.model_cfg.CTR_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)


        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            channel_in = 256#self.model_cfg.SHARED_CONV_CHANNEL

            if self.layer_types[k] == 'SA_Layer':
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=sa_config.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,                                                
                        dilated_group=sa_config.DILATED_GROUP[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class = self.num_class
                    )
                )
        ##******************************************##
        ##############################################

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):

        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('center_reg_loss_func', loss_utils.RegLossSparse())
        self.add_module(
                'two_label_reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=[1.0,1.0,1.0]),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )

        # classification loss
        if losses_cfg.LOSS_CLS.startswith('WeightedBinaryCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedBinaryCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS.startswith('WeightedCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedClassificationLoss()
            )
        elif losses_cfg.LOSS_CLS.startswith('FocalLoss'):
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        else:
            raise NotImplementedError

        # regression loss
        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError

        # instance-aware loss
        if losses_cfg.get('LOSS_INS', None) is not None:
            if losses_cfg.LOSS_INS.startswith('WeightedBinaryCrossEntropy'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.WeightedBinaryCrossEntropyLoss()
                )
            elif losses_cfg.LOSS_INS.startswith('WeightedCrossEntropy'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.WeightedClassificationLoss()
                )
            elif losses_cfg.LOSS_INS.startswith('FocalLoss'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(
                        **losses_cfg.get('LOSS_CLS_CONFIG', {})
                    )
                )
            else:
                raise NotImplementedError

    def assign_stack_targets_IASSD(self, points, gt_boxes, extend_gt_boxes=None, weighted_labels=False,
                             ret_box_labels=False, ret_offset_labels=True,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                             use_query_assign=False, central_radii=2.0, use_ex_gt_assign=False, fg_pc_ignore=False,
                             binary_label=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        box_idxs_labels = points.new_zeros(points.shape[0]).long() 
        gt_boxes_of_fg_points = []
        gt_box_of_points = gt_boxes.new_zeros((points.shape[0], 8))

        for k in range(batch_size):            
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if use_query_assign: ##
                centers = gt_boxes[k:k + 1, :, 0:3]
                query_idxs_of_pts = roiaware_pool3d_utils.points_in_ball_query_gpu(
                    points_single.unsqueeze(dim=0), centers.contiguous(), central_radii
                    ).long().squeeze(dim=0) 
                query_fg_flag = (query_idxs_of_pts >= 0)
                if fg_pc_ignore:
                    fg_flag = query_fg_flag ^ box_fg_flag 
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = query_fg_flag
                    box_idxs_of_pts = query_idxs_of_pts
            elif use_ex_gt_assign: ##
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                extend_fg_flag = (extend_box_idxs_of_pts >= 0)
                
                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[box_fg_flag] #instance points should keep unchanged

                if fg_pc_ignore:
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag 
                    box_idxs_of_pts = extend_box_idxs_of_pts 
                                
            elif set_ignore_flag: 
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            elif use_ball_constraint: 
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 or binary_label else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            bg_flag = (point_cls_labels_single == 0) # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single


        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
            'box_idxs_labels': box_idxs_labels,
            'gt_box_of_points': gt_box_of_points,
        }
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size: int
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                centers_origin: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_coords: List of point_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            target_dict:
            ...
        """
        target_cfg = self.model_cfg.TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:   #nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = input_dict['batch_size']      
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes

        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        center_targets_dict = self.assign_stack_targets_IASSD(
            points=input_dict['centers'].detach(), 
            gt_boxes=extend_gt, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_box_labels=True
        )
        targets_dict_center['center_gt_box_of_fg_points'] = center_targets_dict['gt_box_of_fg_points']
        targets_dict_center['center_cls_labels'] = center_targets_dict['point_cls_labels']
        targets_dict_center['center_box_labels'] = center_targets_dict['point_box_labels'] #only center assign
        targets_dict_center['center_gt_box_of_points'] = center_targets_dict['gt_box_of_points']
        if target_cfg.get('INS_AWARE_ASSIGN', False):
            sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [],[],[],[],[]
            sa_ins_preds = input_dict['sa_ins_preds']
            for i in range(1, len(sa_ins_preds)): # valid when i = 1,2 for IA-SSD
                # if sa_ins_preds[i].__len__() == 0:
                #     continue
                sa_xyz = input_dict['encoder_coords'][i]
                if i == 1:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  #[0.2, 0.2, 0.2]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=True, use_ex_gt_assign= False 
                    )
                if i >= 2:
                # if False:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                    ).view(batch_size, -1, gt_boxes.shape[-1])             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=False, use_ex_gt_assign= True 
                    )
                # else:
                #     extend_gt_boxes = box_utils.enlarge_box3d(
                #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                #     ).view(batch_size, -1, gt_boxes.shape[-1]) 
                #     sa_targets_dict = self.assign_stack_targets_IASSD(
                #         points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                #         set_ignore_flag=False, use_ex_gt_assign= True 
                #     )
                sa_xyz_coords.append(sa_xyz)
                sa_ins_labels.append(sa_targets_dict['point_cls_labels'])
                sa_gt_box_of_fg_points.append(sa_targets_dict['gt_box_of_fg_points'])
                sa_gt_box_of_points.append(sa_targets_dict['gt_box_of_points'])
                sa_box_idxs_labels.append(sa_targets_dict['box_idxs_labels'])                
                
            targets_dict_center['sa_ins_labels'] = sa_ins_labels
            targets_dict_center['sa_gt_box_of_fg_points'] = sa_gt_box_of_fg_points
            targets_dict_center['sa_xyz_coords'] = sa_xyz_coords
            targets_dict_center['sa_gt_box_of_points'] = sa_gt_box_of_points
            targets_dict_center['sa_box_idxs_labels'] = sa_box_idxs_labels

        return targets_dict_center
    
    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    
    def get_heat_map_loss(self, tb_dict = None):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['sp_target_dicts']
        # batch_index = self.forward_ret_dict['batch_index']
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm']) # [58875, 3]
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            loss += hm_loss

        

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'hm_loss': loss.item()})

        return loss, tb_dict
    
    def get_center_reg_loss(self, tb_dict=None):

        loss = 0
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['sp_target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        for idx, pred_dict in enumerate(pred_dicts):
            target_boxes = target_dicts['sp_target_boxes'][idx][...,:3] # [8, 500, 8]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1) # [58875, 8]

            reg_loss = self.center_reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )
            loss += reg_loss.sum()

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'reg_loss': loss.item()})

        return loss, tb_dict
    
    def get_two_center_label_loss(self, tb_dict=None):
        
        loss = 0

        center_cls_preds = self.forward_ret_dict['center_cls_preds'].sigmoid()
        sp_center_cls_preds = self.forward_ret_dict['sp_center_cls_preds']

        pos_mask = sp_center_cls_preds.sum(dim = -1) > 0.4
        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        
        reg_loss = self.two_label_reg_loss_func(
            center_cls_preds[None, ...], sp_center_cls_preds[None, ...], weights=reg_weights[None, ...]
        )
        loss += reg_loss.sum()

        if tb_dict is None:
            tb_dict = {}

        tb_dict.update({'two_center_label_loss': loss.item()})

        return loss, tb_dict


    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        # semantic loss in SA layers
        if self.model_cfg.LOSS_CONFIG.get('LOSS_INS', None) is not None:
            assert ('sa_ins_preds' in self.forward_ret_dict) and ('sa_ins_labels' in self.forward_ret_dict)
            sa_loss_cls, tb_dict_0 = self.get_sa_ins_layer_loss()
            tb_dict.update(tb_dict_0)
        else:
            sa_loss_cls = 0

        # cls loss
        center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss()
        tb_dict.update(tb_dict_4)

        # reg loss
        if self.model_cfg.TARGET_CONFIG.BOX_CODER == 'PointResidualCoder':
            center_loss_box, tb_dict_5 = self.get_box_layer_loss()
        else:
            center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss()
        tb_dict.update(tb_dict_5)    
        
        # corner loss
        if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
            corner_loss, tb_dict_6 = self.get_corner_layer_loss()
            tb_dict.update(tb_dict_6)

        # iou loss
        iou3d_loss = 0
        if self.model_cfg.LOSS_CONFIG.get('IOU3D_REGULARIZATION', False):
            iou3d_loss, tb_dict_7 = self.get_iou3d_layer_loss()          
            tb_dict.update(tb_dict_7)

        # heat map loss
        heat_map_loss, tb_dict_8 = self.get_heat_map_loss()
        assert not torch.isnan(heat_map_loss)
        tb_dict.update(tb_dict_8)
        # center loss
        center_reg_loss, tb_dict_9 = self.get_center_reg_loss()
        assert not torch.isnan(center_reg_loss)
        tb_dict.update(tb_dict_9)

        # two_center_label_loss, tb_dict_10 = self.get_two_center_label_loss()
        # assert not torch.isnan(two_center_label_loss)
        # tb_dict.update(tb_dict_10)

        point_loss = center_loss_cls + center_loss_box + corner_loss + sa_loss_cls + iou3d_loss + heat_map_loss + center_reg_loss# + two_center_label_loss
        assert not torch.isnan(point_loss)
        return point_loss, tb_dict


    def get_center_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['center_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['center_cls_preds'].view(-1, self.num_class)
        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0

        cls_weights = (1.0 *negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        
        if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
            centerness_mask = self.generate_center_ness_mask()
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

        point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'center_loss_cls': point_loss_cls.item(),
            'center_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict
    

    def get_sa_ins_layer_loss(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['sa_ins_labels']
        sa_ins_preds = self.forward_ret_dict['sa_ins_preds']
        sa_centerness_mask = self.generate_sa_center_ness_mask()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_labels)): # valid when i =1, 2
            if len(sa_ins_preds[i]) != 0:
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)

            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                centerness_mask = sa_centerness_mask[i]
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()        
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })

        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict


    def generate_center_ness_mask(self):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        centers = self.forward_ret_dict['centers'][:,1:]
        centers = centers[pos_mask].clone().detach()
        offset_xyz = centers[:, 0:3] - gt_boxes[:, 0:3]

        offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

        template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
        margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
        distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)

        centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
        centerness_mask[pos_mask] = centerness
        return centerness_mask
    

    def generate_sa_center_ness_mask(self):
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels']
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points']
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords']
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)):
            pos_mask = sa_pos_mask[i] > 0
            gt_boxes = sa_gt_boxes[i]
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:]
            xyz_coords = xyz_coords[pos_mask].clone().detach()
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3]
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

            template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
            margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
            distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
            distance[:, 1, :] = -1 * distance[:, 1, :]
            distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
            distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
            centerness = torch.clamp(centerness, min=1e-6)
            centerness = torch.pow(centerness, 1/3)

            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
            centerness_mask[pos_mask] = centerness

            sa_centerness_mask.append(centerness_mask)
        return sa_centerness_mask


    def get_center_box_binori_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        point_loss_box_src = self.reg_loss_func(
            pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_xyzwhl = point_loss_box_src.sum()

        pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
        pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]

        label_ori_bin_id = point_box_labels[:, 6]
        label_ori_bin_res = point_box_labels[:, 7]
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
        loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)

        label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
        pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
        loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
        loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss_ori_cls = loss_ori_cls * loss_weights_dict.get('dir_weight', 1.0)
        point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl.item()})
        tb_dict.update({'center_loss_box_ori_bin': loss_ori_cls.item()})
        tb_dict.update({'center_loss_box_ori_res': loss_ori_reg.item()})
        return point_loss_box, tb_dict


    def get_center_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss = point_loss_box_src.sum()

        point_loss_box = point_loss
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict


    def get_corner_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner.item()})
        return loss_corner, tb_dict


    def get_iou3d_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds'].clone().detach()
        pred_boxes = pred_boxes[pos_mask]
        iou3d_targets, _ = loss_utils.generate_iou3d(pred_boxes[:, 0:7], gt_boxes[:, 0:7])

        iou3d_preds = self.forward_ret_dict['box_iou3d_preds'].squeeze(-1)
        iou3d_preds = iou3d_preds[pos_mask]

        loss_iou3d = F.smooth_l1_loss(iou3d_preds, iou3d_targets)

        loss_iou3d = loss_iou3d * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou3d_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'iou3d_loss_reg': loss_iou3d.item()})
        return loss_iou3d, tb_dict
    
    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def assign_targets_sp_voxel(self, gt_boxes, num_voxels, spatial_indices, spatial_shape):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'sp_target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'sp_gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head_sp_voxel(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    num_voxels=num_voxels[bs_idx], spatial_indices=spatial_indices[bs_idx], 
                    spatial_shape=spatial_shape, 
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['sp_target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['sp_gt_boxes'].append(gt_boxes_list)

        return ret_dict

    def assign_target_of_single_head_sp_voxel(
            self, num_classes, gt_boxes, num_voxels, spatial_indices, spatial_shape, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels) # [3, 11806]

        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1)) # [500, 8]
        inds = gt_boxes.new_zeros(num_max_objs).long() # [500]
        mask = gt_boxes.new_zeros(num_max_objs).long() # [500]

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)  #

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1

            if 'gt_center' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio)

            if 'nearst' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * self.gaussian_ratio)

            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2]
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def _get_predicted_boxes(self, sp_features, pred_dicts, voxel_indices):
        pred_boxes = []
        pred_center_labels = []
        pred_center_sp_features = []
        batch_index = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:]
        
        for idx, pred_dict in enumerate(pred_dicts):
            center = pred_dict['center']
            center_z = pred_dict['center_z']
            xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

            box_part_list = [batch_index.reshape(-1,1),xs, ys, center_z]
            pred_box = torch.cat((box_part_list), dim=-1)
            
            heatmap = pred_dict['hm'].sigmoid().sum(dim = -1)
            ori_heatmap = pred_dict['hm'].sigmoid()
            batch_pred_boxes = []
            batch_pred_center_labels = []
            batch_pred_center_sp_features = []
            for bs_idx in range(self.batch_size):
                batch_inds = batch_index==bs_idx
                sub_heatmap = heatmap[batch_inds]
                sub_pred_box = pred_box[batch_inds]
                _, index = torch.topk(sub_heatmap, 500)
                sub_pred_box = sub_pred_box[index]
                batch_pred_boxes.append(sub_pred_box)
                batch_pred_center_labels.append(ori_heatmap[batch_inds][index])
                batch_pred_center_sp_features.append(sp_features[batch_inds][index])
            
            batch_pred_boxes = torch.stack(batch_pred_boxes)
            batch_pred_center_labels = torch.cat(batch_pred_center_labels)
            batch_pred_center_sp_features = torch.cat(batch_pred_center_sp_features)

            pred_boxes.append(batch_pred_boxes)
            pred_center_labels.append(batch_pred_center_labels)
            pred_center_sp_features.append(batch_pred_center_sp_features)

        return pred_boxes[0], pred_center_labels[0], pred_center_sp_features[0]
    
    def _get_gt_boxes(self, sp_features, pred_dicts, voxel_indices):
        pred_boxes = []
        pred_center_labels = []
        pred_center_sp_features = []
        batch_index = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:]
        
        for idx, pred_dict in enumerate(pred_dicts):
            center = pred_dict['center']
            center_z = pred_dict['center_z']
            xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

            box_part_list = [batch_index.reshape(-1,1),xs, ys, center_z]
            pred_box = torch.cat((box_part_list), dim=-1)
            
            heatmap = pred_dict['hm'].sigmoid().sum(dim = -1)
            ori_heatmap = pred_dict['hm'].sigmoid()
            batch_pred_boxes = []
            batch_pred_center_labels = []
            batch_pred_center_sp_features = []
            for bs_idx in range(self.batch_size):
                batch_inds = batch_index==bs_idx
                sub_heatmap = heatmap[batch_inds]
                sub_pred_box = pred_box[batch_inds]
                _, index = torch.topk(sub_heatmap, 500)
                sub_pred_box = sub_pred_box[index]
                batch_pred_boxes.append(sub_pred_box)
                batch_pred_center_labels.append(ori_heatmap[batch_inds][index])
                batch_pred_center_sp_features.append(sp_features[batch_inds][index])
            
            batch_pred_boxes = torch.stack(batch_pred_boxes)
            batch_pred_center_labels = torch.cat(batch_pred_center_labels)
            batch_pred_center_sp_features = torch.cat(batch_pred_center_sp_features)

            pred_boxes.append(batch_pred_boxes)
            pred_center_labels.append(batch_pred_center_labels)
            pred_center_sp_features.append(batch_pred_center_sp_features)

        return pred_boxes[0], pred_center_labels[0], pred_center_sp_features[0]

    def _get_center_features(batch_dict):
        pass

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                centers_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_xyz: List of points_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                batch_cls_preds: (N1 + N2 + N3 + ..., num_class)
                point_box_preds: (N1 + N2 + N3 + ..., 7)
        """

        self.batch_size = batch_dict['batch_size']
        ret_dict = {}

        # heatmap
        ###################################################
        #*************************************************#
        x_conv = batch_dict['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x_conv)
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x_conv))

        ret_dict['pred_dicts'] = pred_dicts
        ret_dict['batch_index'] = batch_index
        ret_dict['voxel_indices'] = voxel_indices

        self.pts_feats = x_conv
        self.ins_point = batch_dict['encoder_xyz'][3]
        self.heatmap = pred_dicts[0]['hm'].sigmoid()

        if self.training:

            sp_targets_dict = self.assign_targets_sp_voxel(batch_dict['gt_boxes'], 
                                                        num_voxels, 
                                                        spatial_indices, 
                                                        spatial_shape)
            ret_dict['sp_target_dicts'] = sp_targets_dict

        if self.training:
            pred_centers, sp_center_cls_preds, pred_centers_features = self._get_gt_boxes(x_conv.features, sp_targets_dict, voxel_indices)
            ret_dict['sp_center_cls_preds'] = sp_center_cls_preds
        else:
            pred_centers, sp_center_cls_preds, pred_centers_features = self._get_predicted_boxes(x_conv.features, pred_dicts, voxel_indices)
            ret_dict['sp_center_cls_preds'] = sp_center_cls_preds
        
        for i in range(len(self.SA_modules)):
            xyz_input = batch_dict['encoder_xyz'][3]
            feature_input = batch_dict['encoder_features'][3]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = pred_centers[...,1:].contiguous()
                _, center_features, _ = self.SA_modules[i](xyz_input, feature_input, ctr_xyz=ctr_xyz)
        center_features = center_features.permute(0, 2, 1).contiguous().view(-1, center_features.shape[1])
        center_features = torch.cat([center_features, pred_centers_features], dim=-1)

        self.pred_centers = pred_centers[...,1:].contiguous()
        #*************************************************#
        ###################################################

        center_coords = pred_centers.reshape(-1,4)
        batch_dict['centers'] = center_coords
        
        #################
        # self.pred_centers = batch_dict['encoder_xyz'][4][0].squeeze()#center_coords[:,1:].squeeze()
        #################

        center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        box_iou3d_preds = self.box_iou3d_layers(center_features) if self.box_iou3d_layers is not None else None

        ret_dict.update( 
            {
                'center_cls_preds': center_cls_preds,
                'center_box_preds': center_box_preds,
                'ctr_offsets': center_coords,
                'centers': center_coords,
                'sa_ins_preds': batch_dict['sa_ins_preds'],
                'box_iou3d_preds': box_iou3d_preds,
            })
        
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.IOU3D_REGULARIZATION:

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                    points=center_coords[:, 1:4],
                    point_cls_preds=center_cls_preds, point_box_preds=center_box_preds
                )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['box_iou3d_preds'] = box_iou3d_preds
            batch_dict['batch_index'] = center_coords[:,0]
            batch_dict['cls_preds_normalized'] = False

            ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict
