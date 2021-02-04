from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
# from collections import namedtuple
import torch.nn.functional as F

from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule



class Pointnet_Backbone(nn.Module):  # Pointnet++骨干网络
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=use_xyz,
            )
        )
        # ModuleList(
        #   (0): PointnetSAModule(
        #     (groupers): ModuleList(
        #       (0): QueryAndGroup(radius=0.3, nsample=32)
        #     )
        #     (mlps): ModuleList(
        #       (0): SharedMLP(
        #         (layer0): Conv2d(
        #           (conv): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #         (layer1): Conv2d(
        #           (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #         (layer2): Conv2d(
        #           (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #       )
        #     )
        #   )
        #   (1): PointnetSAModule(
        #     (groupers): ModuleList(
        #       (0): QueryAndGroup(radius=0.5, nsample=32)
        #     )
        #     (mlps): ModuleList(
        #       (0): SharedMLP(
        #         (layer0): Conv2d(
        #           (conv): Conv2d(131, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #         (layer1): Conv2d(
        #           (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #         (layer2): Conv2d(
        #           (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #       )
        #     )
        #   )
        #   (2): PointnetSAModule(
        #     (groupers): ModuleList(
        #       (0): QueryAndGroup(radius=0.7, nsample=32)
        #     )
        #     (mlps): ModuleList(
        #       (0): SharedMLP(
        #         (layer0): Conv2d(
        #           (conv): Conv2d(259, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #         (layer1): Conv2d(
        #           (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #         (layer2): Conv2d(
        #           (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #           (normlayer): BatchNorm2d(
        #             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #           )
        #           (activation): ReLU(inplace=True)
        #         )
        #       )
        #     )
        #   )
        # )
        self.cov_final = nn.Conv1d(256, 256, kernel_size=1)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0: 3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        # 拆分，前三维是坐标B * N * 3，后面都是特征B * C * N或者None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        ## type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)


        return l_xyz, l_features, self.cov_final(l_features[-1])

class Pointnet_Tracking(nn.Module):
    r"""
        xorr the search and the template
    """
    def __init__(self, input_channels=3, use_xyz=True, test=False):
        super(Pointnet_Tracking, self).__init__()
        self.test = test
        self.backbone_net = Pointnet_Backbone(input_channels, use_xyz)

        self.aux_modules = nn.ModuleList()  # 附加网络模块
        self.aux_modules.append(
            PointnetFPModule(
                mlp=[512, 256] 
            )
        )
        self.aux_modules.append(
            PointnetFPModule(
                mlp=[384, 256] 
            )
        )
        self.aux_modules.append(
            PointnetFPModule(
                mlp=[256, 256] 
            )
        )
        # ModuleList(
        #   (0): PointnetFPModule(
        #     (mlp): SharedMLP(
        #       (layer0): Conv2d(
        #         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #         (normlayer): BatchNorm2d(
        #           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         )
        #         (activation): ReLU(inplace=True)
        #       )
        #     )
        #   )
        #   (1): PointnetFPModule(
        #     (mlp): SharedMLP(
        #       (layer0): Conv2d(
        #         (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #         (normlayer): BatchNorm2d(
        #           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         )
        #         (activation): ReLU(inplace=True)
        #       )
        #     )
        #   )
        #   (2): PointnetFPModule(
        #     (mlp): SharedMLP(
        #       (layer0): Conv2d(
        #         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #         (normlayer): BatchNorm2d(
        #           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         )
        #         (activation): ReLU(inplace=True)
        #       )
        #     )
        #   )
        # )
        self.cosine = nn.CosineSimilarity(dim=1)

        self.mlp = pt_utils.SharedMLP([4 + 256, 256, 256, 256], bn=True)
        # SharedMLP(
        #   (layer0): Conv2d(
        #     (conv): Conv2d(260, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (normlayer): BatchNorm2d(
        #       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (layer1): Conv2d(
        #     (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (normlayer): BatchNorm2d(
        #       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (layer2): Conv2d(
        #     (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (normlayer): BatchNorm2d(
        #       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        # )
        self.FC_layer_cla = (
                pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(1, activation=None))
        # Seq(
        #   (0): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #       (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (1): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #       (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (2): Conv1d(
        #     (conv): Conv1d(256, 1, kernel_size=(1,), stride=(1,))
        #   )
        # )
        self.fea_layer = (
                pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, activation=None))
        # Seq(
        # (0): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #     (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        # )
        # (1): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
        # )
        # )        
        self.vote_layer = (
                pt_utils.Seq(3 + 256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3 + 256, activation=None))
        # Seq(
        #   (0): Conv1d(
        #     (conv): Conv1d(259, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #       (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (1): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #       (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (2): Conv1d(
        #     (conv): Conv1d(256, 259, kernel_size=(1,), stride=(1,))
        #   )
        # )
        self.vote_aggregation = PointnetSAModule(
                radius=0.3,
                nsample=16,
                mlp=[1 + 256, 256, 256, 256],
                use_xyz=use_xyz)  # 根据投票聚合投影候选中心点
        # PointnetSAModule(
        #   (groupers): ModuleList(
        #     (0): QueryAndGroup()
        #   )
        #   (mlps): ModuleList(
        #     (0): SharedMLP(
        #       (layer0): Conv2d(
        #         (conv): Conv2d(260, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #         (normlayer): BatchNorm2d(
        #           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         )
        #         (activation): ReLU(inplace=True)
        #       )
        #       (layer1): Conv2d(
        #         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #         (normlayer): BatchNorm2d(
        #           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         )
        #         (activation): ReLU(inplace=True)
        #       )
        #       (layer2): Conv2d(
        #         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #         (normlayer): BatchNorm2d(
        #           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         )
        #         (activation): ReLU(inplace=True)
        #       )
        #     )
        #   )
        # )
        self.num_proposal = 64  # 实际上是_PointnetSAModuleBase(nn.Module)的forward中的npoint输入参数
        self.FC_proposal = (
                pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3 + 1 + 1, activation=None))
        # Seq(
        #   (0): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #       (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (1): Conv1d(
        #     (conv): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
        #     (normlayer): BatchNorm1d(
        #       (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (activation): ReLU(inplace=True)
        #   )
        #   (2): Conv1d(
        #     (conv): Conv1d(256, 5, kernel_size=(1,), stride=(1,))
        #   )
        # )
        # self.aux_seg = (
                # pt_utils.Seq(256)
                # .conv1d(1, bn=True, activation=None)
                # .conv1d(256, bn=True)
                # .conv1d(1, activation=None)
        # )
        self.aux_seg = nn.Linear(256, 1, bias=False)
        self.aux_offset = nn.Linear(256, 3, bias=False)
    # @staticmethod
    def xcorr(self, x_label, x_object, template_xyz):       
        r'''
            x_label = search_feature
            x_object = template_feature
        '''
        B = x_object.size(0)  # Batch数目
        f = x_object.size(1)  # d1
        n1 = x_object.size(2)  # M1 = 64
        n2 = x_label.size(2)  # M2 = 128
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B, f, n1, n2), x_label.unsqueeze(2).expand(B, f, n1, n2))
        # similarity map
        fusion_feature = torch.cat((final_out_cla.unsqueeze(1), template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B,3,n1,n2)), dim=1)
        # B* 1* M1* M2 + B* 3* M1* M2
        fusion_feature = torch.cat((fusion_feature, x_object.unsqueeze(-1).expand(B, f, n1, n2)), dim=1)
        # B* (1 + 3)* M1* M2 + B* d1* M1* M2 
        # Similarity map + template seeds * M2  
        fusion_feature = self.mlp(fusion_feature)  # MLP1 on feature channel
        # 此时为B * d2 * M1 * M2
        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1]) 
        # Maxpool on M1 channel，输出为B * d2 * 1 * M2
        fusion_feature = fusion_feature.squeeze(2)  # B * d2 * M2
        fusion_feature = self.fea_layer(fusion_feature)  # MLP2 on feature channel，输出为B * d2 * M2

        return fusion_feature

    def forward(self, template, search):
        r"""
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template_xyz_list, template_feature_list, template_feature = self.backbone_net(template, [256, 128, 64])
        # 分别为B * 64 * 3，B * d1 * 64
        search_xyz_list, search_feature_list, search_feature = self.backbone_net(search, [512, 256, 128])
        # 分别为B * 128 * 3，B * d2 * 128
        template_xyz = template_xyz_list[-1]
        search_xyz = search_xyz_list[-1]
        fusion_feature = self.xcorr(search_feature, template_feature, template_xyz)  # B * d2 * M2 = B * d2 * 128

        estimation_cla = self.FC_layer_cla(fusion_feature).squeeze(1)  # B * M2

        score = estimation_cla.sigmoid()  # B * M2

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(), fusion_feature), dim=1)
        # Search area seeds with target-specific feature
        # B * (3 + d2) * M2
        offset = self.vote_layer(fusion_xyz_feature)  # Voting，输出为B * (d2 + 3) * M2
        vote = fusion_xyz_feature + offset  # B * (d2 + 3) * M2
        vote_xyz = vote[:, 0: 3, :].transpose(1, 2).contiguous()  # 前三维是坐标，与之前cat的顺序对应
        # B * M2 * 3 
        vote_feature = vote[:, 3:, :]  # B * d2 * M2

        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)  # 用于投影候选中心点的投票特征
        # B * (1 + d2) * M2

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)
        # B * self.num_proposal * 3, B * d2 * M2

        proposal_offsets = self.FC_proposal(proposal_features)
        # B * 5 * self.num_proposal，C=5，坐标3维 + X-Y面旋转1维 + proposal-wise targetness得分1维
        estimation_boxs = torch.cat((proposal_offsets[:, 0: 3, :]+center_xyzs.transpose(1, 2).contiguous(), proposal_offsets[:, 3: 5, :]), dim=1)
        # B * 5 * self.num_proposal = B * 5 * 64
        if self.test == False:  # 附加任务只在训练时进行
            search_feature_list.append(None)  # 最后一层的unknown_feats没有
            new_feature = list(range(len(self.aux_modules)))
            for i in range(len(self.aux_modules)):
                if i == 0:
                    new_feature[i] = self.aux_modules[i](search_xyz_list[2 - i], search_xyz_list[3 - i], search_feature_list[2 - i], search_feature)
                else:
                    new_feature[i] = self.aux_modules[i](search_xyz_list[2 - i], search_xyz_list[3 - i], search_feature_list[2 - i], new_feature[i - 1])
            aux_feature = new_feature[-1].transpose(1, 2)
            estimation_seg = self.aux_seg(aux_feature)  # 分割得分，B * 1024 * 1
            estimation_offset = self.aux_offset(aux_feature)  # 偏移估计，B * 1024 * 3
            return estimation_cla, vote_xyz, estimation_boxs.transpose(1, 2).contiguous(), center_xyzs, estimation_seg.squeeze(-1), estimation_offset
        else:
            search_feature_list.append(None)  # 最后一层的unknown_feats没有
            new_feature = list(range(len(self.aux_modules)))
            for i in range(len(self.aux_modules)):
                if i == 0:
                    new_feature[i] = self.aux_modules[i](search_xyz_list[2 - i], search_xyz_list[3 - i], search_feature_list[2 - i], search_feature)
                else:
                    new_feature[i] = self.aux_modules[i](search_xyz_list[2 - i], search_xyz_list[3 - i], search_feature_list[2 - i], new_feature[i - 1])
            aux_feature = new_feature[-1].transpose(1, 2)
            estimation_seg = self.aux_seg(aux_feature)  # 分割得分，B * 1024 * 1
            return estimation_cla, vote_xyz, estimation_boxs.transpose(1, 2).contiguous(), center_xyzs, estimation_seg.squeeze(-1)
