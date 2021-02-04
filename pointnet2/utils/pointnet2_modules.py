from __future__ import (
    division,
    # 导入python未来支持的语言特征division(精确除法)
    # 除不尽可以有小数，不被截断
    absolute_import,
    # 绝对引入可以绕过同目录的xx.py文件引入系统自带标准的xx.py文件
    with_statement,
    # 可以用with语法
    print_function,
    # 即使在python2.X，使用print就得像python3.X那样加括号使用
    unicode_literals,
    # 使得当前文件下的编码默认就是unicode
)
# 上述导入主要是对Python2.X环境发挥作用
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils

from pointnet2.utils import pointnet2_utils

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class _PointnetSAModuleBase(nn.Module):  # 自定义set abstraction层的基础类
    # 主要是定义点聚合操作、通过MLP-MaxPool得到新的点组特征的方法
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None  # 组
        self.mlps = None  # MLP

    def forward(self, xyz, features, npoint):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
            B = Batch, N = Number of Points
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features
            C是特征维数

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        self.npoint = npoint  # 聚合后的质心数量
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        # contiguous：view只能用在内存占用连续的数据块的tensor上
        # 如果在view之前用了transpose, permute等
        # 需要用contiguous()来返回一个contiguous copy
        # (B, 3, N)
        
        new_xyz = (
            pointnet2_utils.gather_operation(
                #xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                xyz_flipped,
                torch.arange(self.npoint).repeat(xyz.size(0), 1).int().cuda()
            )
            .transpose(1, 2)
            .contiguous()
        )  # 聚合后得到的centroids，B * npoint * 3

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            # (batch, channel, feature_num, in_group_num)
            new_features = self.mlps[i](new_features)
            # (B, mlp[-1], npoint, nsample)
            # (batch, mlp最后一层输出维度, feature_num, in_group_num)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            # (batch, mlp最后一层输出维度, feature_num, 1)
            # 全局max_pool
            new_features = new_features.squeeze(-1)
            # (B, mlp[-1], npoint)
            # (batch, mlp最后一层输出维度, feature_num)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)
        # torch.cat连接列表里面的所有new_features


class PointnetSAModuleMSG(_PointnetSAModuleBase):  # 带多尺度聚合的点集提取层
    r"""Pointnet set abstraction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features 特征数量
    radii : list of float32
        list of radii to group with 球查询半径列表
    nsamples : list of int32
        Number of samples in each ball query 球查询包含的采样点数量
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, radii, nsamples, mlps, bn=True, use_xyz=True, vote = False):
        # type: (PointnetSAModuleMSG, int, list[float], list[int], list[list[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()
        # 继承了_PointnetSAModuleBase中定义的操作方法

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]  # 多尺度就在于球查询有多种半径
            nsample = nsamples[i]
            if vote is False:
                self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
                # self.add_module(str(len(self)), module)
            else:  # 如果要霍夫投票，就要计算得分
                self.groupers.append(
                pointnet2_utils.QueryAndGroup_score(radius, nsample, use_xyz=use_xyz))
            
            mlp_spec = mlps[i]
            if use_xyz:  # 如果要使用坐标信息，则MLP的输入要加三个
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):  # 点集提取层
    r"""Pointnet set abstraction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, radius=None, nsample=None, bn=True, use_xyz=True):
        # type: (PointnetSAModule, list[int], float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetProposalModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    mlp : list
        Spec of the pointnet before the global max_pool
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    bn : bool
        Use batchnorm
    vote : bool

    """

    def __init__(
        self, mlp, radius=None, nsample=None, bn=True, use_xyz=True, vote = True
    ):
        # type: (PointnetSAModule, list[int], int, float, int, bool, bool) -> None
        super(PointnetProposalModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            vote = vote  # 相比上一个多一个投票的选项
        )


    def forward(self, xyz, features, npoint, score):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor, torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features
        score : torch.Tensor


        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        self.npoint = npoint
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()  # B * 3 * N
        
        new_xyz = (
            pointnet2_utils.gather_operation(
                #xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                xyz_flipped, torch.arange(self.npoint).repeat(xyz.size(0),1).int().cuda()
            )
            .transpose(1, 2)
            .contiguous()
        )  # 输入B * 3 * N和B * self.npoint，输出new_xyz是B * self.npoint * C

        for i in range(len(self.groupers)):
            new_features, score_id = self.groupers[i](xyz, new_xyz, score, features)  # (B, C, npoint, nsample)
            #score_id = new_features[:,3,:,:].sum(dim = 2).argmax(dim = 1)

            #B
            #new_features_cpu = new_features.squeeze(0).detach().cpu().numpy()
            #np.savetxt('vote4.txt',new_features_cpu[0:4,i,:])
            idx = torch.arange(new_features.size(0))
            new_features = new_features[idx, :, score_id, :]
            #B*C*nsample
            new_features = new_features.unsqueeze(2)
            #B*C*1*nsample
            new_xyz = new_xyz[idx, score_id, :]
            #B*3

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1).squeeze(-1)  # (B, mlp[-1])

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetFPModule(nn.Module):
    r"""Propogates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
    # def __init__(self):  # 不用mlp处理
        ## type: (PointnetFPModule, list[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknown_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknown_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propogated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propogated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)  # 三点最近邻
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 计算权重

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)  # 插值
            
        else:
            interpolated_feats = known_feats.expand(*(known_feats.size()[0: 2] + [unknown.size(1)]))

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
        # return new_features  # 不用mlp处理

class PointnetFPModule2(nn.Module):
    r"""Propogates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self):
    # def __init__(self):  # 不用mlp处理
        ## type: (PointnetFPModule, list[int], bool) -> None
        super(PointnetFPModule2, self).__init__()
        # self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknown_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propogated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propogated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)  # 三点最近邻
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 计算权重

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)  # 插值
            
        else:
            interpolated_feats = known_feats.expand(*(known_feats.size()[0: 2] + [unknown.size(1)]))

        new_features = interpolated_feats

        # new_features = new_features.unsqueeze(-1)
        # new_features = self.mlp(new_features)

        # return new_features.squeeze(-1)
        return new_features  # 不用mlp处理

if __name__ == "__main__":
    # from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(2, 9, 3).cuda()
    xyz_feats = torch.randn(2, 9, 6).cuda()
    xyz.requires_grad_(True)
    xyz_feats.requires_grad_(True)
    # xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    # xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]])
    test_module.cuda()
    print(test_module(xyz, xyz_feats, npoint=2))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats, npoint=2)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
