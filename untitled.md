# 网络设计方案

## 骨干网络和附加网络的搭配

- 骨干网络Pointnet++，附加网络SA-SSD改进
- 骨干网络和附加网络都是SA-SSD改进

## 实现方法

- 在P2B基础上修改

## SA-SSD多实例分割改进为仅对目标实例分割

- 考虑结合候选区域进行限定

## ==TODO==

- [ ] 首先实现主体功能
  ~~- [ ]  *pointnet_tracking.py*第186行，要输出l_xyz和l_features完整列表，作为*pointnet_modules.py*的第275行的known和known_feats~~
  - [x]  是否在PointnetFPModule里面加入mlp对提取出的new_features进行处理：
            2020.12.25根据pointnet++原文决定需要加入
  - [x] 生成search点云逐点标签label_seg，用于分割任务——~~可能会用到mmdet.ops.pts_in_boxes3d~~直接对dataset的getitem进行了操作
~~- [ ] 完善QueryAndGroup的显示功能，能显示设定的radius和nsample~~可以但没必要
~~- [ ] 候选提案框数量self.num_proposal会否影响性能~~
- [ ] train_tracking.py第162行中label_reg的self.num_proposal选取方法是否影响性能
- [ ] 论文第六页右下：

 >_Search area feature A and B did not improve or even harm the performance.  Note that we already combined template features in both conditions. This may reveal that search area features only capture spatial context rather than target clue, and hence turns useless for target-specific feature augmentation._

  说明*Search area feature*没有得到较好利用，可以从此处进行改进

- [ ] SA-SSD多实例分割改进为仅对目标实例和同类干扰实例分割

## 已知的情报

1. 先对输入点，进行聚合，得到npoint个质心new_xyz

2. 然后根据半径、采样点数nsample、原始输入点坐标xyz、质心坐标new_xyz进行求查询，得到包含所有原始输入点索引idx(B, npoint, nsample)

3. 然后根据索引对原始输入点进行分组，得到分组后的点grouped_xyz(B, 3, npoint, nsample)，减去质心坐标，相对坐标仍是(B, 3, npoint, nsample)

4. ~~如果原始各点本就带有特征，也可对特征根据idx进行分组，~~ 但这里实际上每个点只有坐标，没有特征，所以最终得到的特征new_features就是分组后的点相对坐标

5. new_features经过MLP、Maxpool等操作后，得到的就是一个scale的特征new_features_per_scale，append到一个List里面

6. ~~xyz、new_features、new_xyz、new_features_per_scale~~下一维度的search_xyz、下一维度的search_feats、当前维度的search_xyz、当前维度的new_feats结果可以分别作为unknown、unknown_feats、known、known_feats输入到self.aux_modules(list of PointnetFPModule，一个scale对应一个PointnetFPModule)中，各个scale经过3-NN~~和~~、加权插值和unit pointnet后的特征~~在最后concatenate起来~~，分别进行逐点分类得分和框回归得分的计算

7. 计算分割任务损失函数

## 学习了

- preact=False，True的时候是先BN并通过激活函数，再卷积；False的时候是先卷积，再BN并通过激活函数

## milestone

2020.12.25
完成了网络结构设计v1
后续任务是完成逐点标签数据集的生成，方法是~~移植利用'mmdet/ops/points_op/src/points_op.cpp'中的pts_in_boxes3d_cpu功能
具体使用方法参考'mmdet/models/necks/cmn.py'中的build_aux_target函数~~
加入到P2B的dataset构建中去

2020.12.28
完成了逐点分割标签和偏移标签的生成
对应kitty_utils.py的regularizePCwithlabel函数中的sample_seg_label和sample_seg_offset
