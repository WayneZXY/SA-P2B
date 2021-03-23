import torch
import os
import copy
import numpy as np
from pyquaternion import Quaternion
from data_classes import PointCloud
from metrics import estimateOverlap
# from scipy.optimize import leastsq  # 使一组方程的平方和最小化 MSE


def distanceBB_Gaussian(box1, box2, sigma=1):
    off1 = np.array([box1.center[0], box1.center[2], Quaternion(matrix=box1.rotation_matrix).degrees])  # 把角度也算进来了
    off2 = np.array([box2.center[0], box2.center[2], Quaternion(matrix=box2.rotation_matrix).degrees])
    dist = np.linalg.norm(off1 - off2)  # ord=None，默认Frobenius norm
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return score


# IoU or Gaussian score map
def getScoreGaussian(offset, sigma=1):
    coeffs = [1, 1, 1 / 5]
    dist = np.linalg.norm(np.multiply(offset, coeffs))  # np.multiply是矩阵对应位置相乘
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return torch.tensor([score])


def getScoreIoU(a, b):
    score = estimateOverlap(a, b)
    return torch.tensor([score])


def getScoreHingeIoU(a, b):  # 也许对学习负样本有好处？
    score = estimateOverlap(a, b)
    if score < 0.5:
        score = 0.0
    return torch.tensor([score])


def getOffsetBB(box, offset):
    r'''
        根据偏移得到新框
    '''
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    # REMOVE PREVIOUS TRANSfORMATION FIRST
    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)
    # print(new_box.center)

    # 旋转
    if len(offset) == 3:
        new_box.rotate(Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    elif len(offset) == 4:
        new_box.rotate(Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))
    if offset[0] > new_box.wlh[0]:
        offset[0] = np.random.uniform(-1, 1)
    if offset[1] > min(new_box.wlh[1], 2):
        offset[1] = np.random.uniform(-1, 1)
    # 限定一下offset的大小，超出范围的从N(-1, 1)中随机生成
    new_box.translate(np.array([offset[0], offset[1], 0]))  # 平移

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def getOffsetBB2(box, offset):
    r'''
        根据偏移得到新框
    '''
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    # REMOVE PREVIOUS TRANSfORMATION FIRST
    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)
    # print(new_box.center)

    # 旋转
    if len(offset) == 3:
        new_box.rotate(Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    elif len(offset) == 4:
        new_box.rotate(Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))
    if offset[0] > new_box.wlh[0]:
        offset[0] = np.random.uniform(-1, 1)
    if offset[1] > min(new_box.wlh[1], 2):
        offset[1] = np.random.uniform(-1, 1)
    # 限定一下offset的大小，超出范围的从N(-1, 1)中随机生成
    new_box.translate(np.array([offset[0], offset[1], 0]))  # 平移

    # APPLY PREVIOUS TRANSFORMATION
    # new_box.rotate(rot_quat)
    # new_box.translate(trans)
    return new_box


def voxelize(PC, dim_size=[48, 108, 48]):  # 似乎没用到
    '''体素化'''
    # PC = normalizePC(PC)
    if np.isscalar(dim_size):  # 如果dim_size只是一个数字
        dim_size = [dim_size] * 3
    dim_size = np.atleast_2d(dim_size).T
    # [[48, 108, 48]].T
    # 加一维应该是要与点云的格式相匹配
    # 因为data_classes.PointCloud类处理之后的点云是(3, n)大小的
    PC = (PC + 0.5) * dim_size  # 点云都放入体素中
    # truncate to integers
    xyz = PC.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dim_size), 0)  # 逗号后面的0表示axis=0
    # ~运算符，补码取反
    # 对任何范围之外，值却非0的点，去掉
    xyz = xyz[:, valid_ix]
    out = np.zeros(dim_size.flatten(), dtype=np.float32)
    # out是一个一维全0数组
    # out.shape = (48, 108, 48)
    out[tuple(xyz)] = 1  # 有点云的体素取为1
    # print(out)
    return out


def regularizePC_test(PC, input_size, istrain=False):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:  # 点数目三个及以上
        if PC.shape[0] > 3:  # 只要前三位坐标
            PC = PC[0: 3, :]
        if PC.shape[1] != input_size:
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape((3, input_size)).T
        # 回归向量纵向复制size(label)次，实际是128次，拼成新的reg，大小为128*4

    else:
        PC = np.zeros((3, input_size)).T

    return torch.from_numpy(PC).float()


def regularizePC(PC, input_size, istrain=True):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:  # 两个以上点
        if PC.shape[0] > 3:
            PC = PC[0: 3, :]  # 只取前三维（点云坐标）
        if PC.shape[1] != int(input_size / 2):
        # if PC.shape[1] != int(input_size):
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=int(input_size / 2), dtype=np.int64)
            # new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=int(input_size), dtype=np.int64)
            PC = PC[:, new_pts_idx]  # 如果PC.shape[1] < input_size / 2，取的点会有重复，但还是input_size / 2个
        PC = PC.reshape((3, int(input_size / 2))).T
        # PC = PC.reshape((3, int(input_size))).T
        # PC变成了一个int(input_size/2) * 3的tensor，只取input_size / 2个点

    else:  # 只有一个点
        PC = np.zeros((3, int(input_size / 2))).T
        # PC = np.zeros((3, int(input_size))).T

    return torch.from_numpy(PC).float()


def regularizePCwithlabel(PC, label, reg, input_size, istrain=True):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:  # 点数目三个及以上
        if PC.shape[0] > 3:  # 只要前三位坐标
            PC = PC[0: 3, :]
        if PC.shape[1] != input_size:
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
            label = label[new_pts_idx]
        sample_seg_label = copy.deepcopy(label)  # 用于附加任务的采样点云逐点分割标签
        PC = PC.reshape((3, input_size)).T
        sample_seg_offset = PC - reg[:3]  # 用于附加任务的采样点云逐点偏移标签
        # label = label[0: 64]
        label = label[0: 128]
        reg = np.tile(reg, [np.size(label), 1])
        # 回归向量纵向复制size(label)次，实际是128次，拼成新的reg，大小为128*4

    else:
        PC = np.zeros((3, input_size)).T
        # label = np.zeros(64)
        label = np.zeros(128)
        sample_seg_offset = PC - reg[:3]
        reg = np.tile(reg,[np.size(label),1])
        sample_seg_label = np.zeros(input_size)

    return torch.from_numpy(PC).float(), torch.from_numpy(label).float(), torch.from_numpy(reg).float(), \
                torch.from_numpy(sample_seg_label).float(), torch.from_numpy(sample_seg_offset).float()


def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):
    r'''
        读取模板点云
    '''
    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))  # array([], shape=(3, 0), dtype=float64)

    for PC, box in zip(PCs, boxes):
        cropped_PC = cropAndCenterPC(PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

    PC = PointCloud(points)

    return PC


def cropPC(PC, box, offset=0, scale=1.0):
    r'''
        根据缩放和偏移后的边界盒截取点云
    '''
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), axis=1) + offset
    mini = np.min(box_tmp.corners(), axis=1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC

def getlabelPC(PC, box, offset=0, scale=1.0):
    '''获取点云逐点样本标签'''
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate((rot_mat))
    box_tmp.rotate(Quaternion(matrix=(rot_mat)))
    
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_label = np.zeros(new_PC.points.shape[1])
    new_label[close] = 1
    # 满足close条件，在范围里面的点云
    # 对应的label为1，即为正样本
    return new_label

def cropPCwithlabel(PC, box, label, offset=0, scale=1.0):
    '''截取点云并更新标签'''
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), axis=1) + offset
    mini = np.min(box_tmp.corners(), axis=1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])  
    # 在范围内的点被输入到PointCloud类里面封装成点云
    new_label = label[close]
    # 在范围内的点对应的标签被留下
    return new_PC, new_label

def weight_process(include, low, high):
    '''计算权重'''
    if include < low:
        weight = 0.7
    elif include > high:
        weight = 1
    else:
        weight = (include * 2.0 + 3.0 * high - 5.0 * low) / (5 * (high - low))
    return weight

def func(a, x):
    k, b = a
    return k * x + b

def dist(a, x, y):
    return func(a, x) - y

def weight_process2(k):
    k = abs(k)
    if k > 1:
        weight = 0.7
    else:
        weight = 1 - 0.3 * k
    return weight

def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):
    '''截取点云并校准'''
    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)  # 除以框的宽长高
    return new_PC

def Centerbox(sample_box, gt_box):
    '''平移校准边界盒'''
    rot_mat = np.transpose(gt_box.rotation_matrix)
    trans = -gt_box.center

    new_box = copy.deepcopy(sample_box)
    new_box.translate(trans)
    return new_box


def cropAndCenterPC_label(PC,
                          sample_box,
                          gt_box,
                          sample_offsets,
                          offset=0,
                          scale=1.0,
                          normalize=False):
    '''截取并校准采样点云、获取标签'''
    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)
    # 根据采样边界框，加两倍偏移、四倍scale截取一个粗略大范围的点云

    new_box = copy.deepcopy(sample_box)  # 复制采样盒为新边界盒
    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    # 根据真值边界盒获取采样点云的正负样本标签，0或者1
    new_box_gt = copy.deepcopy(gt_box)
    # 复制真值边界盒为新真值盒
    # new_box_gt2 = copy.deepcopy(gt_box)

    #rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    # 先去掉原先采样框带来的变换
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset + 2.0, scale=1 * scale)
    # 在之前较大范围点云中再根据带偏移采样框的小范围点云，作为最终的采样点云，同时留下这些点云对应的标签
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    label_reg = [new_box_gt.center[0],\
                 new_box_gt.center[1],\
                 new_box_gt.center[2],\
                 -sample_offsets[2]]
                 # 框回归标签：真值盒的中心点坐标、采样偏移[2]
    label_reg = np.array(label_reg)
    # label_reg = np.tile(label_reg,[np.size(new_label),1])

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC, new_label, label_reg

def cropAndCenterPC_label_test_time(PC, sample_box, offset=0, scale=1.0):
    '''测试中使用的cropAndCenterPC_label，没有真值盒和采样偏移'''
    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset+2.0, scale=scale)

    return new_PC

def cropAndCenterPC_label_test(PC, sample_box, gt_box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    new_box_gt = copy.deepcopy(gt_box)
    # new_box_gt2 = copy.deepcopy(gt_box)

    rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))
    # new_box_gt2.translate(trans)
    # new_box_gt2.rotate(rot_quat.inverse)

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+2.0, scale=1 * scale)
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    label_reg = [new_box_gt.center[0], new_box_gt.center[1], new_box_gt.center[2]]
    label_reg = np.array(label_reg)
    # label_reg = (new_PC.points - np.tile(new_box_gt.center,[np.size(new_label),1]).T) * np.expand_dims(new_label, axis=0)
    # newoff = [new_box_gt.center[0],new_box_gt.center[1],new_box_gt.center[2]]
    # newoff = np.array(newoff)

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC, new_label, label_reg, new_box, new_box_gt

def distanceBB(box1, box2):
    '''计算边界盒距离'''
    eucl = np.linalg.norm(box1.center - box2.center)
    angl = Quaternion.distance(
        Quaternion(matrix=box1.rotation_matrix),
        Quaternion(matrix=box2.rotation_matrix))
    return eucl + angl


def generate_boxes(box, search_space=[[0, 0, 0]]):
    '''# Generate more candidate boxes based on prior and search space
    # 基于先验和搜索空间确定更多候选框
    # Input : Prior position, search space and search size
    # Output : List of boxes'''

    candidate_boxes = [getOffsetBB(box, offset) for offset in search_space]
    return candidate_boxes


def getDataframeGT(anno):
    '''根据标注获取真值dataframe'''
    df = {
        "scene": anno["scene"],  # 场景（序列）编号，如0000
        "frame": anno["frame"],  # 帧编号，如000001
        "track_id": anno["track_id"],  # 与实例对应的编号
        # 以上三个都是路径或者文件名
        "type": anno["type"],  # 实例类型，如car/Van/Pedestrian
        "truncated": anno["truncated"],  # 截断flag
        # 0表示没截断，1表示截断了，截断表示该实例点云超出成像范围
        "occluded": anno["occluded"],  # 遮挡flag
        # 0表示完全看得见，1表示部分遮挡，2表示大部分遮挡，3表示未知
        "alpha": anno["alpha"],
        # 表示观察实例对象的角度，范围为[-pi,pi]
        "bbox_left": anno["bbox_left"],  # 边界盒坐标信息
        "bbox_top": anno["bbox_top"],
        "bbox_right": anno["bbox_right"],
        "bbox_bottom": anno["bbox_bottom"],
        "height": anno["height"],  # 边界盒高宽长信息
        "width": anno["width"],
        "length": anno["length"],
        "x": anno["x"],  # 边界盒中心坐标信息
        "y": anno["y"],
        "z": anno["z"],
        "rotation_y": anno["rotation_y"]  # 关于相机坐标系y轴的旋转信息
    }
    return df


def getDataframe(anno, box, score):
    '''根据实时box获取dataframe'''
    myquat = (box.orientation * Quaternion(axis=[1, 0, 0], radians=-np.pi / 2))
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": 0.0,
        "bbox_left": 0.0,
        "bbox_top": 0.0,
        "bbox_right": 0.0,
        "bbox_bottom": 0.0,
        "height": box.wlh[2],
        "width": box.wlh[0],
        "length": box.wlh[1],
        "x": box.center[0],
        "y": box.center[1] + box.wlh[2] / 2,
        "z": box.center[2],
        "rotation_y":
        np.sign(myquat.axis[1]) * myquat.radians,  # this_anno["rotation_y"], #
        "score": score
    }
    return df


def saveTrackingResults(df_3D, dataset_loader, export=None, epoch=-1):
    '''保存跟踪结果'''
    for i_scene, scene in enumerate(df_3D.scene.unique()):
        new_df_3D = df_3D[df_3D["scene"] == scene]
        new_df_3D = new_df_3D.drop(["scene"], axis=1)
        new_df_3D = new_df_3D.sort_values(by=['frame', 'track_id'])

        os.makedirs(os.path.join("results", export, "data"), exist_ok=True)
        if epoch == -1:
            path = os.path.join("results", export, "data", "{}.txt".format(scene))
        else:
            path = os.path.join("results", export, "data",
                                "{}_epoch{}.txt".format(scene,epoch))
        new_df_3D.to_csv(
            path, sep=" ", header=False, index=False, float_format='%.6f')