from torch.utils.data import Dataset
from data_classes import PointCloud, Box
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import json 
import os
import torch
from tqdm import tqdm
import kitty_utils as utils
from kitty_utils import getModel
from searchspace import KalmanFiltering
import logging


class kittiDataset():

    def __init__(self, path):  # 指定数据位置啦
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")

    def getSceneID(self, split):  # 划分训练、验证、测试集
        if "TRAIN" in split.upper():  # Training Set 17个序列
            if "TINY" in split.upper():  # 小数据集
                sceneID = [0]
            else:
                sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set 2个序列
            if "TINY" in split.upper():  # 小数据集
                sceneID = [18]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set 2个序列
            if "TINY" in split.upper():  # 小数据集
                sceneID = [19]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getBBandPC(self, anno):  # 读取边界盒和点云
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        # 相机位置矫正矩阵
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car"):  # 读取标注
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        # print(self.list_of_scene)
        list_of_tracklet_anno = []
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == category_name]  # 选出属于category_name的目标及其相关标注
            df.insert(loc=0, column="scene", value=scene)
            # 加入是属于第几个序列的信息
            for track_id in df.track_id.unique():  # 把track_id一致的标注放在一个序列中
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                # 这波是按行读取某个track_id对应的实例对象在整个序列的标注信息
                # 也即这个对象的在本序列中的轨迹标注信息
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno
        # 所以这个是所有场景中出现的所有属于category_name类的实例对象的轨迹标注信息

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
            axis=[1, 0, 0], radians=np.pi / 2)  # 这波啊，这波是X-Y平面的旋转
            # 绕Y正半轴逆时针旋转，弧度为box["rotation_y"]
            # 再绕X正半轴逆时针旋转，角度为90°
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '{:06}.bin'.format(box["frame"]))
            # ./data/traning/velodyne/box["scene"]/box["frame"].bin
            # box["scene"]，例如0020文件夹，对应第二十一个场景序列
            # box["frame"].bin，例如000001.bin
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)  # 校准
        except :
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin) 果然没了
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                        # (1, 12)reshape为(3, 4)，才可以进行前面的np.vstack操作
                        # 拼成(4, 4)
                except ValueError:
                    pass
        return data


class SiameseDataset(Dataset):

    def __init__(self,
                 input_size,
                 path,
                 split,  # 训练、验证还是测试集
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0):

        self.dataset = kittiDataset(path=path)  # 读入kitti数据集相关部分

        self.input_size = input_size
        self.split = split  # Train or Valid or Test
        self.sceneID = self.dataset.getSceneID(split=split)
        # 确定读取的是训练集、验证集或者测试集
        self.getBBandPC = self.dataset.getBBandPC

        self.category_name = category_name
        self.regress = regress

        self.list_of_tracklet_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name) 
        # 数据集中所有场景中出现的所有属于category_name类的实例对象的轨迹标注信息
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]  # 把属于category_name类的每个实例对象的标注都拆开拉通成一个标注列表

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class SiameseTrain(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 sigma_Gaussian=1,
                 offset_BB=0,
                 scale_BB=1.0):
        # 主要预加载点云和模板点云的训练集
        super(SiameseTrain, self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB)  # 从SiameseDataset父类继承属性

        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.saved_train_BBs_dir = '/mnt/ssd-data/RUNNING/data/train_data.list_of_BBs.npy'
        self.saved_train_PCs_dir = '/mnt/ssd-data/RUNNING/data/train_data.list_of_PCs.npy'
        self.saved_valid_BBs_dir = '/mnt/ssd-data/RUNNING/data/valid_data.list_of_BBs.npy'
        self.saved_valid_PCs_dir = '/mnt/ssd-data/RUNNING/data/valid_data.list_of_PCs.npy'
        # self.saved_train_BBs_dir = "E:\RUNNING\data\\train_data.list_of_BBs.npy"
        # self.saved_train_PCs_dir = "E:\RUNNING\data\\train_data.list_of_PCs.npy"
        # self.saved_valid_BBs_dir = "E:\RUNNING\data\\valid_data.list_of_BBs.npy"
        # self.saved_valid_PCs_dir = "E:\RUNNING\data\\valid_data.list_of_PCs.npy"
        # self.saved_model_PCs_dir = '/media/zhouxiaoyu/本地磁盘/RUNNING/P2B/train_data.model_PC.npy'
        # self.saved_annos_dir = '/media/zhouxiaoyu/本地磁盘/RUNNING/P2B/train_data.list_of_anno.npy'
        self.num_candidates_perframe = 4  # 每帧的候选目标数目？

        logging.info("preloading PC...")
        self.list_of_PCs = [None] * len(self.list_of_anno)
        # self.list_of_anno从SiameseDataset继承过来
        self.list_of_BBs = [None] * len(self.list_of_anno)
        if np.load(self.saved_train_PCs_dir, allow_pickle=True) is not None:
            if self.split == 'Train':
                self.list_of_PCs = np.load(self.saved_train_PCs_dir, allow_pickle=True).tolist()
                self.list_of_BBs = np.load(self.saved_train_BBs_dir, allow_pickle=True).tolist()
            else:
                self.list_of_PCs = np.load(self.saved_valid_PCs_dir, allow_pickle=True).tolist()
                self.list_of_BBs = np.load(self.saved_valid_BBs_dir, allow_pickle=True).tolist()          
        else:
            for index in tqdm(range(len(self.list_of_anno)), desc='all annotations'):
                anno = self.list_of_anno[index]  # 获取标注
                PC, box = self.getBBandPC(anno)  # 根据标注得到点云和边界盒
                # 点云4维，边界盒17维（可参考dataset_class.Box.__repr__）
                new_PC = utils.cropPC(PC, box, offset=10)  # 根据缩放和偏移后的边界盒裁切点云

                self.list_of_PCs[index] = new_PC
                self.list_of_BBs[index] = box
        logging.info("PC preloaded!")

        logging.info("preloading Model..")
        self.model_PC = [None] * len(self.list_of_tracklet_anno)
        # len_model_PC = [None] * len(self.list_of_tracklet_anno)
        # 保存模板点云的空列表
        # 长度为数据集包含的所有序列里属于某一类的实例对象的个数
        for i in tqdm(range(len(self.list_of_tracklet_anno)), desc='annotations of a certain instance'):
            list_of_anno = self.list_of_tracklet_anno[i]
            # 读出某一实例对象的所有标注
            PCs = []
            BBs = []
            cnt = 0
            for anno in list_of_anno:
                this_PC, this_BB = self.getBBandPC(anno)
                PCs.append(this_PC)
                BBs.append(this_BB)
                anno["model_idx"] = i
                # 在anno的pd.Series里面加入model_idx属性
                # 表示是第几个实例对象
                anno["relative_idx"] = cnt
                # 在anno的pd.Series里面加入relative_idx属性
                # 表示是这个实例对象的第几个标注
                cnt += 1

            self.model_PC[i] = getModel(PCs, BBs, offset=self.offset_BB, scale=self.scale_BB)
            # len_model_PC[i] = len(self.model_PC[i])
            # 读出该对象的所有模板点云和对应真值框
        logging.info("Model preloaded!")

    def __getitem__(self, index):  # 重写__getitem__属性
        return self.getitem(index)

    def getPCandBBfromIndex(self, anno_idx):  # 根据标注索引读取点云和边界盒
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        return this_PC, this_BB

    def getitem(self, index):  # 实际getitem属性
        # 依据给定索引读取对应采样点云、标签、边界盒回归信息、真值点云
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(3)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 5])
            # 实例化之后就先根据bud执行reset方法
            # 初始化平均值、方差、数据和得分权重
            sample_offsets = gaussian.sample(1)[0]
            # 得到服从多变量正态分布的随机采样偏移

        this_anno = self.list_of_anno[anno_idx]
        this_PC, this_BB = self.getPCandBBfromIndex(anno_idx)
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)
        # 在现有标注盒基础上得到随机偏移采样边界盒

        # sample_PC = utils.cropAndCenterPC(
        #     this_PC, sample_BB, offset=self.offset_BB, scale=self.scale_BB)
        sample_PC, sample_label, sample_reg = utils.cropAndCenterPC_label(
            this_PC, sample_BB, this_BB, sample_offsets,
            offset=self.offset_BB, scale=self.scale_BB)
        # sample_PC：采样点云
        # sample_label：采样点云的正负样本标签
        # sample_reg：采样点云的边界盒回归信息
        if sample_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        # 如果采样到的点数量较少
        # 就读取一定数量的点云
        # sample_PC = utils.regularizePC(sample_PC, self.input_size)[0]
        sample_PC, sample_label, sample_reg, sample_seg_label, sample_seg_offset = \
            utils.regularizePCwithlabel(sample_PC, sample_label, sample_reg, self.input_size)

        if this_anno["relative_idx"] == 0:  # 如果这是这个实例对象的第一个标注
            prev_idx = 0  # 前一个索引值
            fir_idx = 0  # 第一个索引值
        else:
            prev_idx = anno_idx - 1  # 前一个索引值
            fir_idx = anno_idx - this_anno["relative_idx"]
            # 第一个索引值 = 当前索引值 - 在本实例中的相对索引值
        gt_PC_pre, gt_BB_pre = self.getPCandBBfromIndex(prev_idx)
        gt_PC_fir, gt_BB_fir = self.getPCandBBfromIndex(fir_idx)

        if sample_idx == 0:
            samplegt_offsets = np.zeros(3)
        else:
            samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            samplegt_offsets[2] = samplegt_offsets[2] * 5.0
        gt_BB_pre = utils.getOffsetBB(gt_BB_pre, samplegt_offsets)

        gt_PC = getModel([gt_PC_fir, gt_PC_pre],
                         [gt_BB_fir, gt_BB_pre],
                         offset=self.offset_BB,
                         scale=self.scale_BB)

        if gt_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        gt_PC = utils.regularizePC(gt_PC, self.input_size)
        # gt_PC = np.array(gt_PC.points, dtype=np.float32)
        # gt_PC = torch.from_numpy(gt_PC).float()

        return sample_PC, sample_label, sample_reg, gt_PC, sample_seg_label, sample_seg_offset

    def __len__(self):
        nb_anno = len(self.list_of_anno)
        return nb_anno * self.num_candidates_perframe
        # 采样点数为总实例个数乘以4

    def getAnnotationIndex(self, index):  # 计算标注的索引
        return int(index / (self.num_candidates_perframe))
        # 1个实例对象的标注对应4个随机采样搜索空间，所以取除以4的商

    def getSearchSpaceIndex(self, index):  # 计算搜索空间的索引
        return int(index % self.num_candidates_perframe)
        #1个实例对象的标注对应4个随机采样搜索空间，所以取除以4的余数


class SiameseTest(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0):
        super(SiameseTest, self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB)
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
        return PCs, BBs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)


if __name__ == '__main__':
    dataset_Training = SiameseTrain(
        input_size=2048,
        path='/data/qihaozhe/Kitty_data/training',
        split='Tiny_Train',
        category_name='Car',
        offset_BB=0,
        scale_BB=1.15)
    aa = dataset_Training.getitem(201)
    aa = dataset_Training.getitem(30)
    aa = dataset_Training.getitem(100)
    aa = dataset_Training.getitem(120)
    aa = dataset_Training.getitem(200)
    asdf = aa[2].numpy()