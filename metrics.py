import numpy as np
from shapely.geometry import Polygon


class AverageMeter(object):  # 
    r"""
        Computes and stores the average and current value
        计算并存储平均值和当前值    
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def estimateAccuracy(box_a, box_b, dim=3):
    if dim == 3:  # 三维情况下就是边界盒的中心距离
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:  # 二维BEV情况下就是X-Z平面上的坐标距离
        return np.linalg.norm(box_a.center[[0, 2]] - box_b.center[[0, 2]], ord=2)
    else:
        pass


def fromBoxToPoly(box):  # 把边界盒参数转化为几何信息
    return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))


def estimateOverlap(box_a, box_b, dim=2):  # 计算重叠交并比
    # if box_a == box_b:
    #     return 1.0

    Poly_anno = fromBoxToPoly(box_a)
    Poly_subm = fromBoxToPoly(box_b)

    box_inter = Poly_anno.intersection(Poly_subm)  # 返回X-Z平面交集
    box_union = Poly_anno.union(Poly_subm)  # 返回X-Z平面并集
    if dim == 2:  # 二维边界框，IoU就是交集面积除以并集面积
        return box_inter.area / box_union.area

    else:  # 三维边界盒，IoU与两个盒交集并集体积有关

        ymax = min(box_a.center[1], box_b.center[1])
        ymin = max(box_a.center[1] - box_a.wlh[2], box_b.center[1] - box_b.wlh[2])

        inter_vol = box_inter.area * max(0, ymax - ymin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]  # 两边界盒体积
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
        # overlap = inter_vol / union_vol

    return overlap


class Success(object):  # 计算并存储成功率
    """Computes and stores the Success"""

    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        # array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
        self.reset()  # 初始化重叠值列表

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val):
        self.overlaps.append(val)

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):  # 具体算成功率
        succ = [
            np.sum(i >= thres
                   for i in self.overlaps).astype(float) / self.count
            for thres in self.Xaxis
        ]  # 实际就是IoU大于每一个阈值的数量除以总的IoU数量
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap
        # np.trapz梯形积分，被积函数是value，X轴是Xaxis


class Precision(object):  # 计算并存储准确率，与成功率计算类似
    """Computes and stores the Precision"""

    def __init__(self, n=21, max_accuracy=2):
        self.max_accuracy = max_accuracy
        self.Xaxis = np.linspace(0, self.max_accuracy, n)
        self.reset()
        # max_accuracy=2?

    def reset(self):
        self.accuracies = []

    def add_accuracy(self, val):
        self.accuracies.append(val)

    @property
    def count(self):
        return len(self.accuracies)

    @property
    def value(self):
        prec = [
            np.sum(i <= thres
                   for i in self.accuracies).astype(float) / self.count
            for thres in self.Xaxis
        ]
        return np.array(prec)

    @property
    def average(self):
        if len(self.accuracies) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_accuracy