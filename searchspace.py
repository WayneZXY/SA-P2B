import numpy as np
from pomegranate import MultivariateGaussianDistribution, GeneralMixtureModel
import logging


class SearchSpace(object):  # 定义搜索区域的属性

    def reset(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def addData(self, data, score):
        return


class ExhaustiveSearch(SearchSpace):

    def __init__(self,
                 search_space=[[-3.0, 3.0], [-3.0, 3.0], [-10.0, 10.0]],
                 search_dims=[7, 7, 3]):

        x_space = np.linspace(
            search_space[0][0], search_space[0][1],
            search_dims[0])
            # array([-3., -2., -1., 0., 1., 2., 3.])

        y_space = np.linspace(
            search_space[1][0], search_space[1][1],
            search_dims[1])
            # array([-3., -2., -1., 0., 1., 2., 3.]) 

        a_space = np.linspace(
            search_space[2][0], search_space[2][1],
            search_dims[2])
            # array([-10., 0., 10.])

        X, Y, A = np.meshgrid(x_space, y_space, a_space)  # create mesh grid

        self.search_grid = np.array([X.flatten(), Y.flatten(), A.flatten()]).T

        self.reset()

    def reset(self):
        return
        # return后面没有参数，就是结束程序

    def sample(self, n=0):
        return self.search_grid


class ParticleFiltering(SearchSpace):
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.reset()

    def sample(self, n=10):
        samples = []
        for i in range(n):  # 采集n轮，组成采样点列表
            if len(self.data) > 0:
                i_mean = np.random.choice(
                    list(range(len(self.data))),
                    # 从list(range(len(self.data)))中随机抽取数字
                    p=self.score / np.linalg.norm(self.score, ord=1))
                # np.linalg.norm对求score的1范数
                # p是抽样概率，相当于给了个权重
                sample = np.random.multivariate_normal(
                    mean=self.data[i_mean], cov=np.diag(np.array(self.bnd)))
                # np.random.multivariate_normal生成多元正态分布
                # 采样点通过多元正态分布生成

            else:
                sample = np.random.multivariate_normal(
                    mean=np.zeros(len(self.bnd)),
                    cov=np.diag(np.array(self.bnd) * 3))

            samples.append(sample)
        return np.array(samples)

    def addData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = data
        self.score = score

    def reset(self):
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.ones(np.shape(self.data)[0])
        self.score = self.score / np.linalg.norm(self.score, ord=1)


class KalmanFiltering(SearchSpace):
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.reset()

    def sample(self, n=10):
        # np.random.multivariate_normal方法用于根据实际情况生成一个多元正态分布矩阵
        # self.mean是1维的，(1, 3)
        # 数组大小为(n, 3)
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def addData(self, data, score):
        score = score.clip(min=1e-5)
        # prevent sum=0 in case of bad scores
        # 最小值不小于1e-5
        # 更新下列参数
        self.data = np.concatenate((self.data, data))
        self.score = np.concatenate((self.score, score))
        self.mean = np.average(self.data, weights=self.score, axis=0)
        # 加权平均
        self.cov = np.cov(self.data.T, ddof=0, aweights=self.score)
        # ddof=0，返回简单平均值，使用aweights(analytic weights)赋予观测向量权重

    def reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.array([])


class GaussianMixtureModel(SearchSpace):

    def __init__(self, n_comp=5, dim=3):
        self.dim = dim
        self.reset(n_comp)
        # n_comp是用来生成混合模型时的components数量

    def sample(self, n=10):
        try:
            X1 = np.stack(self.model.sample(int(np.round(0.8 * n))))
            # np.round 取最近的整数
            # 从多元高斯分布中采样8个点
            if self.dim == 2:
                mean = np.mean(X1, axis=0)
                # 由X1的采样点计算均值
                std = np.diag([1.0, 1.0])
                gmm = MultivariateGaussianDistribution(mean, std)
                # 构建一个新的多元高斯分布
                X2 = np.stack(gmm.sample(int(np.round(0.1 * n))))
                # 采样1个点

                mean = np.mean(X1, axis=0)
                std = np.diag([1e-3, 1e-3])  # 标准差不同了
                gmm = MultivariateGaussianDistribution(mean, std)
                X3 = np.stack(gmm.sample(int(np.round(0.1 * n))))
                # 又采样了一个点

            else:
                mean = np.mean(X1, axis=0)
                std = np.diag([1.0, 1.0, 1e-3])
                gmm = MultivariateGaussianDistribution(mean, std)
                X2 = np.stack(gmm.sample(int(np.round(0.1 * n))))

                mean = np.mean(X1, axis=0)
                std = np.diag([1e-3, 1e-3, 10.0])
                gmm = MultivariateGaussianDistribution(mean, std)
                X3 = np.stack(gmm.sample(int(np.round(0.1 * n))))

            X = np.concatenate((X1, X2, X3))  # 多尺度采样？

        except ValueError:
            print("exception caught on sampling")
            if self.dim == 2:
                mean = np.zeros(self.dim)
                std = np.diag([1.0, 1.0])
                gmm = MultivariateGaussianDistribution(mean, std)
                X = gmm.sample(int(n))
            else:
                mean = np.zeros(self.dim)
                std = np.diag([1.0, 1.0, 5.0])
                gmm = MultivariateGaussianDistribution(mean, std)
                X = gmm.sample(int(n))
        return X

    def addData(self, data, score):
        score = score.clip(min=1e-5)
        self.data = data
        self.score = score

        score_normed = self.score / np.linalg.norm(self.score, ord=1)
        try:
            model = GeneralMixtureModel.from_samples(
                MultivariateGaussianDistribution,
                n_components=self.n_comp,
                X=self.data,  # 用来训练分布的数据
                weights=score_normed  # 每个样本的初始权重
                )
            self.model = model
        except:
            logging.info("catched an exception")

    def reset(self, n_comp=5):
        self.n_comp = n_comp
        # components数量

        if self.dim == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.ones(np.shape(self.data)[0])
        self.score = self.score / np.linalg.norm(self.score, ord=1)
        if self.dim == 2:
            self.model = MultivariateGaussianDistribution(
                np.zeros(self.dim), np.diag([1.0, 1.0]))
            # self.model是一个类实例
            # "class" :"Distribution"
            # "name" :"MultivariateGaussianDistribution",
        else:
            self.model = MultivariateGaussianDistribution(
                np.zeros(self.dim), np.diag([1.0, 1.0, 5.0]))
