import argparse  # 解析命令行的库
import os
import random
import time
import logging
# import pdb  # 用于调试的库
from tqdm import tqdm
import numpy as np
# import scipy.io as sio
from visualize import Visualizer
import torch
import torch.nn as nn
import torch.nn.parallel  # 多GPU情况下分布式训练
# import torch.backends.cudnn as cudnn  # 貌似没有用到
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler  # 可用来调整学习率
import torch.utils.data
from torch.utils.dlpack import to_dlpack
from cupy.core.dlpack import fromDlpack
# import torch.nn.functional as F
# from torch.autograd import Variable

from Dataset import SiameseTrain  # 训练数据集
from pointnet2.models import Pointnet_Tracking, Pointnet_Tracking2, Pointnet_Tracking3

from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default = 0,  help='number of input point features')
parser.add_argument('--data_dir', type=str, default = '/mnt/ssd-data/RUNNING/data/training',  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Van',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
parser.add_argument('--env', type=str, default = 'SA-P2B',  help='visdom environment')
parser.add_argument('--vis_port', type=int, default = 8097,  help='visdom port')

opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
vis = Visualizer(opt.env, port = opt.vis_port)
opt.manualSeed = 0
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)  # 设置用于生成随机数的种子，返回Torch._C.Generator对象。

save_dir = opt.save_root_dir

try:
	os.makedirs(save_dir)
except OSError:
	pass

# 记录训练信息
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
					# level=logging.INFO，日志第2等级，确认一切按预期运行
logging.info('======================================================')

# 1. Load data
train_data = SiameseTrain(
	input_size=1024,
	path=opt.data_dir,
	split='Train',
	category_name=opt.category_name,
	offset_BB=0,
	scale_BB=1.25)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)

valid_data = SiameseTrain(
    input_size=1024,
    path=opt.data_dir,
    split='Valid',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25)

test_dataloader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=int(opt.batchSize / 2),
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)

										  
print('#Train data:', len(train_data), '#Valid data:', len(valid_data))

# 2. Define model, loss and optimizer
# netR = Pointnet_Tracking3(input_channels=opt.input_feature_num, use_xyz=True, test=False)
# netR = Pointnet_Tracking(input_channels=opt.input_feature_num, use_xyz=True, test=False)
netR = Pointnet_Tracking(input_channels=opt.input_feature_num, use_xyz=True)

if opt.ngpu > 1:  # 不止一张GPU时
	netR = torch.nn.DataParallel(netR, range(opt.ngpu))
	# torch.distributed.init_process_group(backend="nccl")
	# netR = torch.nn.parallel.DistributedDataParallel(netR)
if opt.model != '':
	netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
	  
netR.cuda()
print(netR)

# 分类、回归、目标分数、BBox的损失函数
def aux_loss(pts_labels, center_targets, point_cls, point_reg):  # 附加任务的损失函数
	r'''
		pts_labels = sample_seg_label, B * 512
		center_targets = sample_seg_offset, B * 512 * 3
		point_cls = estimation_seg, B * 512
		point_reg = estimation_offset, B * 512 * 3
	'''
	N = len(pts_labels)
	rpn_cls_target = pts_labels.float()
	pos = (pts_labels > 0).float()
	neg = (pts_labels == 0).float()

	pos_normalizer = pos.sum()
	pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

	cls_weights = pos + neg
	cls_weights = cls_weights / pos_normalizer

	reg_weights = pos
	reg_weights = reg_weights / pos_normalizer

	aux_loss_cls = weighted_sigmoid_focal_loss(
		point_cls, rpn_cls_target, weight=cls_weights, avg_factor=1.)
	aux_loss_cls /= N

	aux_loss_reg = weighted_smoothl1(
		point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
	aux_loss_reg /= N

	return aux_loss_cls, aux_loss_reg

criterion_cla = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])).cuda()
criterion_reg = nn.SmoothL1Loss(reduction='none').cuda()
criterion_objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]), reduction='none').cuda()
criterion_box = nn.SmoothL1Loss(reduction='none').cuda()
optimizer = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.2)
# 将每个参数组的学习率设置为每训练Step_Size轮，学习率就乘以Gamma这个衰减系数
# first_batch = next(iter(train_dataloader))
# 3. Training and testing
for epoch in range(opt.nepoch):
	# scheduler.step(epoch)
	print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
	# 3.1 switch to train mode
	torch.cuda.synchronize()  # 等待当前设备上所有流中的所有内核完成，目的是获得准确的时间记录
	netR.train()
	train_mse = 0.0
	timer = time.time()

	batch_correct = 0.0
	batch_cla_loss = 0.0
	batch_reg_loss = 0.0
	batch_box_loss = 0.0
	batch_objective_loss = 0.0
	batch_aux_seg_loss = 0.0
	batch_aux_offset_loss = 0.0
	batch_num = 0.0
	# batch_iou = 0.0
	batch_true_correct = 0.0
	for i, data in enumerate(tqdm(train_dataloader, position=0, desc='traning datas')):
	# for i, data in enumerate(tqdm([first_batch] * 1000, position=0, desc='trianing datas')):
		# 总数为78088 = batch size * batch个数（比如8 * 9761） = 标注数量19522 * 数据类型4种
		# 用于训练的17个序列中Car实例共有441个，但属于每一个的标注量各不相同
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()  
		# 3.1.1 load inputs and targets
		# data = data.cuda()
		label_point_set, label_cla, label_reg, object_point_set, sample_seg_label, sample_seg_offset = data
		# 一个标注对应一组采样点云、分类标签、边界盒回归信息、真值点云、采样点云逐点分割标签、采样点云逐点偏移量
		label_cla = label_cla.cuda()
		label_reg = label_reg.cuda()
		object_point_set = object_point_set.cuda()
		label_point_set = label_point_set.cuda()
		sample_seg_label = sample_seg_label.cuda()
		sample_seg_offset = sample_seg_offset.cuda()
		label_point_set.requires_grad_(False)  # 对应search，大小为B * 512 * 3
		label_cla.requires_grad_(False)  # 对应分类标签，大小为B * 64
		label_reg.requires_grad_(False)  # 对应边界框回归信息，大小为B * 64 * [三维坐标，偏移]，即B * 64 * 4
		object_point_set.requires_grad_(False)  # 对应template，大小为B * 256 * 3
		sample_seg_label.requires_grad_(False)  # 对应search的逐点标签，用于附加分割任务，大小为B * 512
		sample_seg_offset.requires_grad_(False)  # 对应search的逐点偏移量，用于附加回归任务，大小为B * 512 * 3

		# 3.1.2 compute output
		optimizer.zero_grad()
		estimation_cla, estimation_reg, estimation_box, center_xyz, estimation_seg, estimation_offset = \
			netR(object_point_set, label_point_set)
		# 分类相似度B * 64，参与投票的点坐标B * 64 * 3，边界框估计结果B * 64 * 5，候选提案框中心坐标B * 64 * 3，
		# 附加任务逐点分割得分B * 512，附加任务逐点偏移估计值B * 3 * 512
		# label_cla = label_cla[:, :64]
		loss_cla = criterion_cla(estimation_cla, label_cla)
		loss_reg = criterion_reg(estimation_reg, label_reg[:, :, 0: 3])
		loss_reg = (loss_reg.mean(2) * label_cla).sum() / (label_cla.sum() + 1e-6)
		# .mean(2)返回第三维的平均值，与label_cla相乘，只让属于Car的样本的回归损失起作用

		dist = torch.sum((center_xyz - label_reg[:, 0: 64, 0: 3]) ** 2, dim=-1)  # B * 64
		dist = torch.sqrt(dist + 1e-6)
		B = dist.size(0)
		K = 64
		objectness_label = torch.zeros((B, K), dtype=torch.float).cuda()
		objectness_mask = torch.zeros((B, K)).cuda()
		objectness_label[dist < 0.3] = 1  # dist < 0.3的候选提案是正样本
		objectness_mask[dist < 0.3] = 1  # dist < 0.3的正样本和dist > 0.6的负样本都要在计算损失时起作用
		objectness_mask[dist > 0.6] = 1
		loss_objective = criterion_objective(estimation_box[:, :, 4], objectness_label)  # 把proposal-wise targetness得分拿来计算损失
		loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
		loss_box = criterion_box(estimation_box[:, :, 0: 4], label_reg[:, 0: 64, :])  # 把提案框坐标加X-Y面旋转拿来计算损失
		loss_box = (loss_box.mean(2) * objectness_label).sum() / (objectness_label.sum() + 1e-06)	
		loss_aux_seg, loss_aux_offset = aux_loss(sample_seg_label, sample_seg_offset, estimation_seg, estimation_offset)	
		loss_aux_seg = loss_aux_seg.squeeze(0)
		loss_aux_offset = loss_aux_offset.squeeze(0)
		# loss = loss_reg + 0.2 * loss_cla + 1.5 * loss_objective + 0.2 * loss_box + 1.8 * loss_aux_seg + 4 * loss_aux_offset
		loss = loss_reg + 0.2 * loss_cla + 1.5 * loss_objective + loss_box + 0.9 * loss_aux_seg + 2 * loss_aux_offset		
		# loss = 2 * loss_reg + 0.4 * loss_cla + 3 * loss_objective + 0.4 * loss_box + 0.9 * loss_aux_seg + 2 * loss_aux_offset

		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		optimizer.step()
		torch.cuda.synchronize()
		
		# 3.1.4 update training error
		# estimation_cla_cpu = estimation_cla.sigmoid().detach().cpu().numpy()
		# label_cla_cpu = label_cla.detach().cpu().numpy()
		estimation_cla_cpu = estimation_cla.sigmoid().detach()
		label_cla_cpu = label_cla.detach()	
		# correct = float(np.sum((estimation_cla_cpu[0: len(label_point_set), :] > 0.4) == label_cla_cpu[0: len(label_point_set), :])) / 256.0
		correct = torch.sum(((estimation_cla_cpu[0: len(label_point_set), :] > 0.4).float() == label_cla_cpu[0: len(label_point_set), :]).float()) / 128.0		
		# 计算batch中正确分类的比例
		# true_correct = float(np.sum((np.float32(estimation_cla_cpu[0: len(label_point_set), :] > 0.4)
		# + label_cla_cpu[0: len(label_point_set), :]) == 2) / (np.sum(label_cla_cpu[0: len(label_point_set), :])))
		true_correct = torch.sum(((estimation_cla_cpu[0: len(label_point_set), :] > 0.4).float()
		+ label_cla_cpu[0: len(label_point_set), :] == 2).float()) / torch.sum(label_cla_cpu[0: len(label_point_set), :])
		# 计算batch中正样本正确分类的比例
					
		train_mse = train_mse + loss.data * len(label_point_set)
		batch_correct += correct
		batch_cla_loss += loss_cla.data
		batch_reg_loss += loss_reg.data
		batch_box_loss += loss_box.data
		batch_objective_loss += loss_objective.data		
		batch_aux_seg_loss += loss_aux_seg.data
		batch_aux_offset_loss += loss_aux_offset.data
		batch_num += len(label_point_set)
		batch_true_correct += true_correct
		show_num = 10  # 20个batch显示一次
		if (i + 1) % show_num == 0:
			print('\n ---- batch: %03d of epoch: %03d ----' % (i + 1, epoch + 1))
			print('cla_loss: %f, reg_loss: %f, box_loss: %f, aux_seg_loss: %f, aux_offset_loss: %f' \
			% (batch_cla_loss / show_num, batch_reg_loss / show_num, batch_box_loss / show_num, \
				batch_aux_seg_loss / show_num, batch_aux_offset_loss / show_num))
			print('accuracy: %f' % (batch_correct / float(batch_num)))
			print('true accuracy: %f' % (batch_true_correct / show_num))			
			# vis.plot('cla_loss', (batch_cla_loss / show_num).item())
			# vis.plot('reg_loss', (batch_reg_loss / show_num).item())
			# vis.plot('box_loss', (batch_box_loss / show_num).item())
			# vis.plot('aux_seg_loss', (batch_aux_seg_loss / show_num).item())
			# vis.plot('aux_offset_loss', (batch_aux_offset_loss / show_num).item())
			# vis.plot('accuracy', (batch_correct / float(batch_num)))
			# vis.plot('true accuracy', (batch_true_correct / show_num))
			vis.plot('cla_loss', fromDlpack(to_dlpack(batch_cla_loss / show_num)).item())
			vis.plot('reg_loss', fromDlpack(to_dlpack(batch_reg_loss / show_num)).item())
			vis.plot('box_loss', fromDlpack(to_dlpack(batch_box_loss / show_num)).item())
			vis.plot('objective_loss', fromDlpack(to_dlpack(batch_objective_loss / show_num)).item())			
			vis.plot('aux_seg_loss', fromDlpack(to_dlpack(batch_aux_seg_loss / show_num)).item())
			vis.plot('aux_offset_loss', fromDlpack(to_dlpack(batch_aux_offset_loss / show_num)).item())
			vis.plot('accuracy', fromDlpack(to_dlpack(batch_correct / float(batch_num))).item())
			vis.plot('true accuracy', fromDlpack(to_dlpack(batch_true_correct / show_num)).item())		
			batch_correct = 0.0
			batch_cla_loss = 0.0
			batch_reg_loss = 0.0
			batch_box_loss = 0.0
			batch_aux_seg_loss = 0.0
			batch_aux_offset_loss = 0.0
			batch_num = 0.0
			batch_true_correct = 0.0
           
	# time taken within one epoch
	train_mse = train_mse / len(train_data)
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
	print('netR_ %d.pth is ready' % (epoch))
	torch.save(netR.state_dict(), '%s/netR_%d_%s.pth' % (save_dir, epoch, opt.category_name))
	torch.save(optimizer.state_dict(), '%s/optimizer_%d_%s.pth' % (save_dir, epoch, opt.category_name))
	
	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	netR.eval()
	test_cla_loss = 0.0
	test_reg_loss = 0.0
	test_box_loss = 0.0
	test_correct = 0.0
	test_true_correct = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(test_dataloader, position=0, desc='evaluating datas')):
		# 总数为5416 = batch size * batch个数（比如4 * 1354） = 标注数量1354 * 数据类型4种
		# 用于验证的2个序列中Car实例共有18个，但属于每一个的标注量各不相同
		torch.cuda.synchronize()
		# 3.2.1 load inputs and targets
		val_label_point_set, val_label_cla, val_label_reg, val_object_point_set, _, _ = data
		val_label_point_set = val_label_point_set.cuda()
		val_label_reg = val_label_reg.cuda()
		val_object_point_set = val_object_point_set.cuda()
		val_label_cla = val_label_cla.cuda()
		val_label_point_set.requires_grad_(False)  # 对应search，大小为B * 512 * 3
		val_label_cla.requires_grad_(False)  # 对应分类标签，大小为B * 64
		val_label_reg.requires_grad_(False)  # 对应边界框回归信息，大小为B * 64 * [三维坐标，偏移]，即B * 64 * 4
		val_object_point_set.requires_grad_(False)  # 对应template，大小为B * 256 * 3

		# 3.2.2 compute output
		# val_label_cla = val_label_cla[:, :64]
		val_estimation_cla, val_estimation_reg, val_estimation_box, val_center_xyz, _, _ = netR(val_object_point_set, val_label_point_set)
		val_loss_cla = criterion_cla(val_estimation_cla, val_label_cla)
		val_loss_reg = criterion_reg(val_estimation_reg, val_label_reg[:, :, 0: 3])
		val_loss_reg = (val_loss_reg.mean(2) * val_label_cla).sum() / (val_label_cla.sum() + 1e-06)

		dist = torch.sum((val_center_xyz - val_label_reg[:, 0: 64 ,0: 3]) ** 2, dim=-1)
		dist = torch.sqrt(dist + 1e-6)
		B = dist.size(0)
		K = 64
		val_objectness_label = torch.zeros((B, K), dtype=torch.float).cuda()
		val_objectness_mask = torch.zeros((B, K)).cuda()
		val_objectness_label[dist < 0.3] = 1
		val_objectness_mask[dist < 0.3] = 1
		val_objectness_mask[dist > 0.6] = 1
		val_loss_objective = criterion_objective(val_estimation_box[:, :, 4], val_objectness_label)
		val_loss_objective = torch.sum(val_loss_objective * val_objectness_mask) / (torch.sum(val_objectness_mask) + 1e-6)
		val_loss_box = criterion_box(val_estimation_box[:, :, 0: 4], val_label_reg[:, 0: 64, :])
		val_loss_box = (val_loss_box.mean(2) * val_objectness_label).sum() / (val_objectness_label.sum() + 1e-06)
		val_loss = 0.2 * val_loss_cla + val_loss_reg + 1.5 * val_loss_objective + 0.2 * val_loss_box

		torch.cuda.synchronize()
		test_cla_loss = test_cla_loss + val_loss_cla.data * len(val_label_point_set)
		test_reg_loss = test_reg_loss + val_loss_reg.data * len(val_label_point_set)
		test_box_loss = test_box_loss + val_loss_box.data * len(val_label_point_set)
		# val_estimation_cla_cpu = val_estimation_cla.sigmoid().detach().cpu().numpy()
		val_estimation_cla_cpu = val_estimation_cla.sigmoid().detach()
		# val_label_cla_cpu = val_label_point_set.detach().cpu().numpy()
		val_label_cla_cpu = val_label_cla.detach()
		# correct = float(np.sum((val_estimation_cla_cpu[0: len(val_label_point_set), :] > 0.4) == val_label_cla_cpu[0: len(val_label_point_set), :])) / 256.0
		correct = torch.sum(((val_estimation_cla_cpu[0: len(val_label_point_set), :] > 0.4).float() == val_label_cla_cpu[0: len(val_label_point_set), :]).float()) / 128.0
		# true_correct = float(np.sum((np.float32(val_estimation_cla_cpu[0: len(val_label_point_set), :] > 0.4)
		# + val_label_cla_cpu[0: len(val_label_point_set), :]) == 2) / (np.sum(val_label_cla_cpu[0: len(val_label_point_set), :])))
		true_correct = torch.sum(((val_estimation_cla_cpu[0: len(val_label_point_set), :] > 0.4).float()
		+ val_label_cla_cpu[0: len(val_label_point_set), :] == 2).float()) / torch.sum(val_label_cla_cpu[0: len(val_label_point_set), :])	
		test_correct += correct
		test_true_correct += true_correct * len(val_label_point_set)
	scheduler.step(epoch)  

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(valid_data)
	print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
	# print mse
	test_cla_loss = test_cla_loss / len(valid_data)
	test_reg_loss = test_reg_loss / len(valid_data)
	test_box_loss = test_box_loss / len(valid_data)
	print('cla_loss: %f, reg_loss: %f, box_loss: %f, #valid_data = %d' % (test_cla_loss, test_reg_loss, test_box_loss, len(valid_data)))
	test_correct = test_correct / len(valid_data)
	print('mean-correct of 1 sample: %f, #valid_data = %d' % (test_correct, len(valid_data)))
	test_true_correct = test_true_correct / len(valid_data)
	print('true correct of 1 sample: %f' % (test_true_correct))
	# log
	logging.info('Epoch#%d: train error=%e, test error=%e, %e, %e, test correct=%e, %e, lr = %f'
					% (epoch,
						train_mse,
						test_cla_loss, 
						test_reg_loss,
						test_box_loss, 
						test_correct, 
						test_true_correct, 
						scheduler.get_lr()[0])
					)
logging.info('Training accomplished!')
print('Training accomplished!')