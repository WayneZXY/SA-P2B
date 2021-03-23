import time
import os
import logging
import argparse
import random
import copy
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import torch

import kitty_utils as utils
from Dataset import SiameseTest, SiameseTrain
from pointnet2.models import Pointnet_Tracking, Pointnet_Tracking3 

def test(loader,
        model,
        epoch=-1,
        shape_aggregation=" ",
        reference_BB=" ",
        max_iter=-1,
        IoU_Space=3):
    # switch to evaluate mode
    model.eval()

    dataset = loader.dataset
    batch_num = 0

    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno), position=0, desc='testing datas') as t:
        for batch in loader:          
            batch_num = batch_num + 1
            for PCs, BBs, _ in batch:
                results_BBs = []  # 保存预测边界框列表
                candidate_PCs_torchs = []
                # estimation_boxs = []
                # acboxs = []
                # gtboxs = []
                # if batch_num == 59:
                #     segs = []
                #     cs = []
                for i, _ in enumerate(PCs):
                    # this_anno = list_of_anno[i]
                    pboxes = []
                    this_BB = BBs[i]  # 当前帧真值边界框
                    this_PC = PCs[i]  # 当前帧点云
                    # INITIAL FRAME
                    if i == 0:
                        box = BBs[i]
                        results_BBs.append(box)
                        model_PC = utils.getModel([this_PC], [this_BB], offset=dataset.offset_BB, scale=dataset.scale_BB)

                    else:
                        previous_BB = BBs[i - 1]  # 前一帧真值边界框

                        # DEFINE REFERENCE BB
                        if ("previous_result".upper() in reference_BB.upper()):
                            ref_BB = results_BBs[-1]  # 参考框选为前一帧预测边界框
                        elif ("previous_gt".upper() in reference_BB.upper()):
                            ref_BB = previous_BB  # 参考框选为前一帧真值边界框
                            # ref_BB = utils.getOffsetBB(this_BB,np.array([-1,1,1]))
                        elif ("current_gt".upper() in reference_BB.upper()):
                            ref_BB = this_BB  # 参考框选为当前帧真值边界框

                        candidate_PC, candidate_label, candidate_reg, _, _ = utils.cropAndCenterPC_label_test(
                                        this_PC,
                                        ref_BB,
                                        this_BB,
                                        offset=dataset.offset_BB,
                                        scale=dataset.scale_BB)
                        
                        candidate_PCs, _, candidate_reg, candidate_seg, _ = utils.regularizePCwithlabel(
                                        candidate_PC,
                                        candidate_label,
                                        candidate_reg,
                                        dataset.input_size,
                                        istrain=False)
                        
                        candidate_PCs_torch = candidate_PCs.unsqueeze(0).cuda()

                        # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                        if ("firstandprevious".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0], PCs[i-1]], [results_BBs[0],results_BBs[i-1]], offset=dataset.offset_BB, scale=dataset.scale_BB)
                        elif ("first".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0]], [results_BBs[0]], offset=dataset.offset_BB, scale=dataset.scale_BB)
                        elif ("previous".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[i-1]], [results_BBs[i-1]], offset=dataset.offset_BB, scale=dataset.scale_BB)
                        elif ("all".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB, scale=dataset.scale_BB)
                        else:
                            model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB, scale=dataset.scale_BB)

                        model_PC_torch = utils.regularizePC(model_PC, dataset.input_size, istrain=False).unsqueeze(0).cuda()
                        model_PC_torch = model_PC_torch.requires_grad_(False)
                        candidate_PCs_torch.requires_grad_(False)
                        candidate_PCs_torchs.append(candidate_PCs_torch)
                        _, _, estimation_box, _, = model(model_PC_torch, candidate_PCs_torch)
                        estimation_boxs_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                        estimation_pboxs = estimation_boxs_cpu[:, 0: 4]
                        rot = Quaternion(matrix=this_BB.rotation_matrix)
                        trans = np.array(this_BB.center)
                        for i in range(len(estimation_pboxs)):
                            pbox = utils.getOffsetBB(ref_BB, estimation_pboxs[i])
                            pbox.translate(-trans)
                            pbox.rotate(rot.inverse)                            
                            pboxes.append(pbox)
                        box_idx = estimation_boxs_cpu[:, 4].argmax()  # 根据proposal-wise targetness得分确定对应的边界框索引
                        estimation_box_cpu = estimation_boxs_cpu[box_idx, 0: 4]
                        
                        box = utils.getOffsetBB(ref_BB, estimation_box_cpu)
                        # rot = Quaternion(matrix=this_BB.rotation_matrix)
                        # trans = np.array(this_BB.center)
                        # gtbox = copy.deepcopy(this_BB)
                        # gtbox.translate(-trans)
                        # gtbox.rotate(rot.inverse)
                        # acbox = utils.getOffsetBB2(ref_BB, estimation_box_cpu)
                        results_BBs.append(box)
                        # acbox = copy.deepcopy(box)
                        # acbox.center = estimation_box_cpu[0:3]
                        # estimation_boxs.append(estimation_box_cpu)
                        # acboxs.append(acbox)
                        # gtboxs.append(gtbox)
                        # if (batch_num == 59) and (i < 25):
                        #     cs.append(candidate_seg)
                        #     segs.append(estimation_seg)
                    t.update(1)  # 推动进度条
                if batch_num == 1:
                    # return candidate_PCs_torchs, gtboxs, acboxs
                    return candidate_PCs_torchs, pboxes
            # else:
            #     t.update(1)  # 推动进度条
            #     continue

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 0'

# args.manualSeed = 0
# random.seed(args.manualSeed)
# torch.manual_seed(args.manualSeed)
ngpu = 1
# model = 'netR_32_Car.pth'
netR = Pointnet_Tracking(input_channels=0, use_xyz=True, test=True).cuda()
# netR = Pointnet_Tracking3(input_channels=0, use_xyz=True, test=True).cuda()
if ngpu > 1:
    netR = torch.nn.DataParallel(netR, range(ngpu))
    # torch.distributed.init_process_group(backend="nccl")
    # netR = torch.nn.DistributedDataParallel(netR)   
netR.load_state_dict(torch.load('model/car_model/netR_32_Car.pth'))   
# netR.cuda()
torch.cuda.synchronize()
# Car/Pedestrian/Van/Cyclist
dataset_Test = SiameseTest(
    input_size=1024,
    path='/mnt/ssd-data/RUNNING/data/trianing',
    split='Test',
    category_name='Car',
    offset_BB=0,
    scale_BB=1.25)

test_loader = torch.utils.data.DataLoader(
    dataset_Test,
    collate_fn=lambda x: x,
    batch_size=1,
    shuffle=False,
    num_workers=0,
        pin_memory=True)

if dataset_Test.isTiny():
    max_epoch = 2
else:
    max_epoch = 1

for epoch in range(max_epoch):
    # candidate_PCs_torchs, gtboxs, acboxs = test(
    candidate_PCs_torchs, pboxes = test(
        test_loader,
        netR,
        epoch=epoch + 1,
        shape_aggregation='firstandprevious',
        reference_BB='previous_result',
        # model_fusion=args.model_fusion,
        IoU_Space=3)