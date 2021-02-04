import time
import os
import logging
import argparse
import random
import copy 
# import numpy as np
from tqdm import tqdm

import torch

import kitty_utils as utils
from Dataset import SiameseTest
from pointnet2.models import Pointnet_Tracking, Pointnet_Tracking3 

def test(loader,
        model,
        epoch=-1,
        shape_aggregation=" ",
        reference_BB=" ",
        # model_fusion="pointcloud",
        max_iter=-1,
        IoU_Space=3):
    # switch to evaluate mode
    model.eval()

    dataset = loader.dataset
    batch_num = 0

    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno), position=0, desc='testing datas') as t:
        for batch in loader:          
            batch_num = batch_num + 1
            # measure data loading time
            # for PCs, BBs, list_of_anno in batch:
            for PCs, BBs, _ in batch:
                results_BBs = []  # 保存预测边界框列表
                estimation_box = []
                for i, _ in enumerate(PCs):
                    # this_anno = list_of_anno[i]
                    this_BB = BBs[i]  # 当前帧真值边界框
                    this_PC = PCs[i]  # 当前帧点云
                    # gt_boxs = []
                    # result_boxs = []

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
                        
                        candidate_PCs, _, candidate_reg, _, _ = utils.regularizePCwithlabel(
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
                        # model_PC_torch = Variable(model_PC_torch, requires_grad=False).cuda()
                        # candidate_PCs_torch = Variable(candidate_PCs_torch, requires_grad=False).cuda()

                        _, _, estimation_box, _ = model(model_PC_torch, candidate_PCs_torch)
                        estimation_boxs_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                        box_idx = estimation_boxs_cpu[:, 4].argmax()  # 根据proposal-wise targetness得分确定对应的边界框索引
                        estimation_box_cpu = estimation_boxs_cpu[box_idx, 0: 4]
                        
                        box = utils.getOffsetBB(ref_BB, estimation_box_cpu)
                        results_BBs.append(box)
                        # acbox = copy.deepcopy(box)
                        # acbox.center = estimation_box_cpu[0:3]
                        # estimation_box.append(estimation_box_cpu)
                        # acboxs.append(acbox)



                    t.update(1)  # 推动进度条
            if batch_num == 112:
                return results_BBs[-1], candidate_PCs_torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--save_root_dir', type=str, default='model/car_model',  help='output folder')
    parser.add_argument('--data_dir', type=str, default = '/media/zhouxiaoyu/本地磁盘/RUNNING/data/trianing',  help='dataset path')
    parser.add_argument('--model', type=str, default = 'netR_32_Car.pth',  help='model name for training resume')
    parser.add_argument('--category_name', type=str, default = 'Car',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
    parser.add_argument('--shape_aggregation',required=False,type=str,default="firstandprevious",help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB',required=False,type=str,default="previous_result",help='previous_result/previous_gt/current_gt')
    parser.add_argument('--model_fusion',required=False,type=str,default="pointcloud",help='early or late fusion (pointcloud/latent/space)')
    parser.add_argument('--IoU_Space',required=False,type=int,default=3,help='IoUBEV vs IoUBox (2 vs 3)')
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args.manualSeed = 0
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    netR = Pointnet_Tracking(input_channels=0, use_xyz=True, test=True).cuda()
    # netR = Pointnet_Tracking3(input_channels=0, use_xyz=True, test=True).cuda()
    if args.ngpu > 1:
        netR = torch.nn.DataParallel(netR, range(args.ngpu))
        # torch.distributed.init_process_group(backend="nccl")
        # netR = torch.nn.DistributedDataParallel(netR) 
    if args.model != '':
        netR.load_state_dict(torch.load(os.path.join(args.save_root_dir, args.model)))    
    # netR.cuda()
    print(netR)
    torch.cuda.synchronize()
    # Car/Pedestrian/Van/Cyclist
    dataset_Test = SiameseTest(
            input_size=1024,
            path=args.data_dir,
            split='Test',
            category_name=args.category_name,
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
        Succ, Prec = test(
            test_loader,
            netR,
            epoch=epoch + 1,
            shape_aggregation=args.shape_aggregation,
            reference_BB=args.reference_BB,
            # model_fusion=args.model_fusion,
            IoU_Space=args.IoU_Space)
