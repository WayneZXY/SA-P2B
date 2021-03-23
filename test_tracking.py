import time
import os
import logging
import argparse
import random

# import numpy as np
from tqdm import tqdm

import torch

import kitty_utils as utils
# import copy
from datetime import datetime


from metrics import AverageMeter, Success, Precision
from metrics import estimateOverlap, estimateAccuracy
# from data_classes import PointCloud
from Dataset import SiameseTest

# import torch.nn.functional as F
# from torch.autograd import Variable

from pointnet2.models import Pointnet_Tracking, Pointnet_Tracking3 

def test(loader,
        model,
        epoch=-1,
        shape_aggregation=" ",
        reference_BB=" ",
        # model_fusion="pointcloud",
        max_iter=-1,
        IoU_Space=3):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()
    Success_batch = Success()
    Precision_batch = Precision()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    dataset = loader.dataset
    batch_num = 0

    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno), position=0, desc='testing datas') as t:
        for batch in loader:          
            batch_num = batch_num + 1
            # measure data loading time
            data_time.update((time.time() - end))
            # for PCs, BBs, list_of_anno in batch:
            for PCs, BBs, _ in batch:
                results_BBs = []  # 保存预测边界框列表

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

                        # candidate_PC, candidate_label, candidate_reg, _, _ = utils.cropAndCenterPC_label_test(
                        #                 this_PC,
                        #                 ref_BB,
                        #                 this_BB,
                        #                 offset=dataset.offset_BB,
                        #                 scale=dataset.scale_BB)

                        candidate_PC = utils.cropAndCenterPC_label_test_time(
                                        this_PC,
                                        ref_BB,
                                        offset=dataset.offset_BB,
                                        scale=dataset.scale_BB)                        
                        
                        # candidate_PCs, _, candidate_reg, _, _ = utils.regularizePCwithlabel(
                        #                 candidate_PC,
                        #                 candidate_label,
                        #                 candidate_reg,
                        #                 dataset.input_size,
                        #                 istrain=False)

                        candidate_PCs = utils.regularizePC_test(
                                        candidate_PC,
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

                    # estimate overlap/accuracy from current sample
                    this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
                    this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)

                    Success_main.add_overlap(this_overlap)
                    Precision_main.add_accuracy(this_accuracy)
                    Success_batch.add_overlap(this_overlap)
                    Precision_batch.add_accuracy(this_accuracy)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    t.update(1)  # 推动进度条

                    if Success_main.count >= max_iter and max_iter >= 0:  # 到一定轮数，因为max_iter=-1，这段并不起作用
                        return Success_main.average, Precision_main.average

                t.set_description('Test {}: '.format(epoch)+
                                    'Time {:.3f}s '.format(batch_time.avg)+
                                    '(it:{:.3f}s) '.format(batch_time.val)+
                                    'Data:{:.3f}s '.format(data_time.avg)+
                                    '(it:{:.3f}s), '.format(data_time.val)+
                                    'Succ/Prec:'+
                                    '{:.1f}/'.format(Success_main.average)+
                                    '{:.1f}'.format(Precision_main.average))
                logging.info('batch {}'.format(batch_num)+
                                    'Succ/Prec:'+
                                    '{:.1f}/'.format(Success_batch.average)+
                                    '{:.1f}'.format(Precision_batch.average))
                Success_batch.reset()
                Precision_batch.reset()

    return Success_main.average, Precision_main.average

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=2, help='# GPUs')
    parser.add_argument('--save_root_dir', type=str, default='model/Van_model',  help='output folder')
    parser.add_argument('--data_dir', type=str, default = '/mnt/ssd-data/RUNNING/data/training',  help='dataset path')
    parser.add_argument('--model', type=str, default = 'netR_35.pth',  help='model name for training resume')
    parser.add_argument('--category_name', type=str, default = 'Van',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
    parser.add_argument('--shape_aggregation',required=False,type=str,default="firstandprevious",help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB',required=False,type=str,default="previous_result",help='previous_result/previous_gt/current_gt')
    parser.add_argument('--model_fusion',required=False,type=str,default="pointcloud",help='early or late fusion (pointcloud/latent/space)')
    parser.add_argument('--IoU_Space',required=False,type=int,default=3,help='IoUBEV vs IoUBox (2 vs 3)')
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 0'

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(args.save_root_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log')), level=logging.INFO)
    logging.info('======================================================')

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

    Success_run = AverageMeter()
    Precision_run = AverageMeter()

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

        Success_run.update(Succ)
        Precision_run.update(Prec)
        logging.info("mean Succ/Prec {}/{}".format(Success_run.avg, Precision_run.avg))
