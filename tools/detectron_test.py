# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from models.pose_hrnet import get_pose_net
from dataset.coco_realtime import COCODataset
from utils.vis import save_batch_heatmaps
import cv2
import glob
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='models/pytorch/pose_coco')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def detectron_validate(config, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        outputs = model(val_dataset.cuda())


def detectron_save_image(crop, model, criterion, final_output_dir):
    model.eval()
    with torch.no_grad():
        outputs = model(crop.cuda())
        grid_img = torchvision.utils.make_grid(crop, padding=0)
        prefix = '{}_{:05d}'.format(
            os.path.join(final_output_dir, 'val'), criterion
        )
        torchvision.utils.save_image(grid_img, prefix + '_im.jpg', normalize=True)
        save_batch_heatmaps(crop, outputs, prefix + '_heat.jpg')


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Create config
    detection_cfg = get_cfg()
    detection_cfg.DATASETS.TRAIN = (os.getcwd() + "/data/coco/images/train2017",)
    detection_cfg.DATASETS.TEST = (os.getcwd() + "../data/coco/images/val2017",)
    detection_cfg.merge_from_file("../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    detection_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    detection_cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

    # Create predictor
    predictor = DefaultPredictor(detection_cfg)
    # Create detector
    model = get_pose_net(cfg, is_train=False)
    '''
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    '''
    # print(model)
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    test_list = glob.glob("{}/{}/*".format(os.getcwd(), '/data/coco/images/val2017'))
    tic = time.time()
    total_image = len(test_list)
    total_person = 0
    detect_time = 0
    estimate_time = 0
    for i in range(len(test_list)):
        inputs = cv2.imread(test_list[i])
        det_start = time.time()
        outputs = predictor(inputs)
        detect_time = detect_time + time.time() - det_start
        human_boxes = outputs['instances'].pred_boxes[outputs['instances'].pred_classes == 0]
        # human_boxes = [i for i in human_boxes if abs(int(boxes[i, 1])-int(boxes[i, 3])) * abs(int(boxes[i, 0])-int(boxes[i, 2])) >= 32*32]
        boxes = human_boxes.tensor
        total_person = total_person + boxes.shape[0]
        if boxes.shape[0] > 0:
            for j in range(boxes.shape[0]):
                cropped_img = cv2.resize(inputs[int(boxes[j, 1]): int(boxes[j, 3]),
                                         int(boxes[j, 0]): int(boxes[j, 2])], dsize=(192, 256))
                if j is 0:
                    crop = torch.unsqueeze(torch.from_numpy(cropped_img), 0)
                else:
                    crop = torch.cat((crop, torch.unsqueeze(torch.from_numpy(cropped_img), 0)), 0)

            crop = torch.transpose(torch.transpose(crop, -1, -2), -2, -3).float()  # NCHW
            crop = ((crop/255.) - torch.tensor([[[[0.485]],[[0.456]],[[0.406]]]]))/torch.tensor([[[[0.229]],[[0.224]],[[0.225]]]])
            est_start = time.time()
            detectron_validate(cfg, crop, model, criterion,
                     final_output_dir, tb_log_dir)
            estimate_time = estimate_time + time.time() - est_start
            detectron_save_image(crop, model, i, final_output_dir)
        else:
            total_image -= 1

    total_time = time.time()-tic

    print('-[only detection]-')
    print('[*] Total elapsed time: {}'.format(detect_time))
    print('[*] image per second: {}'.format(total_image / detect_time))
    print('[*] person per second: {}'.format(total_person / detect_time))
    print('--[only estimation]-')
    print('[*] Total elapsed time: {}'.format(estimate_time))
    print('[*] image per second: {}'.format(total_image / estimate_time))
    print('[*] person per second: {}'.format(total_person / estimate_time))
    print('--[detection+estimation]-')
    print('[*] Total elapsed time: {}'.format(total_time))
    print('[*] image per second: {}'.format(total_image/total_time))
    print('[*] person per second: {}'.format(total_person / total_time))


if __name__ == '__main__':
    main()
