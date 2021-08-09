from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from collections import namedtuple
from utils import colorEncode

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as Metrics
from sklearn.mixture import GaussianMixture
from statsmodels.distributions.empirical_distribution import ECDF
import joblib
import json
from sklearn import manifold

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([255, 255, 255])
colors = np.array(train_id_to_color)
colors = np.uint8(colors)

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_embedding_resnet101'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

def Normalization(x):
    min_value = np.min(x)
    max_value = np.max(x)
    return (x - min_value) / (max_value - min_value)

def Certainty(x, ecdf, thre1, thre2, mean, cov):
    x = ecdf(x)
    # res = x
    # res[res>0.2] = 1
    threshold = ecdf(thre1)
    coefficient = 50
    res = 1 / (1 + np.exp(-coefficient * (x - threshold)))

    return res

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

def Coefficient_map(x, thre):
    lamda = 20
    return 1 / (1 + np.exp(lamda * (x - thre)))

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""


    metrics.reset()
    ret_samples = []
    AUC_scores = []
    AUPR_scores = []
    FPR95_scores = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    # bad = [482, 459, 463, 356, 473]
    # bad = [466,356,477,464,482,465,454,459,461,446,483,472,252,442,463,443,320,221]
    # prototype = [[] for _ in range(19)]
    # prototype = []
    with open('prototype_car_5_shot.json', 'r') as file_object:
        prototype = json.load(file_object)
        print(len(prototype))
    # with open('prototype_car_1_shot.json', 'r') as file_object:
    #     prototype_car = json.load(file_object)
    #     print(len(prototype_car))

    # # # for index_i in range(5):
    # # #     for index_j in range(index_i + 1, 5):
    # # #         print(np.sum((np.array(prototype[index_j]) - np.array(prototype[index_i])) ** 2))
    prototype_car = np.zeros((16,))
    for i in range(5):
        prototype_car += np.array(prototype[i])
    prototype_car /= 5
    # print(prototype_car)

    # prototype_truck = np.zeros((16,))
    # with open('prototype_truck_5_shot.json', 'r') as file_object:
    #     prototype = json.load(file_object)
    #     print(len(prototype))
    # with open('prototype_truck_1_shot.json', 'r') as file_object:
    #     prototype_truck = json.load(file_object)
    #     print(len(prototype_truck))
    # for i in range(5):
    #     prototype_truck += np.array(prototype[i])
    # prototype_truck /= 5
    # print(prototype_truck)
    #
    # prototype_bus = np.zeros((16,))
    # with open('prototype_bus_5_shot.json', 'r') as file_object:
    #     prototype = json.load(file_object)
    #     print(len(prototype))
    # with open('prototype_bus_1_shot.json', 'r') as file_object:
    #     prototype_bus = json.load(file_object)
    #     print(len(prototype_bus))
    # for i in range(5):
    #     prototype_bus += np.array(prototype[i])
    # prototype_bus /= 5
    # print(prototype_bus)
    #
    # with open('prototype_ood.json', 'r') as file_object:
    #     prototype = json.load(file_object)

    # prototype_car = np.zeros((16,))

    # plot tsne
    # tsne = manifold.TSNE(n_components=2)
    # data_total = np.zeros((16,16))
    # label_total = np.zeros((16,1))
    # magnitude = 3
    # for i in range(16):
    #     data_total[i][i] = magnitude
    #     if i <= 12:
    #         label_total[i] = i
    #     else:
    #         label_total[i] = i + 3
    # for i in range(19):
    #     data_temp = np.array(prototype[i])
    #     data_total = np.vstack((data_total, data_temp))
    #     label_temp = i * np.ones((data_temp.shape[0],1))
    #     label_total = np.vstack((label_total, label_temp))
    # # print(data_total.shape, label_total.shape)
    # # print(label_total.squeeze())
    # X_tsne = tsne.fit_transform(data_total)
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # # plt.figure(figsize=(8, 8))
    # # print(plt.cm.tab20(int(label_total.squeeze()[500])))
    # # X_norm = X_tsne
    # plt.figure()
    # for i in range(X_norm.shape[0]):
    #     # plt.text(X_norm[i, 0], X_norm[i, 1], str(int(label_total.squeeze()[i])), color=plt.cm.tab20(int(label_total.squeeze()[i])),
    #     #          fontdict={'weight': 'bold', 'size': 9})
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], c=label_total.squeeze()[i], cmap=plt.cm.tab20)
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend()
    # plt.axis('off')
    # plt.savefig('imgs/tSNE_.jpg', bbox_inches='tight', dpi=600, pad_inches = 0)
    # plt.show()



    with torch.no_grad():
        for i, (images, labels, labels_true) in tqdm(enumerate(loader)):
        # for i, (images) in tqdm(enumerate(loader)):
            # if i not in bad:
            #     continue
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, centers, features = model(images)
            # print(outputs.shape)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            soft_outs = F.softmax(outputs, dim=1)
            scores_auc = 1 - soft_outs.detach().max(dim=1)[0].cpu().numpy()
            scores_auc_softmax = scores_auc.squeeze()
            # plt.figure()
            # plt.imshow(scores_auc_softmax)
            # # plt.colorbar()
            # plt.show()

            #
            dis_sum_map = -np.sum(outputs.squeeze().cpu().numpy(), axis=0)
            dis_sum_map[dis_sum_map > 1000] = 1000
            #
            # labels_true[labels_true == 255] = 0
            # plt.figure()
            # plt.imshow(labels_true.squeeze().cpu().numpy())
            # plt.show()
            # #
            # plt.figure()
            # plt.imshow(-dis_sum_map)
            # # plt.colorbar()
            # plt.show()
            # prob_map_origin = soft_outs.squeeze().cpu().numpy()
            # scores_map = np.zeros_like(dis_sum_map)
            # for cl in range(opts.num_classes):
            #     scores_map += prob_map_origin[cl] * Certainty(dis_sum_map,ecdf_list[cl], thre1_list[cl], thre2_list[cl], mean_list[cl], cov_list[cl])
            dis_sum_map_norm = Normalization(dis_sum_map)
            # Coefficient = Coefficient_map(dis_sum_map_norm, 0.3)
            # prob_map = outputs.squeeze().detach().max(dim=0)[0].cpu().numpy()
            # prob_map = Normalization(prob_map)
            # conf = Coefficient * dis_sum_map_norm + (1 - Coefficient) * prob_map
            scores_auc_dis = 1 - dis_sum_map_norm
            # scores_auc_dis[scores_auc_dis < 0.6] = 0
            # scores_auc_dis[scores_auc_dis > 0.6] = 1
            preds_base = preds[0]
            # print(np.unique(preds_base))
            # preds_base[preds_base == 13] = -3
            # preds_base[preds_base == 14] = -2
            # preds_base[preds_base == 15] = -1
            # preds_base[preds_base == 16] = -3
            # preds_base[preds_base >= 16] -= 3
            # preds_base[preds_base == -3] = 16
            # preds_base[preds_base == -2] = 17
            # preds_base[preds_base == -1] = 18
            # preds_base[preds_base == -3] = 16
            # print(np.unique(preds_base))
            # preds_base[scores_auc_dis > 0.8] = 19
            # labels[labels==255]=19
            # plt.imshow(colorEncode(labels[0].cpu().numpy(), colors))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.savefig('close-set-all/preds_base_'+str(i)+'.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()
            # plt.close()
            # # preds_base[scores_auc_dis > 0.5] = 0
            # # scores_auc = 1 - scores_map
            # plt.figure()
            # plt.imshow(colorEncode(preds_base,colors))
            # # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.savefig('imgs/labels_uncertainty.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()
            # plt.figure()
            # plt.imshow(conf)
            # # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.savefig('imgs/labels_uncertainty.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()

            # intances, counts = np.unique(labels_true, False, False, True)
            # if 15 in intances:
            #     if (counts[np.where(intances == 15)] / np.sum(counts) > 0.05):
            #         features_ood = features.cpu().numpy()[labels_true == 15]
            #         features_ood_mean = np.mean(features_ood, axis=0)
            #         # print(features_ood_mean)
            #         prototype.append(features_ood_mean.tolist())
            #         print(len(prototype))
            # if len(prototype) == 1:
            #     file_name = 'prototype_bus_1_shot.json'
            #     with open(file_name, 'w') as file_object:
            #         json.dump(prototype, file_object)
            #     print('done')

            # only car is incremental class
            b, h, w, c = features.shape
            features = features.view(b, h*w, c).squeeze().cpu().numpy()
            dis_ood_car = -np.sum((features - prototype_car) ** 2, axis=1)
            # dis_ood_truck = -np.sum((features - prototype_truck) ** 2, axis=1)
            # # print(dis_ood.shape)
            dis_ood_car = dis_ood_car.reshape(h,w)
            # dis_ood_truck = dis_ood_truck.reshape(h,w)
            # # #
            # # plt.figure()
            # # plt.imshow(soft_outs.detach().max(dim=1)[0].squeeze().cpu().numpy())
            # # plt.show()
            # #
            # # plt.figure()
            # # ins, cnt = np.unique(preds[0], False, False, True)
            # # print(ins, cnt)
            # # plt.imshow(preds[0])
            # # plt.show()
            preds[0][np.logical_and(dis_ood_car > -1.5, dis_ood_car > outputs.detach().max(dim=1)[0].squeeze().cpu().numpy())] = 16
            # preds[0][dis_ood_car > -2] = 16
            # # preds[0][dis_ood > outputs.detach().max(dim=1)[0].squeeze().cpu().numpy()] = 16
            labels[labels == 13] = -1
            labels[labels >=14] -=1
            labels[labels == -1] = 16
            labels[labels == 254] = 255
            # plt.imshow(preds[0])
            # plt.show()
            # # plt.figure()
            # # plt.imshow(labels.squeeze().cpu().numpy())
            # # plt.show()
            # # print(torch.unique(labels))
            # #

            # car, truck both novel classes
            # b, h, w, c = features.shape
            # features = features.view(b, h * w, c).squeeze().cpu().numpy()
            # dis_ood_car = -np.sum((features - prototype_car) ** 2, axis=1)
            # dis_ood_truck = -np.sum((features - prototype_truck) ** 2, axis=1)
            # dis_ood_bus = -np.sum((features - prototype_bus) ** 2, axis=1)
            # dis_ood_car = dis_ood_car.reshape(h, w)
            # dis_ood_truck = dis_ood_truck.reshape(h, w)
            # dis_ood_bus = dis_ood_bus.reshape(h, w)
            # #
            #
            # plt.figure()
            # ins, cnt = np.unique(preds[0], False, False, True)
            # print(ins, cnt)
            # plt.imshow(preds[0])
            # plt.show()

            # one novel class only
            # preds[0][np.logical_and(dis_ood_car > -1.5, dis_ood_car > outputs.detach().max(dim=1)[0].squeeze().cpu().numpy())] = 16
            # preds[0][dis_ood_car > outputs.detach().max(dim=1)[0].squeeze().cpu().numpy()] = 16
            # labels[labels == 13] = -1
            # labels[labels >= 14] -= 1
            # labels[labels == -1] = 16
            # labels[labels == 254] = 255
            # labels[labels == 255] = 19
            # plt.imshow(colorEncode(labels.squeeze().cpu().numpy(), colors))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.savefig('imgs/preds_car_1_shot.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()
            #
            # preds_base = preds[0]
            # print(np.unique(preds_base))
            # preds_base[preds_base == 13] = -3
            # preds_base[preds_base == 14] = -2
            # preds_base[preds_base == 15] = -1
            # preds_base[preds_base >= 16] -= 3
            # preds_base[preds_base == -3] = 16
            # preds_base[preds_base == -2] = 17
            # preds_base[preds_base == -1] = 18
            # print(np.unique(preds_base))
            # plt.imshow(colorEncode(preds_base, colors))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.savefig('imgs/preds_car_1_shot.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()

            # two novel classes
            # preds[0][np.logical_and(dis_ood_car > -1.5, dis_ood_car > dis_ood_truck)] = 16
            # preds[0][np.logical_and(dis_ood_truck > -1.5, dis_ood_truck > dis_ood_car)] = 17
            # labels[labels == 13] = -2
            # labels[labels == 14] = -1
            # labels[labels>=15] -= 2
            # labels[labels == -2] = 16
            # labels[labels == -1] = 17
            # labels[labels == 253] = 255

            # three novel classes
            # preds[0][np.logical_and(dis_ood_car > -1.5, np.logical_and(dis_ood_car > dis_ood_truck, dis_ood_car > dis_ood_bus))] = 16
            # preds[0][np.logical_and(dis_ood_truck > -1.5, np.logical_and(dis_ood_truck > dis_ood_car, dis_ood_truck > dis_ood_bus))] = 17
            # preds[0][np.logical_and(dis_ood_bus > -1.5, np.logical_and(dis_ood_bus > dis_ood_truck, dis_ood_bus > dis_ood_car))] = 18
            # # labels[labels == 13] = -3
            # # labels[labels == 14] = -2
            # # labels[labels == 15] = -1
            # # labels[labels >= 16] -= 3
            # # labels[labels == -3] = 16
            # # labels[labels == -2] = 17
            # # labels[labels == -1] = 18
            # # labels[labels == 252] = 255
            # preds_base = preds[0]
            # preds_base[preds_base == 13] = -3
            # preds_base[preds_base == 14] = -2
            # preds_base[preds_base == 15] = -1
            # preds_base[preds_base >= 16] -= 3
            # preds_base[preds_base == -3] = 16
            # preds_base[preds_base == -2] = 17
            # preds_base[preds_base == -1] = 18
            # plt.imshow(colorEncode(preds_base,colors))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.savefig('imgs/preds_car_5_shot.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()
            # plt.figure()
            # plt.imshow(labels.squeeze().cpu().numpy())
            # plt.show()
            # print(torch.unique(labels))
            #






            # plt.figure()
            # plt.imshow(scores_auc)
            # plt.show()

            targets = labels.cpu().numpy()

            # print(targets.shape)
            # print(preds_base.shape)
            metrics.update(targets, preds)

            # scores_auc = scores_auc_dis[labels_true.squeeze() != 255]
            # msk_auc = labels[labels_true != 255]
            # msk_auc[msk_auc != 255] = 0
            # msk_auc[msk_auc == 255] = 1

            # msk_auc = labels.clone()
            # msk_auc[msk_auc != 255] = 0
            # msk_auc[msk_auc == 255] = 1

            # msk_auc = msk_auc.cpu().numpy().ravel()
            # scores_auc = scores_auc.ravel()
            # scores_auc = scores.ravel()
            # instance, counts = np.unique(msk_auc, False, False, True)
            # print(instance, counts)
            # if 1 in instance:
            #
            #     # counts_min = np.min(counts)
            #     # counts_min_index = np.argmin(counts)
            #     #
            #     # scores_auc_min = scores_auc[msk_auc == counts_min_index]
            #     # msk_auc_min = msk_auc[msk_auc == counts_min_index]
            #     # # print(scores_auc_min.shape, msk_auc_min.shape)
            #     #
            #     #
            #     # scores_auc_max = scores_auc[msk_auc != counts_min_index]
            #     # msk_auc_max = msk_auc[msk_auc != counts_min_index]
            #     # # print(scores_auc_max.shape, msk_auc_max.shape)
            #     #
            #     #
            #     # counts_max_index = np.random.choice(len(scores_auc_max), counts_min)
            #     # scores_auc_max = scores_auc_max[counts_max_index]
            #     # msk_auc_max = msk_auc_max[counts_max_index]
            #     # # print(scores_auc_max.shape, msk_auc_max.shape)
            #     #
            #     # scores_auc = np.hstack((scores_auc_min, scores_auc_max))
            #     # msk_auc = np.hstack((msk_auc_min, msk_auc_max))
            #     # print(scores_auc.shape)
            #     # print(msk_auc.shape)
            #
            #
            #     auc = Metrics.roc_auc_score(msk_auc, scores_auc)
            #     fpr, tpr, ths = Metrics.roc_curve(msk_auc, scores_auc)
            #     aupr = Metrics.average_precision_score(msk_auc, scores_auc)
            #     fpr95 = fpr[tpr >= 0.95][0]
            #     # print(auc)
            #     AUC_scores.append(auc)
            #     AUPR_scores.append(aupr)
            #     FPR95_scores.append(fpr95)

            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    scores = (255 * scores).squeeze().astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    Image.fromarray(scores).save('results/%d_scores.png' % img_id)

                    # np.save('results/%d_dis_sum.npy' % img_id, dis_sum_map)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
        print(np.mean(np.array(AUC_scores)))
        print(np.mean(np.array(FPR95_scores)))
        print(np.mean(np.array(AUPR_scores)))

    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 16

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=16)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=16)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_embedding_resnet101': network.deeplabv3plus_embedding_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = utils.CrossEntropyLoss(ignore_index=255, alpha=0.01, beta=0.01/80, gamma=0)


    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints_131415_embedding')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        opts.gpu_id = [1]
        # model = nn.DataParallel(model,device_ids=opts.gpu_id)
        model = nn.DataParallel(model)
        model = model.cuda()

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, labels_true) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs, centers, features = model(images)
            loss = criterion(outputs, labels, features)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints_131415_embedding/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints_131415_embedding/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset,opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()  

            if cur_itrs >=  opts.total_itrs:
                return

        
if __name__ == '__main__':
    main()
