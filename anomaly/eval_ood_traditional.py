# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset import ValDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from scipy import stats
import random
from sklearn.mixture import GaussianMixture
import joblib
from statsmodels.distributions.empirical_distribution import ECDF
import anom_utils
import cv2
import seaborn as sns
from collections import namedtuple

# colors = loadmat('data/color150.mat')['colors']

# colors for streerhazards
colors = np.array([[ 0,   191,   255],# // unlabeled     =   0,
       [ 70,  70,  70],# // building      =   1,
       [190, 153, 153],# // fence         =   2,
       [250, 170, 160],# // other         =   3,
       [220,  20,  60],# // pedestrian    =   4,
       [153, 153, 153],# // pole          =   5,
       [157, 234,  50],# // road line     =   6,
       [128,  64, 128],# // road          =   7,
       [244,  35, 232],# // sidewalk      =   8,
       [107, 142,  35],# // vegetation    =   9,
       [  0,   0, 142],# // car           =  10,
       [102, 102, 156],# // wall          =  11,
       [220, 220,   0],# // traffic sign  =  12,
       [ 60, 250, 240],# // anomaly       =  13,

       ])

# colors for bdd100k
# CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
#                                                      'has_instances', 'ignore_in_eval', 'color'])
# classes = [
#     CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
#     CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
#     CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
#     CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
#     CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
#     CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
#     CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
#     CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
#     CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
#     CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
#     CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
#     CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
#     CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
#     CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
#     CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
#     CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
#     CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
#     CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
#     CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
#     CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
#     CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
#     CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
#     CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
#     CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
#     CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
#     CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
#     CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
#     CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
#     CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
#     CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
#     CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
#     CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
#     CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
#     CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
#     CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
# ]
#
# train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
# colors = np.array(train_id_to_color)

colors = np.uint8(colors)
print(colors.dtype)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Normalizatoin(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def Coefficient_map(x, thre):
    lamda = 50
    return 1 / (1 + np.exp(lamda * (x - thre)))

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))

def eval_ood_measure(conf, seg_label, cfg, mask=None):
    out_labels = cfg.OOD.out_labels
    if mask is not None:
        seg_label = seg_label[mask]

    # conf = conf[seg_label != -1]
    # seg_label = seg_label[seg_label != -1]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return None


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    intersection_meter_unknown = [AverageMeter() for _ in range(9)]
    union_meter_unknown = [AverageMeter() for _ in range(9)]
    time_meter = AverageMeter()

    segmentation_module.eval()

    aurocs, auprs, fprs = [], [], []

    pbar = tqdm(total=len(loader))

    # ecdf_list = joblib.load('ecdf_list_embedding_00.pkl')
    # with open('logit_dict.json', 'r', encoding='utf8')as fp:
    #     json_data = json.load(fp)


    # x_lin = np.linspace(0,1000,1000)
    # for i in range(13):
    #     if i != 4:
    #         y_lin = ecdf_list[i][0](x_lin)
    #         plt.figure()
    #         plt.plot(x_lin, y_lin)
    #         plt.show()

    cnt = 0
    for batch_data in loader:
        cnt += 1
        # if cnt != 2:
        #     continue
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)
            ft1 = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            # ft1 = torch.zeros(1, 4096, int(segSize[0] / 4), int(segSize[1] / 4))
            ft1 = async_copy_to(ft1, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                del feed_dict['name']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp, ft_temp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                ft_temp = nn.functional.interpolate(ft_temp, size=ft1.shape[2:], mode='bilinear', align_corners=False)
                ft1 = ft1 + ft_temp / len(cfg.DATASET.imgSizes)

            tmp_scores = scores
            if cfg.OOD.exclude_back:
                tmp_scores = tmp_scores[:,1:]


            mask = None
            logit, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

            # print(ft1.squeeze().permute(1,2,0).cpu().numpy()[seg_label == 0][10000])
            # print(scores.squeeze().permute(1, 2, 0).cpu().numpy()[seg_label == 0][10000])

            # print(np.unique(seg_label))
            # print(np.unique(pred))

            # seg_label_unknown = seg_label.copy()
            # seg_label_unknown[seg_label != 13] = 1
            # seg_label_unknown[seg_label == 13] = 0
            # seg_label[seg_label >= 2] = 2
            # print(np.unique(seg_label))
            # plt.figure()
            # plt.imshow(seg_label)
            # # plt.imshow(colorEncode(seg_label,colors))
            # # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.savefig('imgs/heli_gt_' + str(cnt) + '.jpg', bbox_inches='tight', dpi=600, pad_inches = 0)
            # plt.show()
            # #
            # plt.figure()
            # plt.imshow(colorEncode(pred, colors))
            # # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # plt.savefig('imgs/heli_pred.jpg', bbox_inches='tight', dpi=600, pad_inches = 0)
            # plt.show()


            #
            # plt.figure()
            # plt.imshow(pred)
            # plt.show()
            # plt.figure()
            # plt.imshow(seg_label)
            # plt.show()
            # #

            #
            # conf, _ = torch.max(nn.functional.softmax(tmp_scores, dim=1), dim=1)
            # conf = as_numpy(conf.squeeze(0).cpu())
            # plt.figure()
            # plt.imshow((-conf))
            # plt.show()
            #
            # prob_map = nn.functional.softmax(tmp_scores, dim=1)
            # conf = stats.entropy(prob_map.squeeze().cpu().numpy(), base=2, axis=0)
            # plt.figure()
            # plt.imshow((conf))
            # plt.show()


            #for evaluating MSP
            if cfg.OOD.ood == "msp":
                conf, _  = torch.max(nn.functional.softmax(tmp_scores, dim=1),dim=1)
                conf = as_numpy(conf.squeeze(0).cpu())

                # plt.figure()
                # plt.imshow(-conf)
                # plt.xticks([])
                # plt.yticks([])
                # plt.axis('off')
                # plt.savefig('imgs/heli_msp.jpg',bbox_inches='tight', dpi=600, pad_inches = 0)
                # plt.show()

            elif cfg.OOD.ood == "maxlogit":
                conf, _  = torch.max(tmp_scores,dim=1)
                conf = as_numpy(conf.squeeze(0).cpu())

                # plt.figure()
                # plt.imshow(-conf)
                # plt.colorbar()
                # plt.xticks([])
                # plt.yticks([])
                # plt.axis('off')
                # plt.savefig('imgs/heli_maxlogit.jpg', bbox_inches='tight', dpi=600)
                # plt.show()

            elif cfg.OOD.ood == "dissum":
                dis_sum = torch.sum(tmp_scores,dim=1)
                dis_sum = -as_numpy(dis_sum.squeeze(0).cpu())
                dis_sum[dis_sum >= 400] = 400
                dis_sum = Normalizatoin(dis_sum)

                # plt.figure()
                # plt.imshow((dis_sum))
                # plt.colorbar()
                # plt.show()

                # thre = 0.1
                # dis_sum[dis_sum < thre] = 0
                # dis_sum[dis_sum > thre] = 1
                #
                # plt.figure()
                # plt.imshow((-dis_sum))
                # plt.xticks([])
                # plt.yticks([])
                # plt.axis('off')
                # # plt.colorbar()
                # plt.savefig('imgs/heli_dis_' + str(cnt)+ '.jpg', bbox_inches='tight', dpi=600, pad_inches = 0)
                # # cnt += 1
                # plt.show()


                pb_map = nn.functional.softmax(tmp_scores, dim=1).squeeze().cpu().numpy()

                # Draw the density function
                # cls_names = ['sky','building','fence','other','pedestrian','pole','roadline','road',
                #              'sidewalk','vegetation','car','wall','traffic sign','anomalous object']
                # labels = []
                # bins = range(0, 500, 10)
                # plt.figure()
                # instances, counts = np.unique(seg_label, False, False, True)
                # for cls_idx in range(len(instances)):
                #     print(instances[cls_idx])
                #     print(counts[cls_idx] / np.sum(counts))
                #     if counts[cls_idx] / np.sum(counts) > 0.04:
                #         dis_sum_cls = dis_sum[seg_label == instances[cls_idx]]
                #         labels.append(instances[cls_idx])
                #
                #         # plt.title(str(instance))
                #         plt.hist(dis_sum_cls, bins=bins, alpha=0.5, label=cls_names[instances[cls_idx]], histtype='stepfilled',density=True)
                #         # sns.distplot(dis_sum_cls, bins=bins, kde_kws={"color": "seagreen", "lw": 3}, hist_kws={"color": "b"})
                #         plt.ylim(0,0.035)
                #         plt.legend()
                # plt.savefig('dis_sum.jpg',dpi=1000)
                # plt.show()

                # plt.figure()
                # dis_sum_cls = dis_sum[seg_label != 13]
                # plt.hist(dis_sum_cls, bins=bins, alpha=0.5, label='0')
                # dis_sum_cls = dis_sum[seg_label == 13]
                # plt.hist(dis_sum_cls, bins=bins, alpha=0.5, label='1')
                # plt.legend()
                # plt.show()


                # Detect unknown through distance sum of each class
                # gm_dis = []
                # ecdf_list = []
                # thre_list = []
                # heat_class_map = np.zeros_like(pb_map)
                # for class_id in range(13):
                #     if class_id in np.unique(pred):
                #         gm_dis.append(GaussianMixture(n_components=3, random_state=0).fit(dis_sum[pred==class_id].reshape(-1,1)))
                #         print(gm_dis[class_id].means_)
                #         print(gm_dis[class_id].covariances_)
                #         ecdf_list.append(ECDF(dis_sum[pred==class_id]))
                #         mean_max = np.max(gm_dis[class_id].means_)
                #         mean_max_co = gm_dis[class_id].covariances_[np.argmax(gm_dis[class_id].means_)]
                #         thre_list.append(ecdf_list[class_id](mean_max - 6 * np.sqrt(mean_max_co)))
                #         print(thre_list[class_id])
                #         # plt.figure()
                #         # plt.title(str(class_id))
                #         # plt.hist(dis_sum[pred==class_id], bins=100)
                #         # plt.show()
                #         heat_class_map[class_id] = ecdf_list[class_id](dis_sum)
                #         heat_class_map[class_id] = 1 - Coefficient_map(heat_class_map[class_id], min(thre_list[class_id][0],np.array([0.05])))
                #         heat_class_map[class_id] = pb_map[class_id] * heat_class_map[class_id]
                #         # plt.figure()
                #         # plt.title(str(class_id))
                #         # plt.imshow(heat_class_map[class_id])
                #         # plt.colorbar()
                #         # plt.show()
                #
                #     else:
                #         gm_dis.append([])
                #         ecdf_list.append([])
                #         thre_list.append([])
                #
                # conf = np.sum(heat_class_map, axis=0)
                # plt.figure()
                # plt.imshow(conf)
                # plt.show()






                # pb_map = nn.functional.softmax(tmp_scores, dim=1).squeeze().cpu().numpy()
                # bkg_map = np.sum(pb_map[:11,:,:], axis=0) - np.sum(pb_map[5:8,:,:], axis=0)
                # conf = bkg_map
                # fg_map = np.max(pb_map.take([5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18], axis=0))                # plt.figure()
                # # plt.imshow(bkg_map)
                # # plt.colorbar()
                # # plt.show()
                # conf = -(1 - bkg_map) * (1 - fg_map)
                # # plt.figure()
                # # plt.imshow(conf)
                # # plt.colorbar()
                # # plt.show()

                # fg_cls = [10]
                # # fg_cls = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
                # bkg_map = 1 - np.sum(pb_map.take(fg_cls, axis=0), axis=0)
                # conf = bkg_map
                # fg_map = np.max(pb_map.take(fg_cls, axis=0), axis=0)
                # # plt.figure()
                # # plt.imshow(bkg_map)
                # # plt.colorbar()
                # # plt.show()
                # bkg_fg = -(1 - bkg_map) * (1 - fg_map)
                # bkg_fg = Normalizatoin(bkg_fg)
                # conf = bkg_fg

                # plt.figure()
                # plt.imshow(bkg_fg)
                # plt.colorbar()
                # plt.show()

                prob_map = np.max(nn.functional.softmax(tmp_scores, dim=1).squeeze().cpu().numpy(), axis=0)
                prob_map = Normalizatoin(prob_map)

                # plt.figure()
                # plt.imshow(prob_map)
                # plt.colorbar()
                # plt.show()

                # Coefficient = Normalizatoin(-Coefficient_map(dis_sum, 0.1))
                # plt.figure()
                # plt.imshow((Coefficient))
                # plt.colorbar()
                # plt.show()
                Coefficient = Coefficient_map(dis_sum, 0.2)
                conf = Coefficient * dis_sum + (1 - Coefficient) * prob_map
                # conf = Coefficient * dis_sum + (1 - Coefficient) * bkg_fg
                conf = dis_sum
                # plt.figure()
                # plt.imshow((-conf))
                # plt.colorbar()
                # plt.show()
                # conf += bkg_fg
                # conf = Normalizatoin(conf)
                # conf = dis_sum
                # plt.figure()
                # plt.imshow((-conf))
                # plt.xticks([])
                # plt.yticks([])
                # plt.axis('off')
                # # plt.colorbar()
                # plt.savefig('imgs/heli_dis_msp_03_20.jpg', bbox_inches='tight', dpi=600, pad_inches = 0)
                # plt.show()


            elif cfg.OOD.ood == "background":
                conf = tmp_scores[:, 0]
                conf = as_numpy(conf.squeeze(0).cpu())
            elif cfg.OOD.ood == "crf":
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
                ch,h,w = scores.squeeze(0).size()
                d = dcrf.DenseCRF2D(h, w, ch)  # width, height, nlabels
                tmp_scores = as_numpy(nn.functional.softmax(tmp_scores, dim=1).squeeze(0))
                tmp_scores = as_numpy(tmp_scores)
                U = unary_from_softmax(tmp_scores)
                d.setUnaryEnergy(U)

                pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=13, img=tmp_scores, chdim=0)
                d.addPairwiseEnergy(pairwise_energy, compat=10)
                # Run inference for 100 iterations
                Q_unary = d.inference(100)
                # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                map_soln_unary = np.argmax(Q_unary, axis=0)

                # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                map_soln_unary = map_soln_unary.reshape((h,w))
                conf = np.max(Q_unary, axis=0).reshape((h,w))

            elif cfg.OOD.ood == "crf-gauss":
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
                ch,h,w = scores.squeeze(0).size()
                d = dcrf.DenseCRF2D(h, w, ch)  # width, height, nlabels
                tmp_scores = as_numpy(nn.functional.softmax(tmp_scores, dim=1).squeeze(0))
                tmp_scores = as_numpy(tmp_scores)
                U = unary_from_softmax(tmp_scores)
                d.setUnaryEnergy(U)
                d.addPairwiseGaussian(sxy=3, compat=3)  # `compat` is the "strength" of this potential.

                # Run inference for 100 iterations
                Q_unary = d.inference(100)
                # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
                map_soln_unary = np.argmax(Q_unary, axis=0)

                # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
                map_soln_unary = map_soln_unary.reshape((h,w))
                conf = np.max(Q_unary, axis=0).reshape((h,w))
            elif cfg.OOD.ood == "knn":
                neighbor_size = 9
                ft1 = ft1.squeeze()
                cosdis_map = torch.zeros((ft1.shape[1:]))
                cosdis_map = async_copy_to(cosdis_map, gpu)

                c, h, w = ft1.shape
                for shift_num_h in range(1,neighbor_size):
                    for shift_num_w in range(1,neighbor_size):
                        shifted_ft = torch.zeros_like(ft1)
                        shifted_ft[:, 0:(h - shift_num_h), 0:(w - shift_num_w)] = ft1[:, shift_num_h:h, shift_num_w:w]
                        cosdis_map += nn.functional.cosine_similarity(ft1, shifted_ft, dim=0)
                        shifted_ft = torch.zeros_like(ft1)
                        shifted_ft[:, shift_num_h:h, shift_num_w:w] = ft1[:, 0:(h - shift_num_h), 0:(w - shift_num_w)]
                        cosdis_map += nn.functional.cosine_similarity(ft1, shifted_ft, dim=0)
                plt.figure()
                cosdis_map = nn.functional.interpolate(cosdis_map.unsqueeze(0).unsqueeze(0), size=segSize, mode='bilinear', align_corners=False)
                plt.imshow((-cosdis_map).squeeze().cpu().numpy())
                plt.show()
                conf = cosdis_map.squeeze().cpu()

            # valid_area = np.asarray(Image.open('LAF_roi_2048.png')) > 0
            # valid_area = (seg_label != 0)
            # invalid_area = (seg_label == 0)
            # print(np.unique(seg_label))
            # # conf[invalid_area] = 1
            # thre = 0.5
            # conf[conf < thre] = 0
            # conf[conf >= thre] = 1
            # plt.figure()
            # plt.imshow((-conf))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.colorbar()
            # # plt.savefig('imgs/heli_dis_msp.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()
            #
            # pred[conf < thre] = 13
            #
            # plt.figure()
            # plt.imshow(colorEncode(pred,colors))
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            # # plt.colorbar()
            # plt.savefig('imgs/heli_open_pred.jpg', bbox_inches='tight', dpi=600, pad_inches=0)
            # plt.show()


            # conf = conf[valid_area]
            # seg_label = seg_label[valid_area]



            res = eval_ood_measure(conf, seg_label, cfg, mask=mask)
            if res is not None:
                auroc, aupr, fpr = res
                aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)
            else:
                pass


        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        #Analyze
        # dis_sum_map = torch.sum(tmp_scores, dim=1)
        # dis_sum_map = -as_numpy(dis_sum_map.squeeze(0).cpu())
        # instances = np.unique(seg_label)
        # for instance in instances:
        #     if instance == 13:
        #         continue
        #     target = (seg_label == instance)
        #     prd = (pred == instance)
        #     TP = target * prd
        #     dis_sum_TP = dis_sum_map[TP].tolist()
        #     if len(dis_sum_TP) > 1500:
        #         dis_sum_TP = random.sample(dis_sum_TP, int(len(dis_sum_TP)))
        #         logit_dic[instance].append(dis_sum_TP)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # # Iou for unknown
        # threshold_list = [0.1 * int(n) for n in range(1,10)]
        # for i in range(9):
        #     unknown_pred = conf.copy()
        #     unknown_pred[conf < threshold_list[i]] = 0
        #     unknown_pred[conf >= threshold_list[i]] = 1
        #     intersection_unknown, union_unknown = intersectionAndUnion(unknown_pred, seg_label_unknown, 2)
        #     intersection_meter_unknown[i].update(intersection_unknown)
        #     union_meter_unknown[i].update(union_unknown)


        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)

    # Analyze

    # for i in range(13):
    #     print(len(logit_dic[i]))
    #     logit_temp = []
    #     for j in range(len(logit_dic[i])):
    #         logit_temp += logit_dic[i][j]
    #     if (len(logit_temp) != 0):
    #         ecdf_list[i].append(ECDF(logit_temp))
    # with open('logit_dict.json', 'w') as  f:
    #     json.dump(logit_dic, f)
    # joblib.dump(ecdf_list, 'ecdf_list.pkl')

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
    print("mean auroc = ", np.mean(aurocs), "mean aupr = ", np.mean(auprs), " mean fpr = ", np.mean(fprs))
    # print("mIoU for unknown")
    # unknown_iou = []
    # for i in range(9):
    #     print("Threshold for unknown is: ", threshold_list[i])
    #     iou_unknown = intersection_meter_unknown[i].sum / (union_meter_unknown[i].sum + 1e-10)
    #     for j, _iou in enumerate(iou_unknown):
    #         print('class [{}], IoU: {:.4f}'.format(j, _iou))
    #     unknown_iou.append(iou_unknown[0])
    # best_iou = np.max(unknown_iou)
    # best_thre = threshold_list[np.argmax(unknown_iou)]
    # print("Best Iou for unknown objects is: ", best_iou)
    # print("Corresponding threshold is: ", best_thre)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=1,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/test_ood_street.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "--ood",
        help="Choices are [msp, crf-gauss, crf, maxlogit, background, knn]",
        default="msp",
    )
    parser.add_argument(
        "--exclude_back",
        help="Whether to exclude the background class.",
        action="store_true",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    ood = ["OOD.exclude_back", args.exclude_back, "OOD.ood", args.ood]

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(ood)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)