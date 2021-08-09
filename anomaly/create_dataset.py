import numpy as np
import scipy
import scipy.io as sio
import scipy.misc
from scipy.misc import imread, imsave
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import os.path
from tqdm import tqdm
import re

# Replace the colors with our colors 
# This is only used for visualization purposes
#color_mat = sio.loadmat("data_ADE/color150.mat")

#StreetHazards colors
#colors = np.array([[ 0,   0,   0],# // unlabeled     =   0,
#        [ 70,  70,  70],# // building      =   1,
#        [190, 153, 153],# // fence         =   2,
#        [250, 170, 160],# // other         =   3,
#        [220,  20,  60],# // pedestrian    =   4,
#        [153, 153, 153],# // pole          =   5,
#        [157, 234,  50],# // road line     =   6,
#        [128,  64, 128],# // road          =   7,
#        [244,  35, 232],# // sidewalk      =   8,
#        [107, 142,  35],# // vegetation    =   9,
#        [  0,   0, 142],# // car           =  10,
#        [102, 102, 156],# // wall          =  11,
#        [220, 220,   0],# // traffic sign  =  12,
#        [ 60, 250, 240],# // anomaly       =  13,
#
#        ])

#color_mat["colors"] = colors
#sio.savemat("data/color150.mat", color_mat)


#####
#create the train and val obgt

# def create_odgt(root_dir, file_dir, ann_dir, out_dir, anom_files=None):
#     if anom_files is  None:
#         anom_files = []
#     _files = []
#
#     count1 = 0
#     count2 = 0
#
#     img_files = sorted(os.listdir(os.path.join(root_dir,file_dir)))
#     for img in img_files:
#         ann_file = img.replace('.jpg', '_train_id.png')
#         ann_file_path = os.path.join(root_dir,ann_dir,ann_file)
#         # print(ann_file_path)
#         if os.path.exists(ann_file_path):
#             dict_entry = {
#                 "dbName": "BDD-anonymous",
#                 "width": 1280,
#                 "height": 720,
#                 "fpath_img": os.path.join(file_dir, img),
#                 "fpath_segm": os.path.join(ann_dir, ann_file),
#             }
#             # If converting BDD100K uncomment out the following
#             img = imread(ann_file_path)
#             # if np.any(np.logical_or( (img == 18))):
#             #    count2 += 1
#             #    anom_files.append(dict_entry)
#             # if 16 in np.unique(img) or 17 in np.unique(img) or 18 in np.unique(img):
#             #    count2 += 1
#             #    anom_files.append(dict_entry)
#             # else:
#             count1 += 1
#             _files.append(dict_entry)
#
#     print("total images in = {} and out =  {}".format(count1, count2))
#
#     with open(out_dir, "w") as outfile:
#         json.dump(_files, outfile)
#
#     # If converting BDD100K uncomment out the following
#     # with open(root_dir + "anom.odgt", "w") as outfile:
#     #    json.dump(anom_files, outfile)
#
#     return anom_files
#
#
# out_dir = "/home/amax_cjh/caijh28/data/bdd100k/bdd100k/seg/train_all.odgt"
# root_dir = "/home/amax_cjh/caijh28/data/bdd100k/bdd100k/seg/"
# train_dir = "images/train"
# ann_dir = "labels/train"
# anom_files = create_odgt(root_dir, train_dir, ann_dir, out_dir)
#
# out_dir = "/home/amax_cjh/caijh28/data/bdd100k/bdd100k/seg/val_all.odgt"
# root_dir = "/home/amax_cjh/caijh28/data/bdd100k/bdd100k/seg/"
# train_dir = "images/val"
# ann_dir = "labels/val"
# create_odgt(root_dir, train_dir, ann_dir, out_dir, anom_files=anom_files)


# out_dir = "data/test_all.odgt"
# root_dir = "data/"
# val_dir = "images/test/"
# ann_dir = "annotations/test/"
# create_odgt(root_dir, val_dir, ann_dir, out_dir)



# BDD100K label map
#colors = np.array(
#    [0,    # road
#    1,     #sidewalk
#    2,     # building
#    3,     # wall
#    4,     # fence
#    5,     # pole
#    6,     # traffic light
#    7,     # traffic sign
#    8,     # vegetation
#    9,     # terrain
#    10,    # sky
#    11,    # person
#    12,    # rider
#    13,    # car
#    14,    # truck
#    15,    # bus
#    16,    # train
#    17,    # motorcycle
#    18,    # bicycle
#    255,]) # other

### convert BDD100K semantic segmentation images to correct labels

def convert_bdd(root_dir, ann_dir):
    count = 0
    for img_loc in tqdm(os.listdir(root_dir+ann_dir)):
        img = imread(root_dir+ann_dir+img_loc)
        if img.ndim <= 1:
            continue
        #swap 255 with -1
        #16 -> 19
        #18 -> 16
        #19 -> 18
        # add 1 to whole array
        loc = img == 255
        img[loc] = -1
        loc = img == 16
        img[loc] = 19
        loc = img == 18
        img[loc] = 16
        loc = img == 19
        img[loc] = 18
        img += 1
        scipy.misc.toimage(img, cmin=0, cmax=255).save(root_dir+ann_dir+img_loc)


# root_dir = "data/"
# ann_dir = "seg/train_labels/train/"
# # convert the BDD100K semantic segmentation images.
# convert_bdd(root_dir, ann_dir)

# def create_odgt_road_anom(root_dir, file_dir, out_dir):
#     _files = []
#
#     count1 = 0
#
#     img_files = sorted(os.listdir(os.path.join(root_dir,file_dir)))
#     for img in img_files:
#         if img.endswith('jpg'):
#
#             ann_file = img.replace('.jpg', '.labels')
#             ann_file_path = os.path.join(root_dir, file_dir, ann_file, 'labels_semantic.png')
#             # print(ann_file_path)
#             if os.path.exists(ann_file_path):
#                 dict_entry = {
#                     "dbName": "BDD-anonymous",
#                     "width": 1280,
#                     "height": 720,
#                     "fpath_img": os.path.join(file_dir, img),
#                     "fpath_segm": os.path.join(file_dir, ann_file, 'labels_semantic.png'),
#                 }
#                 count1 += 1
#                 _files.append(dict_entry)
#                 print(dict_entry)
#
#     print("total images in = {}".format(count1))
#
#     with open(out_dir, "w") as outfile:
#         json.dump(_files, outfile)
#
#     # If converting BDD100K uncomment out the following
#     # with open(root_dir + "anom.odgt", "w") as outfile:
#     #    json.dump(anom_files, outfile)
#
#     return None
#
# out_dir = "/data1/users/caijh28/data/roadanomaly/RoadAnomaly_jpg/anom.odgt"
# root_dir = "/data1/users/caijh28/data/roadanomaly/RoadAnomaly_jpg"
# train_dir = "frames"
# create_odgt_road_anom(root_dir, train_dir, out_dir)

def create_odgt_LAF(root_dir, file_dir, anno_dir, out_dir):
    _files = []
    all_frames = []
    not_interested = []
    seq_intetested = []
    count1 = 0

    cities = sorted(os.listdir(os.path.join(root_dir,file_dir)))
    for city in cities:
        for img in os.listdir(os.path.join(root_dir,file_dir,city)):
            if img.endswith('png'):

                ann_file = img.replace('leftImg8bit', 'gtCoarse_labelIds')
                ann_file_path = os.path.join(root_dir, anno_dir, city, ann_file)
                m = re.compile(r'([0-9]{2})_.*_([0-9]{6})_([0-9]{6})').match(img)
                all_frames.append(dict(scene_id = int(m.group(1)), scene_seq = int(m.group(2)),scene_time = int(m.group(3))))
                # print(all_frames[count1])
                if os.path.exists(ann_file_path):
                    dict_entry = {
                        "dbName": "BDD-anonymous",
                        "width": 1280,
                        "height": 720,
                        "fpath_img": os.path.join(file_dir, city, img),
                        "fpath_segm": os.path.join(anno_dir, city, ann_file),
                    }
                    label = imread(ann_file_path)
                    if len(np.unique(label)) == 1:
                        # not_interested.append(count1)
                        # count1 += 1
                        continue
                    count1 += 1
                    _files.append(dict_entry)
                    # print(dict_entry)
    # print(count1)
    # count = 0
    # scenes_by_id = dict()
    # # print(all_frames[0])
    # # print(all_frames[-1])
    #
    # for fr in all_frames:
    #     scene_seqs = scenes_by_id.setdefault(fr['scene_id'], dict())
    #     seq_times = scene_seqs.setdefault(fr['scene_seq'], dict())
    #     seq_times[fr['scene_time']] = count
    #     count += 1
    # # print(scenes_by_id[2][18][80])
    # # print(scenes_by_id[15][3][160])
    # for sc_name, sc_sequences in scenes_by_id.items():
    #     for seq_name, seq_times in sc_sequences.items():
    #         # ts = list(seq_times.keys())
    #         # ts.sort()
    #         # ts_sel = ts[-1:]
    #         # self.frames_interesting += [seq_times[t] for t in ts_sel]
    #
    #         t_last = max(seq_times.keys())
    #         seq_intetested.append(seq_times[t_last])
    # print(len(seq_intetested))
    #
    # final_files = [_files[index] for index in seq_intetested if index not in not_interested]




    print("total images in = {}".format(len(_files)))

    with open(out_dir, "w") as outfile:
        json.dump(_files, outfile)

    # If converting BDD100K uncomment out the following
    # with open(root_dir + "anom.odgt", "w") as outfile:
    #    json.dump(anom_files, outfile)

    return None

out_dir = "/data1/users/caijh28/data/lost_found/anom_all.odgt"
root_dir = "/data1/users/caijh28/data/lost_found"
train_dir = "leftImg8bit/test"
anno_dir = "gtCoarse/test"
create_odgt_LAF(root_dir, train_dir, anno_dir, out_dir)