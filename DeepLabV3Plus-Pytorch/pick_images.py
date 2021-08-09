from shutil import copyfile
import os
import random

source_dir = '/data1/users/caijh28/data/cityscapes/leftImg8bit/train'
targets_dir = '/data1/users/caijh28/data/cityscapes/gtFine/train'
images_all = []
targets_all = []
file_name_all = []
targets_name_all = []

for city in os.listdir(source_dir):
    img_dir = os.path.join(source_dir, city)
    target_dir = os.path.join(targets_dir, city)
    files_name = os.listdir(img_dir)
    files_name = sorted(files_name)
    for file_name in files_name:
        images_all.append(os.path.join(img_dir, file_name))
        file_name_all.append(file_name)
        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],'gtFine_labelIds.png')
        targets_name_all.append(target_name)
        targets_all.append(os.path.join(target_dir, target_name))

seq = list(range(2975))
random.shuffle(seq)
index = seq[:1000]
print(len(index))
print([images_all[i] for i in index])
print([targets_all[i] for i in index])
print([file_name_all[i] for i in index])

images_100 = [images_all[i] for i in index]
targets_100 = [targets_all[i] for i in index]
files_name_100 = [file_name_all[i] for i in index]
targets_name_100 = [targets_name_all[i] for i in index]

for i in range(len(images_100)):
    copyfile(images_100[i],'/data1/users/caijh28/data/cityscapes/leftImg8bit/car_1000/car_1000/'+files_name_100[i])
    copyfile(targets_100[i], '/data1/users/caijh28/data/cityscapes/gtFine/car_1000/car_1000/' + targets_name_100[i])


