from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import bdlb
import torch
import json

# path_img = './data/test_result/t5/'
# path_img = './results_18_ce_noshuffle/2_'
#
# image = Image.open(path_img + '100.png')
# plt.imshow(image)
# plt.show()
#
# overlay = Image.open(path_img + 'overlay.png')
# plt.imshow(overlay)
# plt.show()
#
# pred = Image.open(path_img + 'pred.png')
# plt.imshow(pred)
# plt.show()
#
# target = Image.open(path_img + 'target.png')
# plt.imshow(target)
# plt.show()
#
# scores = Image.open(path_img + 'scores.png')
# scores = np.array(scores) / 255
# plt.imshow(scores)
# plt.show()
#
# dis_sum = np.load(path_img + 'dis_sum.npy')
# plt.imshow(dis_sum)
# plt.show()

with open('logit_dict.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)

for i in range(13):
    print(len(json_data[i]))
    plt.figure()
    plt.hist(json_data[i])
    plt.show()
