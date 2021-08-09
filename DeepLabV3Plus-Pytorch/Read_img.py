# from PIL import Image
import matplotlib.pyplot as plt
# import numpy as np
import bdlb
import torch

# path_img = './Res/results_18_embedding/7_'
# path_img = './results_18_ce_noshuffle/2_'
#
# image = Image.open(path_img + 'image.png')
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

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset('LostAndFound')
print(data)

# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    plt.imshow(image)
    uncertainty = torch.rand(image.shape[0], image.shape[1])
    return uncertainty

metrics = fs.evaluate(estimator, data.take(5000))
print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
print('My method achieved {:.2f}% AUROC'.format(100 * metrics['auroc']))
print('My method achieved {:.2f}% FPR@95%TPR'.format(100 * metrics['FPR@95%TPR']))
