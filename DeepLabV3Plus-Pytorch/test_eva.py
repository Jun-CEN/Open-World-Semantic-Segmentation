from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F_img

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import bdlb
from scipy import stats

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
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
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


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    AUC_scores = []
    AUC_scores_all = []
    bad = [482, 463, 459]
    with torch.no_grad():
        for i, (images, labels, labels_true) in tqdm(enumerate(loader)):

            # if i not in bad:
            #     continue
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            soft_outs = F.softmax(outputs, dim=1)
            scores = 1 - soft_outs.detach().max(dim=1)[0].cpu().numpy()

            # plt.figure()
            # plt.imshow(images.squeeze().permute(1,2,0).cpu().numpy())
            # plt.show()
            #
            # plt.figure()
            # plt.imshow(labels.squeeze().cpu().numpy())
            # plt.show()
            #
            # plt.figure()
            # plt.imshow(scores.squeeze())
            # plt.show()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            # scores_auc = scores[labels_true != 255]
            # msk_auc = labels[labels_true != 255]
            # msk_auc[msk_auc != 255] = 0
            # msk_auc[msk_auc == 255] = 1

            # scores_auc = scores.copy()
            # msk_auc = labels.clone()
            # msk_auc[msk_auc != 255] = 0
            # msk_auc[msk_auc == 255] = 1

            # msk_auc = msk_auc.cpu().numpy().ravel()
            # scores_auc = scores_auc.ravel()
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
            #     # scores_auc_max = scores_auc[msk_auc != counts_min_index]
            #     # msk_auc_max = msk_auc[msk_auc != counts_min_index]
            #     # # print(scores_auc_max.shape, msk_auc_max.shape)
            #     #
            #     # counts_max_index = np.random.choice(len(scores_auc_max), counts_min)
            #     # scores_auc_max = scores_auc_max[counts_max_index]
            #     # msk_auc_max = msk_auc_max[counts_max_index]
            #     # # print(scores_auc_max.shape, msk_auc_max.shape)
            #     #
            #     # scores_auc = np.hstack((scores_auc_min, scores_auc_max))
            #     # msk_auc = np.hstack((msk_auc_min, msk_auc_max))
            #     # print(scores_auc.shape, msk_auc.shape)
            #
            #     auc = Metrics.roc_auc_score(msk_auc, scores_auc)
            #     print(auc)
            #     AUC_scores.append(auc)
            #     AUC_scores_all.append(auc)
            # else:
            #     AUC_scores_all.append(1)


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
        # print(np.mean(np.array(AUC_scores)))
        # print(np.min(np.array(AUC_scores)))
        # print(np.max(np.array(AUC_scores)))
        # print(np.sort(np.array(AUC_scores_all)))
        # print(np.argsort(np.array(AUC_scores_all)))
    return score, ret_samples

def estimator(image):
    """Assigns a random uncertainty per pixel."""
    # plt.imshow(image)
    # uncertainty = torch.rand(image.shape[0], image.shape[1])
    image = image.numpy()
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    image = F_img.to_tensor(image)
    # print(image[200][200])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = F_img.normalize(image, mean, std)
    # matplotlib.image.imsave('test.jpg', image)
    # img = Image.fromarray(image)
    # img.save('test.jpg')
    image = image.to(device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        soft_outs = F.softmax(outputs, dim=1)

        # prob_map = nn.functional.softmax(tmp_scores, dim=1)
        uncertainty = torch.tensor(stats.entropy(soft_outs.cpu(), base=2, axis=1))

        # uncertainty = 1 - soft_outs.detach().max(dim=1)[0].cpu()


    return uncertainty

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
    criterion = utils.CrossEntropyLoss(ignore_index=255, size_average=True)


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

utils.mkdir('checkpoints_6cls_ce')
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
    model = nn.DataParallel(model)
    model.to(device)

#==========   Train Loop   ==========#
vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                  np.int32) if opts.enable_vis else None  # sample idxs for visualization
denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

# image_test = Image.open('test.jpg').convert('RGB')
# image_test = np.array(image_test)
# plt.figure()
# plt.imshow(image_test)
# plt.show()
# # image_test = F_img.to_tensor(image_test)
# # mean = [0.485, 0.456, 0.406]
# # std = [0.229, 0.224, 0.225]
# # image = F_img.normalize(image, mean, std)
# print(image_test[200][200])

model.eval()
if opts.test_only:
    fs = bdlb.load(benchmark="fishyscapes")
    # automatically downloads the dataset
    data = fs.get_dataset('LostAndFound')
    metrics = fs.evaluate(estimator, data.take(5000))
    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
    print('My method achieved {:.2f}% AUROC'.format(100 * metrics['auroc']))
    print('My method achieved {:.2f}% FPR@95%TPR'.format(100 * metrics['FPR@95%TPR']))
