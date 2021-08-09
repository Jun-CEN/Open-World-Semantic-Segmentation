import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class CrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0, beta=0, gamma=0, size_average=True, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, logit, target, features_in):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,size_average=self.size_average)

        if self.cuda:
            criterion = criterion.cuda()

        CE_loss = criterion(logit, target.long())
        return CE_loss / n
        VAR_loss = Variable(torch.Tensor([0])).cuda()
        Inter_loss = Variable(torch.Tensor([0])).cuda()
        Center_loss = Variable(torch.Tensor([0])).cuda()
        for i in range(n):
            label = target[i]
            label = label.flatten().cpu().numpy()
            features = logit[i]
            features = features.permute(1, 2, 0).contiguous()
            shape = features.size()
            features = features.view(shape[0]*shape[1], shape[2])
            features_in_temp = features_in[i]

            instances, counts = np.unique(label, False, False, True)
            # print('counts', counts)
            total_size = int(np.sum(counts))
            for instance in instances:

                if instance == self.ignore_index:  # Ignore background
                    continue

                locations = torch.LongTensor(np.where(label == instance)[0]).cuda()
                vectors = torch.index_select(features, dim=0, index=locations)
                features_temp = torch.index_select(features_in_temp, dim=0, index=locations)
                centers_temp = torch.mean(features_temp, dim=0)
                features_temp = features_temp - centers_temp
                Center_loss += torch.sum(features_temp ** 2) / total_size
                # print(size)
                # print(-vectors[:,int(instance)])
                # get instance mean and distances to mean of all points in an instance
                VAR_loss += torch.sum((-vectors[:,int(instance)]))/total_size
                Inter_loss += (torch.sum(vectors) - torch.sum((vectors[:,int(instance)]))) / total_size

                # total_size += size

            # VAR_loss += var_loss/total_size

        loss = (CE_loss + self.alpha * VAR_loss + self.beta * Inter_loss +self.gamma * Center_loss) / n
        # print(CE_loss/n, self.alpha * VAR_loss/n, self.beta * Inter_loss/n, self.gamma * Center_loss/n)

        return loss

class CrossEntropyLoss_dis(nn.Module):
    def __init__(self, alpha=0, beta=0, gamma=0, size_average=True, ignore_index=255):
        super(CrossEntropyLoss_dis, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, logit, target, features_1, features_2):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,size_average=self.size_average)

        if self.cuda:
            criterion = criterion.cuda()

        CE_loss = criterion(logit, target.long())

        return CE_loss / n

        DIS_loss = Variable(torch.Tensor([0])).cuda()

        appendix_lay = torch.zeros(n,w,h,1).cuda()
        features_1 = torch.cat((features_1, appendix_lay), dim=3)
        # print('features_1.shape: ', features_1.shape)
        # print('features_2.shape: ', features_2.shape)

        for i in range(n):
            features_origin = features_1[i][target[i] != 16]
            features_new = features_2[i][target[i] != 16]
            features_diff = features_new - features_origin
            DIS_loss += torch.sum(features_diff ** 2) / (features_diff.shape[0])

        loss = CE_loss / n + 0.01 * DIS_loss / n
        # print(CE_loss, DIS_loss)



        return loss