import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import json

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class _SimpleSegmentationModel_embedding(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel_embedding, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.centers = torch.zeros(17, 17)
        # idx = 0
        # for i in range(19):
        #     if i <= 12 or i >=16:
        #         self.centers[idx] = torch.tensor(np.mean(np.array(prototype[idx]), axis=0))
        #         idx += 1
        magnitude = 3

        for i in range(17):
            self.centers[i][i] = magnitude

        # cnt = 0
        # for i in range(17):
        #     if i <= 12:
        #         self.centers[cnt][cnt] = magnitude
        #         cnt += 1
        #     elif i > 13:
        #         self.centers[cnt+1][cnt] = magnitude
        #         cnt += 1
        # self.centers[13] = torch.ones(1,16) * 3

        # print(self.centers)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        output_size = x.size()
        # print(output)
        # print(np.unique(output.cpu().numpy()[0][0]))
        features = x.permute(0, 2, 3, 1).contiguous()  # batch * h * w * num_class
        features_out = features
        shape = features.size()
        features = features.view(shape[0], shape[1] * shape[2], shape[3])  # batch * hw * num_class
        num_classes = output_size[1]
        features_shape = features.size()
        features = features.unsqueeze(2).expand(features_shape[0], features_shape[1], num_classes,
                                                features_shape[2])  # batch * hw * num_class * num_class
        # print(features.size())
        # print(self.centers.size())

        self.centers = torch.zeros(shape[3], shape[3])
        m = 3
        for i in range(shape[3]):
            self.centers[i][i] = m
        # print(self.centers.shape)

        dists = features - self.centers.cuda()  # batch * hw * num_classes * c
        # print(dists.size())
        dist2mean = -torch.sum(dists ** 2, 3)  # batch * hw * num_classes
        # print(dist2mean.size())
        # m = nn.Softmax(dim=2)
        # prob = m(dist2mean)  # batch * hw * num_classes
        # print(prob)
        x = dist2mean.permute(0, 2, 1).contiguous().view(output_size[0], num_classes, output_size[2],
                                                         output_size[3])
        return x, self.centers.cuda(), features_out

class _SimpleSegmentationModel_embedding_self_distillation(nn.Module):
    def __init__(self, backbone):
        super(_SimpleSegmentationModel_embedding_self_distillation, self).__init__()
        self.backbone = backbone
        self.classifier_list = ['classifier']
        self.cls_novel = 1
        for i in range(self.cls_novel):
            self.classifier_list.append('classifier_' + str(i+1))
        inplanes = 2048
        low_level_planes = 256
        aspp_dilate = [6, 12, 18]
        num_classes = 16
        self.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        for i in range(self.cls_novel):
            self.__setattr__(self.classifier_list[i+1], DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes + i + 1, aspp_dilate))
        self.centers = torch.zeros(17, 17)

    def forward(self, x):
        # for m in self.__getattr__(self.classifier_list[-1]).modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.train()
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = []
        centers = []
        features_out = []
        logits_0, centers_0, features_out_0 = self.forward_single(self.classifier, features, input_shape)
        logits.append(logits_0)
        centers.append(centers_0)
        features_out.append(features_out_0)
        for i in range(self.cls_novel):
            classifier_temp = self.__getattr__(self.classifier_list[i+1])
            logits_tmp, centers_tmp, features_out_tmp = self.forward_single(classifier_temp, features, input_shape)
            logits.append(logits_tmp)
            centers.append(centers_tmp)
            features_out.append(features_out_tmp)
        return logits, centers, features_out

    def forward_single(self, classifier, features, input_shape):
        x = classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        output_size = x.size()
        # print(output)
        # print(np.unique(output.cpu().numpy()[0][0]))
        features = x.permute(0, 2, 3, 1).contiguous()  # batch * h * w * num_class
        features_out = features
        shape = features.size()
        features = features.view(shape[0], shape[1] * shape[2], shape[3])  # batch * hw * num_class
        num_classes = output_size[1]
        features_shape = features.size()
        features = features.unsqueeze(2).expand(features_shape[0], features_shape[1], num_classes,
                                                features_shape[2])  # batch * hw * num_class * num_class
        # print(features.size())
        # print(self.centers.size())

        self.centers = torch.zeros(shape[3], shape[3])
        m = 3
        for i in range(shape[3]):
            self.centers[i][i] = m
        # print(self.centers)

        dists = features - self.centers.cuda()  # batch * hw * num_classes * c
        # print(dists.size())
        dist2mean = -torch.sum(dists ** 2, 3)  # batch * hw * num_classes
        # print(dist2mean.size())
        # m = nn.Softmax(dim=2)
        # prob = m(dist2mean)  # batch * hw * num_classes
        # print(prob)
        x = dist2mean.permute(0, 2, 1).contiguous().view(output_size[0], num_classes, output_size[2],
                                                         output_size[3])



        return x, self.centers.cuda(), features_out


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module