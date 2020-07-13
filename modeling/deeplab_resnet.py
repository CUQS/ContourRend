import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.nn import BatchNorm2d
from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from utils.contour_mask_utils import gather_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _conv_up(input_channels, concat_channels, scale_factor=2, upflag=True):
    concat = nn.Conv2d(input_channels, concat_channels, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(concat_channels)
    relu = nn.ReLU(inplace=True)
    res_concat = nn.Sequential(concat, bn, relu)
    if upflag:
        up = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        res_concat_up = nn.Sequential(concat, bn, relu, up)
        return res_concat, res_concat_up
    else:
        return res_concat


def _make_conv_final(concat_channels):
    conv_final_1 = nn.Conv2d(4*concat_channels, 512, kernel_size=3,
                             padding=1, stride=2, bias=False)
    bn_final_1 = nn.BatchNorm2d(512)
    relu_final_1 = nn.LeakyReLU(inplace=True)
    conv_final_2 = nn.Conv2d(512, 1024, kernel_size=3,
                             padding=1, stride=2, bias=False)
    bn_final_2 = nn.BatchNorm2d(1024)
    relu_final_2 = nn.LeakyReLU(inplace=True)
    conv_final_3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, bias=False)
    bn_final_3 = nn.BatchNorm2d(2048)
    relu_final_3 = nn.LeakyReLU(inplace=True)
    conv_final = nn.Sequential(conv_final_1, bn_final_1, relu_final_1,
                               conv_final_2, bn_final_2, relu_final_2,
                               conv_final_3, bn_final_3, relu_final_3)
    return conv_final


def _make_conv_final_cat():
    final_conv1_cat = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
    final_conv1_cat_bn = nn.BatchNorm2d(512)
    final_conv2_cat = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
    final_conv2_cat_bn = nn.BatchNorm2d(512)
    conv_final_cat = nn.Sequential(final_conv1_cat, final_conv1_cat_bn,
                                   final_conv2_cat, final_conv2_cat_bn)
    return conv_final_cat


def _make_edge_annotation_concat(gcn_dim):
    concat_channels = gcn_dim
    edge_annotation_cnn_tunner_1 = nn.Conv2d(512 + 2, concat_channels, kernel_size=3, padding=1, bias=False)
    edge_annotation_cnn_tunner_bn_1 = nn.BatchNorm2d(concat_channels)
    edge_annotation_cnn_tunner_relu_1 = nn.ReLU(inplace=True)

    edge_annotation_cnn_tunner_2 = nn.Conv2d(concat_channels, concat_channels, kernel_size=3, padding=1, bias=False)
    edge_annotation_cnn_tunner_bn_2 = nn.BatchNorm2d(concat_channels)
    edge_annotation_cnn_tunner_relu_2 = nn.ReLU(inplace=True)

    edge_annotation_concat = nn.Sequential(edge_annotation_cnn_tunner_1,
                                           edge_annotation_cnn_tunner_bn_1,
                                           edge_annotation_cnn_tunner_relu_1,
                                           edge_annotation_cnn_tunner_2,
                                           edge_annotation_cnn_tunner_bn_2,
                                           edge_annotation_cnn_tunner_relu_2)

    return edge_annotation_concat


class DeepLabResnet(nn.Module):
    def __init__(self, concat_channels=64, gcn_dim=514):
        super(DeepLabResnet, self).__init__()
        self.cnn_feature_grids = [112, 56, 28, 28]
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.concat_channels = concat_channels
        self.feat_size = 28
        self.gcn_dim = gcn_dim

        self.image_feature_dim = 256
        # self.resnet = build_backbone("resnet", 8, SynchronizedBatchNorm2d)
        self.resnet = build_backbone("resnet", 8, BatchNorm2d)

        self.conv1_concat = _conv_up(64, concat_channels, upflag=False)
        self.res1_concat, self.res1_concat_up = _conv_up(256, concat_channels, 2)
        self.res2_concat, self.res2_concat_up = _conv_up(512, concat_channels, 4)
        self.res4_concat, self.res4_concat_up = _conv_up(2048, concat_channels, 4)
        # self.res5_concat = _conv_up(512, concat_channels, upflag=False)

        self.edge_annotation_concat = _make_edge_annotation_concat(self.gcn_dim)
        self.edge_annotation_channels = 64 * 5

        self.conv_final = _make_conv_final(concat_channels)

        # self.final_PSP = build_aspp("resnet", 8, SynchronizedBatchNorm2d)
        self.final_PSP = build_aspp("resnet", 8, BatchNorm2d)

        self.conv_final_cat = _make_conv_final_cat()

        self.prob_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(64, 2, kernel_size=3, padding=1, bias=False))

    def forward(self, x, final=False):
        x = self.normalize(x)

        conv1_f, layer1_f, layer2_f, layer3_f, layer4_f = self.resnet(x)

        conv1_f_up = self.conv1_concat(conv1_f)  # (bs, 64, 112, 112)
        layer1_f_up = self.res1_concat_up(layer1_f)  # (bs, 64, 112, 112)
        layer2_f_up = self.res2_concat_up(layer2_f)  # (bs, 64, 112, 112)
        layer4_f_up = self.res4_concat_up(layer4_f)  # (bs, 64, 112, 112)
        concat_features = torch.cat((conv1_f_up, layer1_f_up,
                                     layer2_f_up, layer4_f_up), dim=1)  # (bs, 256, 112, 112)

        final_features = self.conv_final(concat_features)  # (bs, 2048, 28, 28)
        final_features = self.final_PSP(final_features)  # (bs, 512, 28, 28)
        x_prob = F.interpolate(final_features, size=224, mode='bilinear', align_corners=True)
        x_prob = self.prob_conv(x_prob)  # (bs, 2, 224, 224)

        return x_prob, final_features

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)

    def edge_annotation_cnn(self, feature):

        final_feature_map = self.edge_annotation_concat(feature)

        # return final_feature_map.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[-1]**2, self.edge_annotation_channels)
        return final_feature_map

    def sampling(self, ids, features):
        cnn_out_feature = []
        for i in range(ids.size()[1]):
            id = ids[:, i, :]
            cnn_out = gather_feature(id, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features

