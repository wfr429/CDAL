import torch.nn as nn
import torch_dct as dct
import timm
import torch
import torch.nn.functional as F
import random
import kornia
import numpy as np
import math
from timm.models.layers import CondConv2d

EPSILON = 1e-6

def get_input_data(input_img, data='dct'):
    if data == 'dct':
        return dct.dct_2d(input_img)
    elif data == 'img':
        return input_img

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class CDAL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cfi = CFI(in_channels, out_channels)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = self.cfi(features)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + EPSILON)
        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)

        return feature_matrix, counterfactual_feature, fake_att



class CEC(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1,
        groups=1, bias=False, num_experts=4
    ):
        super().__init__()
        self.routing = nn.Linear(in_channels, num_experts)
        self.cond_conv = CondConv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, num_experts
        )

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled))
        return self.cond_conv(x, routing_weights)


class CFI(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, ratio=2,
        dw_kernel_size=3, stride=1, num_experts=4
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        expand_channels = init_channels * (ratio - 1)

        self.cross = nn.Sequential(
            CEC(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False, num_experts=num_experts),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.depth = nn.Sequential(
            CEC(init_channels, expand_channels, dw_kernel_size, stride=1,
                padding=dw_kernel_size // 2, groups=init_channels, bias=False, num_experts=num_experts),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.cross(x)
        x2 = self.depth(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class CAA(nn.Module):
    def __init__(self, kernel_size=11, sigma=7, scale_factor=0.5, noise_rate=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.noise_rate = noise_rate

        self.gaussian_blur = kornia.filters.GaussianBlur2d(
            (kernel_size, kernel_size), (sigma, sigma)
        )

    def forward(self, features, att_map1, att_map2):
        att_map1, index = self._sample_attention(att_map1)
        att_map2, _ = self._sample_attention(att_map2)
        augmented = self._augment(features, att_map1, att_map2)
        return augmented, index

    def _sample_attention(self, attention_maps):
        B, C, H, W = attention_maps.shape

        channel_weights = torch.sqrt(attention_maps.sum(dim=(2, 3)) + EPSILON)
        probs = F.normalize(channel_weights, p=1, dim=1)

        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        index_mask = sampled_indices.view(B, 1, 1, 1).expand(B, 1, H, W)

        selected_attention = torch.gather(attention_maps, dim=1, index=index_mask)
        return selected_attention, sampled_indices

    def _augment(self, x, att1, att2):
        B, C, H, W = x.shape
        xs = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        xs = self.gaussian_blur(xs)
        xs += torch.randn_like(xs) * self.noise_rate
        xs = F.interpolate(xs, size=(H, W), mode='bilinear', align_corners=True)

        return x * att1 + xs * att2


class Simple_CNN(nn.Module):
    def __init__(self, class_num=15, out_feature_result=False):
        super(Simple_CNN, self).__init__()
        nf = 64
        nc = 3

        self.main1 = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),
        )
        self.main2 = nn.Sequential(
            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),
        )
        self.main3 = nn.Sequential(
            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),
        )
        self.main4 = nn.Sequential(
            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),
        )

        self.fc = nn.Linear(nf * 8 * 8 * 8, nf * 8, bias=True)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(nf * 8, class_num, bias=True)
        )
        self.out_feature_result=out_feature_result
        self.encoder_channels = nf * 4
        self.M = 64
        self.cdal = CDAL(self.encoder_channels, self.M)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.M * self.encoder_channels, class_num, bias=True)
        )
        self.cfi = CFI(self.encoder_channels, self.M)
        self.augment = CAA()


    def fss_features(self, feature_maps):
        attention_maps = self.cfi(feature_maps)
        feature_matrix, feature_matrix_hat, attention_maps_hat = self.cdal(feature_maps, attention_maps)

        p = self.fc1(feature_matrix * 100.)
        p_hat = self.fc1(feature_matrix_hat * 100.)
        return p, p_hat, attention_maps, attention_maps_hat

    def att_feat(self, feature_maps, attention_maps):
        feature_matrix = torch.einsum('imjk,injk->imn', attention_maps, feature_maps)  # [B, M, C_F]
        feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)

        return feature_matrix

    def forward(self, input, data='dct'):

        input = get_input_data(input, data)
        embedding1 = self.main1(input)
        embedding2 = self.main2(embedding1)
        embedding3 = self.main3(embedding2)
        embedding = self.main4(embedding3)
        feature = embedding.view(embedding.shape[0], -1)
        feature = self.fc(feature)
        cls_output = self.classification_head(feature)

        #CDAL
        res1 = embedding3
        y_pred_raw, y_pred_aux, attention_maps, attention_maps_hat = self.fss_features(res1) # torch.Size([128, 16384])

        aug, index = self.augment(res1, attention_maps, attention_maps_hat)
        y_aug_raw, y_aug_aux, attention_maps_aug, attention_maps_hat_aug = self.fss_features(aug)

        feature_matrix = self.att_feat(res1, attention_maps)
        feature_matrix_aug = self.att_feat(aug, attention_maps_aug)
        one_hot = F.one_hot(index, self.M)

        match_loss = torch.mean(torch.norm(feature_matrix - feature_matrix_aug, dim = -1) * (torch.ones_like(one_hot) - one_hot))


        if self.out_feature_result:
            return cls_output, feature, y_pred_raw, y_pred_aux, y_aug_raw, y_aug_aux, match_loss
        else:
            return cls_output