import warnings
import numpy as np
import timm
from loguru import logger

from timm.models.layers import CondConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import kornia
import math
from pytorch_wavelets import DWTForward

__all__ = ['BinaryClassifier']
EPSILON = 1e-6

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)

warnings.filterwarnings("ignore")


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


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(
            np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - \
            (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size),
                           stride=list(stride_size))
        x = avg(x)
        return x


class BaseClassifier(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='no',
                 pretrained=False,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck

        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool = AdaptiveAvgPool2dCustom((1, 1))

        self.dropout = nn.Dropout(drop_rate)

        if self.neck == 'bnneck':
            logger.info('Using BNNeck')
            self.bottleneck = nn.BatchNorm1d(self.num_features)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc2 = nn.Linear(
                self.num_features, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.fc2.apply(weights_init_classifier)
        else:
            self.fc2 = nn.Linear(self.num_features, self.num_classes)

    def forward_featuremaps(self, x):
        featuremap = self.encoder.forward_features(x)
        return featuremap

    def forward_features(self, x):
        featuremap = self.encoder.forward_features(x)
        feature = self.pool(featuremap).flatten(1)

        if self.neck == 'bnneck':
            feature = self.bottleneck(feature)

        return feature

    def forward(self, x, label=None):
        feature = self.forward_features(x)

        x = self.dropout(feature)
        method = self.fc2(x)

        y = method

        if self.is_feat:
            return y, feature

        return y


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



class CPLClassifier(nn.Module):
    def __init__(self,
                 encoder, #resnet50
                 num_classes=20,
                 num_patch=3,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='bnneck',
                 pretrained=False,
                 M=64,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_patch = num_patch
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck
        self.M = M

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.backbone = timm.create_model(encoder, features_only=True, out_indices=(1, 4), pretrained=pretrained)
        self.encoder_channels = self.backbone.feature_info.channels()

        self.num_features_low = self.encoder_channels[0]
        self.num_features_high = self.encoder_channels[1]
        
        # CDAL
        self.cdal = CDAL(self.num_features_low, self.M)
        self.cfi = CFI(self.num_features_low, self.M)
        self.augment = CAA()
        self.fc1 = nn.Linear(self.M * self.num_features_low, self.num_classes, bias=False)
        self.fc1.apply(weights_init_classifier)

        self.pool = AdaptiveAvgPool2dCustom((1, 1))
        self.part = AdaptiveAvgPool2dCustom((self.num_patch, self.num_patch))

        self.dropout = nn.Dropout(drop_rate)

        self.bottleneck = nn.BatchNorm1d(self.num_features_high)
        self.bottleneck.bias.requires_grad_(False)
        self.fc2 = nn.Linear(self.num_features_high, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)

        for i in range(self.num_patch ** 2):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(self.num_features_high))
            getattr(self, name).bias.requires_grad_(False)
            getattr(self, name).apply(weights_init_kaiming)

    def forward_features(self, featuremap):
        f_g = self.pool(featuremap).flatten(1)
        f_p = self.part(featuremap)
        f_p = f_p.view(f_p.size(0), f_p.size(1), -1)

        fs_p = []
        for i in range(self.num_patch ** 2):
            f_p_i = f_p[:, :, i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            fs_p.append(f_p_i)
        fs_p = torch.stack(fs_p, dim=-1)

        f_g = self.bottleneck(f_g)

        return (f_g, fs_p)

    def fss_features(self, feature_maps):
        attention_maps = self.cfi(feature_maps)
        feature_matrix, feature_matrix_hat, attention_maps_hat = self.cdal(feature_maps, attention_maps)

        feature_matrix = self.dropout(feature_matrix)
        feature_matrix_hat = self.dropout(feature_matrix_hat)
        p = self.fc1(feature_matrix * 100.)
        p_hat = self.fc1(feature_matrix_hat * 100.)
        return p, p_hat, attention_maps, attention_maps_hat

    def att_feat(self, feature_maps, attention_maps):
        feature_matrix = torch.einsum('imjk,injk->imn', attention_maps, feature_maps)

        feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)

        return feature_matrix

    def forward(self, x, label=None):
        res1, res4 = self.backbone(x)
        feature = self.forward_features(res4)
        y_pred_raw, y_pred_aux, attention_maps, attention_maps_hat = self.fss_features(res1)

        aug, index = self.augment(res1, attention_maps, attention_maps_hat)
        y_aug_raw, y_aug_aux, attention_maps_aug, attention_maps_hat_aug = self.fss_features(aug)

        feature_matrix = self.att_feat(res1, attention_maps)
        feature_matrix_aug = self.att_feat(aug, attention_maps_hat_aug)
        one_hot = F.one_hot(index, self.M)
        match_loss = torch.mean(torch.norm(feature_matrix - feature_matrix_aug, dim = -1) * (torch.ones_like(one_hot) - one_hot))

        (f_g, _) = feature

        x = self.dropout(f_g)
        y = self.fc2(x)

        if self.is_feat:
            return y, y_pred_raw, y_pred_aux, y_aug_raw, y_aug_aux, match_loss, feature

        return y



class MPSLClassifier(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 num_patch=3,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='bnneck',
                 pretrained=False,
                 M=64,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_patch = num_patch
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck

        self.xfm = DWTForward(J=1, wave='db1', mode='zero')
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.backbone = timm.create_model(encoder, features_only=True, out_indices=(1, 4), pretrained=pretrained)
        self.encoder_channels = self.backbone.feature_info.channels()

        self.num_features_low = self.encoder_channels[0]
        self.num_features_high = self.encoder_channels[1]

        # CDAL
        self.cdal = CDAL(self.num_features_low, self.M)
        self.cfi = CFI(self.num_features_low, self.M)
        self.augment = CAA()
        self.fc1 = nn.Linear(self.M * self.num_features_low, self.num_classes, bias=False)
        self.fc1.apply(weights_init_classifier)

        self.pool = AdaptiveAvgPool2dCustom((1, 1))
        # self.part = AdaptiveAvgPool2dCustom((self.num_patch, self.num_patch))
        self.part = []
        for i in range(self.num_patch, 8, 2):
            self.part.append(AdaptiveAvgPool2dCustom((i, i)))

        self.dropout = nn.Dropout(drop_rate)

        self.bottleneck = nn.BatchNorm1d(self.num_features_high)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc2 = nn.Linear(self.num_features_high, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)
            
        for pat in range(self.num_patch, 8, 2):
            for i in range(pat ** 2):
                name = 'bnneck' + str(pat) + '_' + str(i)
                setattr(self, name, nn.BatchNorm1d(self.num_features_high))
                getattr(self, name).bias.requires_grad_(False)
                getattr(self, name).apply(weights_init_kaiming)


    def forward_features(self, featuremap):
        f_g = self.pool(featuremap).flatten(1)
        # f_g_freq = self.pool(dct_2d(featuremap, 'ortho')).flatten(1)

        _, fh = self.xfm(featuremap)
        fh = fh[0] # [B, C, 3, H/2, W/2]
        freq_p_all = []
        for i in range(3):
            # [B, C, 3, H/2, W/2] -> [B, C, H/2 * W/2]
            freq_p = fh[:, :, i].view(fh.size(0), fh.size(1), -1)
            freq_p_all.append(freq_p)

        f_p = [pool(featuremap) for pool in self.part]
        f_p = [f_p_i.view(f_p_i.size(0), f_p_i.size(1), -1) for f_p_i in f_p]

        fs_p_all = []
        now = 0
        for pat in range(self.num_patch, 8, 2):
            fs_p = []
            for i in range(pat ** 2):
                f_p_i = f_p[now][:, :, i]
                f_p_i = getattr(self, 'bnneck' + str(pat) + '_' + str(i))(f_p_i)
                fs_p.append(f_p_i)
            now += 1
            fs_p = torch.stack(fs_p, dim=-1)
            fs_p_all.append(fs_p)

        f_g = self.bottleneck(f_g)
        # f_g_freq = self.bottleneck_freq(f_g_freq)

        return (f_g, (freq_p_all, fs_p_all))
        # return (f_g, (f_g_freq, fs_p_all))
    
    def fss_features(self, feature_maps):
        attention_maps = self.cfi(feature_maps)
        feature_matrix, feature_matrix_hat, attention_maps_hat = self.cdal(feature_maps, attention_maps)

        feature_matrix = self.dropout(feature_matrix)
        feature_matrix_hat = self.dropout(feature_matrix_hat)
        p = self.fc1(feature_matrix * 100.)
        p_hat = self.fc1(feature_matrix_hat * 100.)
        return p, p_hat, attention_maps, attention_maps_hat

    def att_feat(self, feature_maps, attention_maps):
        feature_matrix = torch.einsum('imjk,injk->imn', attention_maps, feature_maps)  # [B, M, C_F]
        feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)

        return feature_matrix

    def forward(self, x, label=None):
        res1, res4 = self.backbone(x)
        feature = self.forward_features(res4)
        y_pred_raw, y_pred_aux, attention_maps, attention_maps_hat = self.fss_features(res1)
        
        aug, index = self.augment(res1, attention_maps, attention_maps_hat)
        y_aug_raw, y_aug_aux, attention_maps_aug, attention_maps_hat_aug = self.fss_features(aug)
        
        feature_matrix = self.att_feat(res1, attention_maps)
        feature_matrix_aug = self.att_feat(aug, attention_maps_aug)
        one_hot = F.one_hot(index, self.M)
        match_loss = torch.mean(torch.norm(feature_matrix - feature_matrix_aug, dim = -1) * (torch.ones_like(one_hot) - one_hot))
        
        (f_g, _) = feature

        x = self.dropout(f_g)
        y = self.fc2(x)

        if self.is_feat:
            return y, y_pred_raw, y_pred_aux, y_aug_raw, y_aug_aux, match_loss, feature

        return y