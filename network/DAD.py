import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .modules import FeatureFusionBlock, RF, BasicConv2d, Overlapped_Window_Cross_Level_Semantic_Guidance
from .backbone.resnet import resnet50
from .backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from .backbone.pvtv2 import pvt_v2_b2, pvt_v2_b4
from .backbone.swin import SwinTransformer
from .backbone.vgg import VGG
from torchvision import models
from .backbone.resnext101_regular import ResNeXt101
import warnings
warnings.filterwarnings("ignore")

class DAD(nn.Module):
    def __init__(self, method, group, win_size, backbone_name='v2_b2', channel=64):
        super(DAD, self).__init__()

        self.backbone_name = backbone_name
        self.method = method
        self.backbone, self.encode_channels = self.get_backbone(backbone_name)

        self.conv_layers = nn.ModuleList([
            BasicConv2d(self.encode_channels[i], channel, 3, 1, 1) for i in range(len(self.encode_channels))
        ])

        self.path_blocks = nn.ModuleList([
            FeatureFusionBlock(channel) for _ in range(len(self.encode_channels))
        ])

        self.RF_blocks = nn.ModuleList([
            RF(channel, channel) for _ in range(len(self.encode_channels))
        ])

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

        self.ESG_fore = Overlapped_Window_Cross_Level_Semantic_Guidance(
            base_channel=channel, high_channel=channel, num_heads=8,
            window_sizes = win_size, dropout=0.1, residual=True, group_size=group
        )
        self.ESG_back = Overlapped_Window_Cross_Level_Semantic_Guidance(
            base_channel=channel, high_channel=channel, num_heads=8,
            window_sizes = win_size, dropout=0.1, residual=True, group_size=group
        )
        self.refined_conv = nn.Sequential(
            nn.Conv2d(2* channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def get_backbone(self, backbone_name):
        if backbone_name == 'resnet':
            backbone = timm.create_model(model_name="resnet50", pretrained=False, in_chans=3, features_only=True)
            path = './pretrained_model/resnet50_b1k-532a802a.pth'
            save_model = torch.load(path)
            model_dict = backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            encode_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'res2net':
            backbone = res2net50_v1b_26w_4s(pretrained=True)
            encode_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'vgg':
            backbone = VGG()
            encode_channels = [128, 256, 512, 512]
        elif backbone_name == 'v2_b2':
            backbone = pvt_v2_b2()
            path = './pretrained_model/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            encode_channels = [64, 128, 320, 512]
        elif backbone_name == 'v2_b4':
            backbone = pvt_v2_b4()
            path = './pretrained_model/pvt_v2_b4.pth'
            save_model = torch.load(path)
            model_dict = backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            encode_channels = [64, 128, 320, 512]
        elif backbone_name =='resnext101':
            backbone = ResNeXt101(backbone_path='./pretrained_model/resnext_101_32x4d.pth')
            encode_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'swin':
            backbone = SwinTransformer(
                img_size=384, embed_dim=128, depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32], window_size=12
            )
            pretrained_dict = torch.load('./pretrained_model/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
            backbone.load_state_dict(pretrained_dict)
            encode_channels = [128, 256, 512, 1024]
        else:
            raise ValueError(f"Unknown backbone name: {backbone_name}")

        return backbone, encode_channels



    def calculate_dynamic_weights(self, features, mask):
        mask = F.interpolate(mask, size=features.size()[2:], mode='bilinear', align_corners=True)
        mask = mask > 0.5
        intensity = torch.sum(features, dim=1, keepdim=True)
        intensity = intensity / torch.max(intensity)
        weights = intensity * mask.float()
        return weights

    def calculate_prototype_with_weights(self, features, mask):
        weights = self.calculate_dynamic_weights(features, mask)
        masked_features = features * weights
        prototype = masked_features.sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + 1e-8)
        return prototype

    def compute_similarity(self, features, prototype, method='cosine'):
        if method == 'cosine':
            similarity = F.cosine_similarity(features, prototype.unsqueeze(-1).unsqueeze(-1), dim=1)
            similarity = similarity.unsqueeze(1)
        elif method == 'euclidean':
            diff = features - prototype.unsqueeze(-1).unsqueeze(-1)
            similarity = -torch.norm(diff, dim=1, keepdim=True)
        elif method == 'manhattan':
            diff = features - prototype.unsqueeze(-1).unsqueeze(-1)
            similarity = -torch.sum(torch.abs(diff), dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        return similarity

    def forward(self, x):
        size = x.size()[2:]
        if self.backbone_name == 'swin':
            _, layer4, layer3, layer2, layer1 = self.backbone(x)
        elif self.backbone_name == 'v2_b2' or self.backbone_name == 'v2_b4':
            layer1, layer2, layer3, layer4 = self.backbone(x)
        elif self.backbone_name == 'vgg':
            _, layer1, layer2, layer3, layer4 = self.backbone(x)

        else:
            _, layer1, layer2, layer3, layer4 = self.backbone(x)


        # Apply convolutions
        layer1 = self.conv_layers[0](layer1)
        layer2 = self.conv_layers[1](layer2)
        layer3 = self.conv_layers[2](layer3)
        layer4 = self.conv_layers[3](layer4)

        # Apply RF blocks
        layer4 = self.RF_blocks[3](layer4)
        layer3 = self.RF_blocks[2](layer3)
        layer2 = self.RF_blocks[1](layer2)
        layer1 = self.RF_blocks[0](layer1)

        # Path through feature fusion blocks
        high_level = self.path_blocks[3](layer4)
        high_level = self.path_blocks[2](high_level, layer3)
        Guide_Map = self.out_conv(high_level)

        low_level = self.path_blocks[1](high_level, layer2)
        low_level = self.path_blocks[0](low_level, layer1)

        fore_prototype = self.calculate_prototype_with_weights(high_level, self.sigmoid(Guide_Map))
        back_prototype = self.calculate_prototype_with_weights(high_level, self.sigmoid(1 - Guide_Map))

        # Compute similarities
        fore_sim = self.compute_similarity(low_level, fore_prototype, method=self.method)
        back_sim = self.compute_similarity(low_level, back_prototype, method=self.method)

        _, fore_low_level = self.ESG_fore(high_level, fore_sim * low_level)
        _, back_low_level = self.ESG_back(high_level, back_sim * low_level)

        low_level = torch.cat((fore_low_level,back_low_level), dim=1)

        Refined_Map = self.refined_conv(low_level)

        # print(Guide_Map.shape, Refined_Map.shape)

        return F.interpolate(Guide_Map, size, mode='bilinear', align_corners=True),\
        F.interpolate(Refined_Map, size, mode='bilinear', align_corners=True)


