import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Res2Net_v1b import *
from network import resnet
from network.pvtv2 import pvt_v2_b2
from torchvision import models
from network.resnext101_regular import ResNeXt101


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DGM(nn.Module):
    def __init__(self):
        super(DGM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        x2_cat = x2
        for i in range(C-1):
            x2_cat = torch.cat((x2,x2_cat),dim=1)
        x1_view = x1.view(B,C,-1)
        x2_view = x2_cat.view(B,C,H*W).permute(0,2,1)
        energy = torch.bmm(x1_view, x2_view)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        relation = self.softmax(energy_new)
        x1_new = x1.view(B, C, -1)
        out = torch.bmm(relation,x1_new)
        out = out.view(B, C, H, W)
        out = self.gamma * out +x1

        return out

class DAE(nn.Module):
    def __init__(self, in_dim_low, nclasses):
        super(DAE, self).__init__()
        self.seg_map = nn.Sigmoid()
        self.cgm = DGM()
        self.classes = nclasses
        self.low_channel = in_dim_low
        self.out = nn.Sequential(
            nn.Conv2d(self.low_channel, self.low_channel, 3, 1, 1),
            nn.Conv2d(self.low_channel, self.classes, 3, 1, 1))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.low_channel)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.low_channel)
        self.relu2 = nn.ReLU()

    def forward(self, low_feature, x):
        low_feature = F.upsample(low_feature, x.size()[2:], mode='bilinear', align_corners=True)
        low_feature = self.cgm(low_feature,x)
        seg_prob = self.seg_map(x)
        foreground = low_feature * seg_prob
        background = low_feature * (1 - seg_prob)
        refine1 = self.alpha * foreground
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)
        refine2 = self.beta * background
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        fusion = refine1 - refine2
        output_map = self.out(fusion)

        return output_map

class Guided_Map_Generator_vgg(nn.Module):
    def  __init__(self,nclass, layer_name):
        super(Guided_Map_Generator_vgg, self).__init__()
        self.head = DARNetHead(256)
        self.fem = FEM(512,256)
        self.decode = Decoder_coarse(nclass, 64, layer_name)

    def forward(self, x1 ,x2):
        x1_fem = self.fem(x1)
        x1_head = self.head(x1_fem)
        out = self.decode(x1_head, x2)

        return out

class Guided_Map_Generator_pvt(nn.Module):
    def  __init__(self,nclass, layer_name):
        super(Guided_Map_Generator_pvt, self).__init__()
        self.head = DARNetHead(256)
        self.fem = FEM(512,256)
        self.decode = Decoder_coarse(nclass, 64, layer_name)

    def forward(self, x1 ,x2):
        x1_fem = self.fem(x1)
        x1_head = self.head(x1_fem)
        out = self.decode(x1_head, x2)

        return out

class Decoder_coarse(nn.Module):
    def __init__(self, num_classes,in_ch,layer_name):
        super(Decoder_coarse, self).__init__()
        if layer_name =='layer1':
            low_level_inplanes = 64
        elif layer_name =='layer2':
            low_level_inplanes = 256
        elif layer_name =='layer3':
            low_level_inplanes = 512
        elif layer_name =='layer4':
            low_level_inplanes = 1024

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)  #256->48 1*1
        self.bn1 = nn.BatchNorm2d(48) #nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(in_ch + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  #插值上采样
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_coarse_v2(nn.Module):
    def __init__(self, num_classes,in_ch):
        super(Decoder_coarse_v2, self).__init__()

        self.last_conv = nn.Sequential(nn.Conv2d(in_ch, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x):
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Guided_Map_Generator_res(nn.Module):
    def  __init__(self,nclass,layer_name):
        super(Guided_Map_Generator_res, self).__init__()
        self.head = DARNetHead(256)
        self.fem = FEM(2048,256)
        if layer_name =='layer5':
            self.decode = Decoder_coarse_v2(nclass, 64)
        else:
            self.decode = Decoder_coarse(nclass, 64, layer_name)

    def forward(self, x1 ,x2):
        x1_fem = self.fem(x1)
        x1_head = self.head(x1_fem)
        out = self.decode(x1_head,x2)

        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
            解释  :
                bmm : 实现batch的叉乘
                Parameter：绑定在层里，所以是可以更新的
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())



class DARNetHead(nn.Module):
    def __init__(self, in_channels):
        super(DARNetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        sa_feat = self.conv5a(x)
        sa_feat = self.sa(sa_feat)
        sa_feat = self.conv51(sa_feat)

        sc_feat = self.conv5c(x)
        sc_feat = self.sc(sc_feat)
        sc_feat = self.conv52(sc_feat)

        # 两个注意力是相加的
        feat_sum = sa_feat + sc_feat

        output = self.dropout(feat_sum)
        return output




class FEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FEM, self).__init__()
        self.relu = nn.ReLU(True)
        out_channel_sum = out_channel * 3

        self.branch0 = nn.Sequential(
            _ConvBnReLU(in_channel, out_channel, 1, 1, 0, 1),
        )
        self.branch1 = nn.Sequential(
            _ConvBnReLU(in_channel, out_channel, 1, 1, 0, 1),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 4, 4),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 8, 8),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 16, 16),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 32, 32)
        )
        self.branch2 = nn.Sequential(
            _ConvBnReLU(in_channel, out_channel, 1, 1, 0, 1),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 2, 2),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 4, 4),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 8, 8),
            _ConvBnReLU(out_channel, out_channel, 3, 1, 16, 16)
        )
        self.conv_cat =_ConvBnReLU(out_channel_sum, out_channel, 3, 1, 1, 1)
        self.conv_res = _ConvBnReLU(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class DAD_res2net50(nn.Module):
    def __init__(self, nclasses=1):
        super(DAD_res2net50, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.coarse_out = Guided_Map_Generator_res(nclasses,layer_name='layer1')
        self.layer2_process = nn.Sequential(
                                            _ConvBnReLU(1024, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer1_process = nn.Sequential(_ConvBnReLU(512, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer0_process = nn.Sequential(_ConvBnReLU(256, 128, 3, 2, 1, 1),
                                            FEM(128, 32))

        self.out_1 = DAE(96,nclasses)
        self.out_2 = DAE(96, nclasses)

    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        layer0 = self.resnet.layer1(x)  # bs, 256, 88, 88
        layer1 = self.resnet.layer2(layer0)  # bs, 512, 44, 44
        layer2 = self.resnet.layer3(layer1)  # bs, 1024, 22, 22
        layer3 = self.resnet.layer4(layer2)
        coarse_out = self.coarse_out(layer3,layer0)
        layer2_process = self.layer2_process(layer2)
        layer2_up = F.upsample(layer2_process, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1_process = self.layer1_process(layer1)
        layer0_process = self.layer0_process(layer0)
        layer_0_1_2 = torch.cat((layer0_process,layer1_process,layer2_up),dim = 1)
        result_1 = self.out_1(layer_0_1_2,coarse_out)
        result_2 = self.out_2(layer_0_1_2, result_1)
        return F.upsample(coarse_out, size, mode='bilinear', align_corners=True), F.upsample(result_1, size, mode='bilinear', align_corners=True), \
               F.upsample(result_2, size, mode='bilinear', align_corners=True)


class DAD_vgg(nn.Module):
    def __init__(self, nclasses=1):
        super(DAD_vgg, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64
        self.coarse_out = Guided_Map_Generator_vgg(nclasses, layer_name='layer1')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer4_process = nn.Sequential(
            _ConvBnReLU(512, 128, 1, 1, 0, 1),
            FEM(128, 32))
        self.layer3_process = nn.Sequential(_ConvBnReLU(256, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer2_process = nn.Sequential(_ConvBnReLU(128, 128, 3, 2, 1, 1),
                                            FEM(128, 32))

        self.out_1 = DAE(96,nclasses)
        self.out_2 = DAE(96, nclasses)

    def forward(self, x):
        size = x.size()[2:]
        layer1 = self.inc(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)
        layer1 = self.maxpool(layer1)
        layer1 = self.inc(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)
        layer1 = self.maxpool(layer1)
        # print(layer1.size(),layer2.size(),layer3.size(),layer4.size(),layer5.size())
        coarse_out = self.coarse_out(layer5, layer1)
        layer4_process = self.layer4_process(layer4)
        layer4_up = F.upsample(layer4_process, layer3.size()[2:], mode='bilinear', align_corners=True)
        layer3_process = self.layer3_process(layer3)
        layer2_process = self.layer2_process(layer2)
        layer_2_3_4 = torch.cat((layer2_process, layer3_process, layer4_up), dim=1)
        result_1 = self.out_1(layer_2_3_4,coarse_out)
        result_2 = self.out_2(layer_2_3_4, result_1)
        return F.upsample(coarse_out, size, mode='bilinear', align_corners=True), F.upsample(result_1, size, mode='bilinear', align_corners=True), \
               F.upsample(result_2, size, mode='bilinear', align_corners=True)


class DAD_resnet50(nn.Module):
    def __init__(self, nclasses=1):
        super(DAD_resnet50, self).__init__()
        self.resnet = resnet.resnet50(backbone_path='./pretrain_models/resnet/resnet50-19c8e357.pth')  # 孪生网络，共享权重
        self.coarse_out = Guided_Map_Generator_res(nclasses,layer_name='layer1')
        self.layer2_process = nn.Sequential(
                                            _ConvBnReLU(1024, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer1_process = nn.Sequential(_ConvBnReLU(512, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer0_process = nn.Sequential(_ConvBnReLU(256, 128, 3, 2, 1, 1),
                                            FEM(128, 32))

        self.out_1 = DAE(96,nclasses)
        self.out_2 = DAE(96, nclasses)

    def forward(self, x):
        size = x.size()[2:]
        x, layer0, layer1, layer2, layer3 = self.resnet(x)
        coarse_out = self.coarse_out(layer3,layer0)
        layer2_process = self.layer2_process(layer2)
        layer2_up = F.upsample(layer2_process, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1_process = self.layer1_process(layer1)
        layer0_process = self.layer0_process(layer0)
        layer_0_1_2 = torch.cat((layer0_process,layer1_process,layer2_up),dim = 1)
        result_1 = self.out_1(layer_0_1_2,coarse_out)
        result_2 = self.out_2(layer_0_1_2, result_1)
        return F.upsample(coarse_out, size, mode='bilinear', align_corners=True), F.upsample(result_1, size, mode='bilinear', align_corners=True), \
               F.upsample(result_2, size, mode='bilinear', align_corners=True)


class DAD_resnext101(nn.Module):
    def __init__(self, nclasses=1):
        super(DAD_resnext101, self).__init__()
        self.resnet = ResNeXt101(backbone_path='/home/lijiepan/Difference_COD/pretrain_models/resnext_101_32x4d.pth')
        self.coarse_out = Guided_Map_Generator_res(nclasses,layer_name='layer2')
        self.layer2_process = nn.Sequential(
                                            _ConvBnReLU(1024, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer1_process = nn.Sequential(_ConvBnReLU(512, 128, 1, 1, 0, 1),
                                            FEM(128, 32))
        self.layer0_process = nn.Sequential(_ConvBnReLU(256, 128, 3, 2, 1, 1),
                                            FEM(128, 32))

        self.out_1 = DAE(96,nclasses)
        self.out_2 = DAE(96, nclasses)

    def forward(self, x):
        size = x.size()[2:]
        x, layer0, layer1, layer2, layer3 = self.resnet(x)
        coarse_out = self.coarse_out(layer3,layer0)
        layer2_process = self.layer2_process(layer2)
        layer2_up = F.upsample(layer2_process, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1_process = self.layer1_process(layer1)
        layer0_process = self.layer0_process(layer0)
        layer_0_1_2 = torch.cat((layer0_process,layer1_process,layer2_up),dim = 1)
        result_1 = self.out_1(layer_0_1_2,coarse_out)
        result_2 = self.out_2(layer_0_1_2, result_1)
        return F.upsample(coarse_out, size, mode='bilinear', align_corners=True), F.upsample(result_1, size, mode='bilinear', align_corners=True), \
               F.upsample(result_2, size, mode='bilinear', align_corners=True)


class DAD_pvt_v2(nn.Module):
    def __init__(self, nclasses=1):
        super(DAD_pvt_v2, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrain_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.coarse_out = Guided_Map_Generator_pvt(nclasses,layer_name='layer1')
        self.layer2_process = nn.Sequential(
                                            _ConvBnReLU(320, 64, 1, 1, 0, 1),
                                            FEM(64, 32))
        self.layer1_process = nn.Sequential(_ConvBnReLU(128, 64, 1, 1, 0, 1),
                                            FEM(64, 32))
        self.layer0_process = nn.Sequential(_ConvBnReLU(64, 64, 3, 2, 1, 1),
                                            FEM(64, 32))

        self.out_1 = DAE(96,nclasses)
        self.out_2 = DAE(96, nclasses)

    def forward(self, x):
        size = x.size()[2:]
        pvt = self.backbone(x)
        layer0 = pvt[0]
        layer1 = pvt[1]
        layer2 = pvt[2]
        layer3 = pvt[3]
        coarse_out = self.coarse_out(layer3,layer0)
        layer2_process = self.layer2_process(layer2)
        layer2_up = F.upsample(layer2_process, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1_process = self.layer1_process(layer1)
        layer0_process = self.layer0_process(layer0)
        layer_0_1_2 = torch.cat((layer0_process,layer1_process,layer2_up),dim = 1)
        result_1 = self.out_1(layer_0_1_2,coarse_out)
        result_2 = self.out_2(layer_0_1_2, result_1)
        return F.upsample(coarse_out, size, mode='bilinear', align_corners=True), F.upsample(result_1, size, mode='bilinear', align_corners=True), \
               F.upsample(result_2, size, mode='bilinear', align_corners=True)



