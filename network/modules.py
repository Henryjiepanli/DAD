import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = F.interpolate(output, size=xs[1].size()[2:], mode="bilinear", align_corners=True)
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        return output


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

def window_unfold(x, window_sizes):
    B, C, H, W = x.shape
    windows = []
    for window_size in window_sizes:
        stride = window_size // 2
        uf = nn.Unfold((window_size, window_size), stride=stride)
        window = uf(x).view(B, C, window_size * window_size, -1).permute(0, 3, 2, 1).flatten(0, 1).contiguous()
        windows.append(window)
    return windows

def window_fold(windows, window_sizes, H, W):
    results = []
    for i, window_size in enumerate(window_sizes):
        stride = window_size // 2
        B_, N, C = windows[i].shape
        nums_h = (H - window_size) // stride + 1
        nums_w = (W - window_size) // stride + 1
        B = B_ // (nums_h * nums_w)
        f = nn.Fold((H, W), (window_size, window_size), stride=stride)
        x = windows[i].view(B, nums_h * nums_w, window_size * window_size, C).permute(0, 3, 2, 1).flatten(1, 2).contiguous()
        x = f(x)
        results.append(x)
    return results



class Overlapped_Window_Cross_Level_Semantic_Guidance(nn.Module):
    def __init__(self, base_channel, high_channel, num_heads, window_sizes, dropout, residual=True, group_size=1):
        super(Overlapped_Window_Cross_Level_Semantic_Guidance, self).__init__()
        self.base_channel = base_channel
        self.high_channel = high_channel
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.group_size = group_size
        self.conv = nn.Conv2d(high_channel, base_channel, 1)
        self.elu = nn.ELU()
        self.attn_dropout = nn.Dropout(p=dropout)
        self.proj_dropout = nn.Dropout(p=dropout)
        self.residual = residual

        # LayerNorm and Linear layers per group
        self.norms = nn.ModuleList([nn.LayerNorm(base_channel // group_size) for _ in range(group_size)])
        self.linears_q = nn.ModuleList([nn.Linear(base_channel // group_size, base_channel // group_size) for _ in range(group_size)])
        self.linears_k = nn.ModuleList([nn.Linear(base_channel // group_size, base_channel // group_size) for _ in range(group_size)])
        self.linears_v = nn.ModuleList([nn.Linear(base_channel // group_size, base_channel // group_size) for _ in range(group_size)])
        self.linears_out = nn.ModuleList([nn.Linear(base_channel // group_size, base_channel // group_size) for _ in range(group_size)])

        self.window_weights = nn.Parameter(torch.ones(len(window_sizes)))
        if self.residual:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, high_stage, low_stage, mask=None):
        high_stage = self.conv(high_stage)
        B, C = low_stage.size()[0], low_stage.size()[1]
        H, W = high_stage.size()[2], high_stage.size()[3]
        Scale_spatial = low_stage.size()[2] // high_stage.size()[2]

        # Split features into groups
        low_stage_groups = torch.split(low_stage, self.base_channel // self.group_size, dim=1)
        high_stage_groups = torch.split(high_stage, self.high_channel // self.group_size, dim=1)

        out_groups = []
        for g in range(self.group_size):

            # Unfold and process windows
            query_windows = window_unfold(low_stage_groups[g], [Scale_spatial * ws for ws in self.window_sizes])
            query_windows = [self.norms[g](q) for q in query_windows]
            query_windows = [self.linears_q[g](q).view(-1, (Scale_spatial * ws) ** 2, self.num_heads, self.base_channel // (self.group_size * self.num_heads)).transpose(-3, -2) for q, ws in zip(query_windows, self.window_sizes)]
            query_windows = [self.elu(q) + 1.0 for q in query_windows]

            feature_windows = window_unfold(high_stage_groups[g], self.window_sizes)
            feature_windows = [self.norms[g](f).view(-1, ws ** 2, self.base_channel // self.group_size).contiguous() for f, ws in zip(feature_windows, self.window_sizes)]
            key_windows = [self.linears_k[g](f).view(-1, ws ** 2, self.num_heads, self.base_channel // (self.group_size * self.num_heads)).transpose(-3, -2) for f, ws in zip(feature_windows, self.window_sizes)]
            key_windows = [self.elu(k) + 1.0 for k in key_windows]
            value_windows = [self.linears_v[g](f).view(-1, ws ** 2, self.num_heads, self.base_channel // (self.group_size * self.num_heads)).transpose(-3, -2) for f, ws in zip(feature_windows, self.window_sizes)]

            out_windows = []
            for query, key, value, ws, weight in zip(query_windows, key_windows, value_windows, self.window_sizes, self.window_weights):
                z = 1 / (query @ key.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
                kv = (key.transpose(-2, -1) * (ws ** -0.5)) @ (value * (ws ** -0.5))
                out = query @ kv * z
                out_windows.append(out * weight)  # Apply weighting

            out_windows = [self.proj_dropout(self.linears_out[g](out.transpose(1, 2).flatten(-2))) for out in out_windows]
            out_windows = window_fold(out_windows, [Scale_spatial * ws for ws in self.window_sizes], Scale_spatial * H, Scale_spatial * W)
            # sum up all window results for this group
            out_group = sum(out_windows) / len(out_windows)
            out_groups.append(out_group)

        # Concatenate all groups along the channel dimension
        out = torch.cat(out_groups, dim=1)
        if self.residual:
            final = self.alpha * out + (1 - self.alpha) * low_stage
            return out_groups, final
        else:
            return out_groups, out
