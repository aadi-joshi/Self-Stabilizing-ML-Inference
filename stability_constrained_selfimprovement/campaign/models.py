"""
Extended architecture zoo for the FTR phase-transition campaign.

Families (relative to the original 8-architecture zoo used in the Research
Square preprint):
  - CNN width sweep       (existing, extended)
  - CNN depth sweep       (existing, extended)
  - CNN no-BatchNorm      (existing, extended to more scales)
  - ResNet (2 blocks/layer, existing widths + new widths)
  - ResNet-Lite (1 block/layer) -- fills the curvature gap between the CNN
    family (tr(H) ~ 70-380) and the original ResNet18_W8 (tr(H) ~ 3254)
  - ResNet-Lite-NoBN      -- BN ablation at ResNet scale
  - Plain MLP             -- zero spatial inductive bias
  - ViT-tiny              -- patch-based transformer, no convolutional locality prior
  - MLP-Mixer-tiny        -- token/channel-mixing MLP, different inductive bias again

All factories take (num_classes) and return an nn.Module with a `.features(x)`
method (used nowhere in the campaign but kept for parity with the original
zoo) and a `.forward(x)` method returning logits.
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CNN family (ported from run_neurips_breakthrough.ScalableCNN)
# ======================================================================
class ScalableCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, base_width=32, num_conv=3, use_bn=True):
        super().__init__()
        layers = []
        widths = [in_channels]
        for i in range(num_conv):
            out_w = base_width * (2 ** min(i, 1))
            layers.append(nn.Conv2d(widths[-1], out_w, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_w))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            widths.append(out_w)
        self.features_conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            feat = self.features_conv(dummy)
            self.feat_dim = feat.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self.feat_dim, min(128, self.feat_dim))
        self.fc2 = nn.Linear(min(128, self.feat_dim), num_classes)
        self.dropout = nn.Dropout(0.25)

    def features(self, x):
        x = self.features_conv(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.dropout(self.features(x)))


# ======================================================================
# ResNet family (ported from run_neurips_breakthrough.ResNetCL/BasicBlock)
# num_blocks_per_layer=1 gives a "Lite" ResNet (4 blocks total instead of 8)
# ======================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            sc = [nn.Conv2d(in_planes, planes, 1, stride=stride, bias=not use_bn)]
            if use_bn:
                sc.append(nn.BatchNorm2d(planes))
            self.shortcut = nn.Sequential(*sc)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetCL(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, base_width=16, num_blocks_per_layer=2, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(in_channels, base_width, 3, stride=1, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_width) if use_bn else nn.Identity()
        w = base_width
        self.layer1 = self._make_layer(w, num_blocks_per_layer, stride=1)
        self.layer2 = self._make_layer(w * 2, num_blocks_per_layer, stride=2)
        self.layer3 = self._make_layer(w * 4, num_blocks_per_layer, stride=2)
        self.layer4 = self._make_layer(w * 8, num_blocks_per_layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w * 8, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, use_bn=self.use_bn))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = l(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def forward(self, x):
        return self.fc(self.features(x))


# ======================================================================
# Plain MLP -- zero spatial inductive bias
# ======================================================================
class PlainMLP(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, hidden=128, depth=2):
        super().__init__()
        in_dim = in_channels * 32 * 32
        dims = [in_dim] + [hidden] * depth
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.25)

    def features(self, x):
        x = x.view(x.size(0), -1)
        return self.body(x)

    def forward(self, x):
        return self.head(self.dropout(self.features(x)))


# ======================================================================
# ViT-tiny -- patch embedding + standard pre-LN transformer blocks
# ======================================================================
class ViTTiny(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, image_size=32, patch_size=4,
                 dim=64, depth=2, heads=2, mlp_ratio=2.0):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'mlp': nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(dim * mlp_ratio), dim),
                ),
            }) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)  # B,C,H/p,W/p,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * p * p)
        return x

    def features(self, x):
        x = self._patchify(x)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            h = blk['ln1'](x)
            attn_out, _ = blk['attn'](h, h, h, need_weights=False)
            x = x + attn_out
            x = x + blk['mlp'](blk['ln2'](x))
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        return self.head(self.features(x))


# ======================================================================
# MLP-Mixer-tiny -- token-mixing + channel-mixing MLP blocks
# ======================================================================
class MixerBlock(nn.Module):
    def __init__(self, n_tokens, dim, tokens_hidden, channels_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, tokens_hidden), nn.GELU(), nn.Linear(tokens_hidden, n_tokens)
        )
        self.ln2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channels_hidden), nn.GELU(), nn.Linear(channels_hidden, dim)
        )

    def forward(self, x):
        # x: B, n_tokens, dim
        y = self.ln1(x).transpose(1, 2)          # B, dim, n_tokens
        y = self.token_mlp(y).transpose(1, 2)     # B, n_tokens, dim
        x = x + y
        x = x + self.channel_mlp(self.ln2(x))
        return x


class MixerTiny(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, image_size=32, patch_size=4,
                 dim=64, depth=2, tokens_hidden=32, channels_hidden=128):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches, dim, tokens_hidden, channels_hidden) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * p * p)
        return x

    def features(self, x):
        x = self.patch_embed(self._patchify(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        return self.head(self.features(x))


# ======================================================================
# Architecture registry
# ======================================================================
def get_architecture_zoo():
    """Returns OrderedDict name -> {'factory': fn(num_classes), 'epochs': int,
    'group': str, 'family': str}. Params counted lazily below."""
    zoo = OrderedDict()

    def add(name, factory, epochs, group, family):
        zoo[name] = {'factory': factory, 'epochs': epochs, 'group': group, 'family': family}

    # ---- CNN width sweep (8 points, matches + extends original) ----
    for w in [8, 16, 24, 32, 48, 64, 96, 128]:
        add(f'CNN_W{w}', (lambda nc, _w=w: ScalableCNN(nc, 3, base_width=_w, num_conv=3, use_bn=True)),
            4, 'width', 'cnn')

    # ---- CNN depth sweep at W=32 ----
    for d in [2, 3, 4, 5]:
        add(f'CNN_D{d}_W32', (lambda nc, _d=d: ScalableCNN(nc, 3, base_width=32, num_conv=_d, use_bn=True)),
            4, 'depth', 'cnn')

    # ---- CNN no-BatchNorm at 3 scales ----
    for w in [8, 32, 64]:
        add(f'CNN_W{w}_NoBN', (lambda nc, _w=w: ScalableCNN(nc, 3, base_width=_w, num_conv=3, use_bn=False)),
            4, 'bn', 'cnn')

    # ---- ResNet (2 blocks/layer): original widths + 2 new ----
    for w in [8, 16, 24, 32]:
        add(f'ResNet18_W{w}', (lambda nc, _w=w: ResNetCL(nc, 3, base_width=_w, num_blocks_per_layer=2, use_bn=True)),
            4, 'resnet', 'resnet')

    # ---- ResNet-Lite (1 block/layer): fills curvature gap between CNN and ResNet18 ----
    for w in [8, 16, 24, 32]:
        add(f'ResNetLite_W{w}', (lambda nc, _w=w: ResNetCL(nc, 3, base_width=_w, num_blocks_per_layer=1, use_bn=True)),
            4, 'resnet_lite', 'resnet')

    # ---- ResNet-Lite no-BN (curvature/BN ablation at ResNet scale) ----
    add('ResNetLite_W8_NoBN', (lambda nc: ResNetCL(nc, 3, base_width=8, num_blocks_per_layer=1, use_bn=False)),
        4, 'resnet_bn', 'resnet')

    # ---- Plain MLP (zero spatial inductive bias) ----
    for h in [128, 256]:
        add(f'MLP_H{h}', (lambda nc, _h=h: PlainMLP(nc, 3, hidden=_h, depth=2)), 4, 'mlp', 'mlp')

    # ---- ViT-tiny (patch-based, no convolutional locality prior) ----
    add('ViT_Tiny', (lambda nc: ViTTiny(nc, 3, dim=64, depth=2, heads=2, mlp_ratio=2.0)), 5, 'vit', 'vit')
    add('ViT_Small', (lambda nc: ViTTiny(nc, 3, dim=96, depth=4, heads=3, mlp_ratio=2.0)), 5, 'vit', 'vit')

    # ---- MLP-Mixer-tiny (token/channel mixing, different inductive bias again) ----
    add('Mixer_Tiny', (lambda nc: MixerTiny(nc, 3, dim=64, depth=2, tokens_hidden=32, channels_hidden=128)),
        5, 'mixer', 'mixer')
    add('Mixer_Small', (lambda nc: MixerTiny(nc, 3, dim=96, depth=3, tokens_hidden=48, channels_hidden=192)),
        5, 'mixer', 'mixer')

    for name, cfg in zoo.items():
        m = cfg['factory'](2)
        cfg['n_params'] = sum(p.numel() for p in m.parameters())
        del m
    return zoo


if __name__ == '__main__':
    zoo = get_architecture_zoo()
    print(f"{len(zoo)} architectures:")
    for name, cfg in zoo.items():
        print(f"  {name:<22s} {cfg['n_params']:>10,} params  [{cfg['family']}/{cfg['group']}]")
