"""Architecture reconstruction from Sec. 2 and Figs. 1–4."""
import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def upsample(feature, size):
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


class SCAF(nn.Module):
    def __init__(self, channels, channel_dilation, spatial_dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.channel_context = nn.Conv1d(1, 1, 3, padding=channel_dilation,
                                         dilation=channel_dilation)
        self.spatial_context = nn.Conv2d(2, 1, 3, padding=spatial_dilation,
                                         dilation=spatial_dilation)

    def channel_branch(self, descriptor):
        # A global C×1×1 descriptor has no spatial neighborhood. This
        # reconstruction applies the dilated kernel along the channel index.
        projected = self.pointwise(self.depthwise(descriptor))
        sequence = projected.squeeze(-1).transpose(1, 2)
        context = self.channel_context(sequence).transpose(1, 2).unsqueeze(-1)
        return descriptor + context

    def forward(self, feature):
        average = feature.mean(dim=(2, 3), keepdim=True)
        maximum = feature.amax(dim=(2, 3), keepdim=True)
        channel_weights = torch.sigmoid(self.channel_branch(average) + self.channel_branch(maximum))
        feature = feature * channel_weights
        descriptors = torch.cat((feature.mean(dim=1, keepdim=True),
                                 feature.amax(dim=1, keepdim=True)), dim=1)
        return feature * torch.sigmoid(self.spatial_context(descriptors))


class FeatureEnhancement(nn.Module):
    """Fuse adjacent finer, current and enhanced coarser pyramid levels."""

    def __init__(self, finer_channels, channels, coarser_channels, reduction):
        super().__init__()
        self.finer_projection = nn.Conv2d(finer_channels, channels, 1)
        self.coarser_projection = nn.Conv2d(coarser_channels, channels, 1)
        hidden = max(4, channels // reduction)
        self.gate = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.refine = conv(channels, channels)

    def forward(self, finer, current, coarser):
        size = current.shape[-2:]
        max_down = F.max_pool2d(finer, 3, stride=2, padding=1)
        avg_down = F.avg_pool2d(finer, 3, stride=2, padding=1, count_include_pad=False)
        finer = self.finer_projection((max_down + avg_down) * 0.5)
        coarser = upsample(self.coarser_projection(coarser), size)
        weights = self.gate(current.mean(dim=(2, 3))).softmax(dim=1)
        branches = torch.stack((finer, current, coarser), dim=1)
        return self.refine((branches * weights[:, :, None, None, None]).sum(dim=1))


def db_head(channels):
    hidden = max(4, channels // 4)
    return nn.Sequential(
        conv(channels, hidden),
        nn.ConvTranspose2d(hidden, hidden, 2, 2, bias=False),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(hidden, 1, 2, 2),
    )


VARIANTS = {
    "baseline": (False, False),
    "feature": (False, True),
    "scaf": (True, False),
    "full": (True, True),
}


class AttentionDBNet(nn.Module):
    def __init__(self, variant, stem_channels, pyramid_channels, neck_channels,
                 channel_dilation, spatial_dilation, reduction, db_k):
        super().__init__()
        use_scaf, use_feature = VARIANTS[variant]
        self.db_k = db_k
        self.stem = nn.Sequential(conv(3, stem_channels, stride=2),
                                  conv(stem_channels, stem_channels))
        self.scaf = SCAF(stem_channels, channel_dilation, spatial_dilation) if use_scaf else nn.Identity()
        channels = [stem_channels, *pyramid_channels]
        self.down = nn.ModuleList(
            conv(finer, current, stride=2) for finer, current in zip(channels, channels[1:])
        )
        self.fusion = nn.ModuleList(
            FeatureEnhancement(finer, current, coarser, reduction)
            for finer, current, coarser in zip(channels, channels[1:], channels[2:])
        ) if use_feature else nn.ModuleList()
        self.lateral = nn.ModuleList(conv(c, neck_channels) for c in pyramid_channels)
        head_channels = neck_channels * len(pyramid_channels)
        self.probability = db_head(head_channels)
        self.threshold = db_head(head_channels)

    def forward(self, image):
        stem = self.stem(image)
        pyramid = [stem]
        feature = self.scaf(stem)
        for down in self.down:
            feature = down(feature)
            pyramid.append(feature)

        levels = pyramid[1:]
        for index in reversed(range(len(self.fusion))):
            levels[index] = self.fusion[index](pyramid[index], levels[index], levels[index + 1])

        size = levels[0].shape[-2:]
        combined = torch.cat([upsample(layer(level), size)
                              for layer, level in zip(self.lateral, levels)], dim=1)
        size = image.shape[-2:]
        probability_logits = self.probability(combined)[..., :size[0], :size[1]]
        threshold = self.threshold(combined)[..., :size[0], :size[1]].sigmoid()
        probability = probability_logits.sigmoid()
        binary = torch.sigmoid(self.db_k * (probability - threshold))
        return {"probability_logits": probability_logits, "probability": probability,
                "threshold": threshold, "binary": binary}
