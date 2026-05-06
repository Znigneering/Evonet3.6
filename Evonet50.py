import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot
import numpy as np
import math


class EpisodeBatchNorm2d(nn.Module):
    """
    Episode-aware BatchNorm: always normalises from the current batch statistics
    (train and eval), so each episode is normalised independently. Drop-in
    replacement for nn.BatchNorm2d(C, affine=True) with no running buffers.
    """

    def __init__(self, num_features, eps=1e-5):
        super(EpisodeBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        C = x.size(1)
        x_flat = x.transpose(0, 1).contiguous().view(C, -1)
        mu = x_flat.mean(1).view(1, C, 1, 1)
        var = x_flat.var(1, unbiased=False).view(1, C, 1, 1)
        x_hat = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, C, 1, 1) * x_hat + self.bias.view(1, C, 1, 1)


class EvolvingBase(nn.Module):
    def __init__(self, density):
        super(EvolvingBase, self).__init__()
        self.density = density

    def init_mask(self):
        with torch.no_grad():
            self.mask.fill_(0)
            num_total = self.weight.numel()
            num_active = int(num_total * self.density)
            perm = torch.randperm(num_total)[:num_active]
            self.mask.view(-1)[perm] = 1
            self.weight.data *= self.mask

    def evolve(self, prune_threshold=0.01, growth_ratio=0.05):
        with torch.no_grad():
            active_weights = torch.abs(self.weight.data)
            mask_prune = (active_weights < prune_threshold) & (self.mask == 1)
            self.mask[mask_prune] = 0
            self.weight.data[mask_prune] = 0.0
            pruned_count = int(mask_prune.sum().item())

            grown_count = 0
            mask_grow = torch.zeros_like(self.mask).byte()

            if self.weight.grad is not None:
                grad_mag = torch.abs(self.weight.grad)
                grad_inactive = grad_mag * (1.0 - self.mask)

                inactive_count = int((self.mask == 0).sum().item())
                max_grow = int(self.weight.numel() * growth_ratio)
                k = min(max_grow, inactive_count)

                if k > 0:
                    top_grads, top_indices = torch.topk(grad_inactive.view(-1), k=k)
                    effective_indices = top_indices[top_grads > 1e-6]

                    if len(effective_indices) > 0:
                        self.mask.view(-1)[effective_indices] = 1
                        mask_grow.view(-1)[effective_indices] = 1
                        grown_count = len(effective_indices)

            changed_mask = mask_prune | mask_grow
            return pruned_count, grown_count, changed_mask


class EvolvingLinear(EvolvingBase):
    def __init__(self, in_features, out_features, density=0.2):
        super(EvolvingLinear, self).__init__(density)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer('mask', torch.ones(out_features, in_features))

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.init_mask()

    def forward(self, x):
        effective_weight = self.weight - (self.weight * (1.0 - self.mask)).detach()
        return F.linear(x, effective_weight, self.bias)


class EvolvingConv2d(EvolvingBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, density=0.5):
        super(EvolvingConv2d, self).__init__(density)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.register_buffer('mask', torch.ones(out_channels, in_channels, kernel_size, kernel_size))
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        self.init_mask()

    def forward(self, x):
        effective_weight = self.weight - (self.weight * (1.0 - self.mask)).detach()
        return F.conv2d(x, effective_weight, self.bias, self.stride, self.padding)


def conv3x3(in_planes, out_planes, stride=1, density=0.5):
    return EvolvingConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, density=density)

def conv1x1(in_planes, out_planes, stride=1, density=0.5):
    return EvolvingConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, density=density)


class EvolvingBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, density=0.5):
        super(EvolvingBottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes, density=density)
        self.bn1 = EpisodeBatchNorm2d(planes)
        self.conv2 = EvolvingConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, density=density)
        self.bn2 = EpisodeBatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, density=density)
        self.bn3 = EpisodeBatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride=stride, density=density),
                EpisodeBatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class EvolvingResNet50(nn.Module):
    def __init__(self, input_channel=3, density=0.5):
        super(EvolvingResNet50, self).__init__()
        self.in_planes = 64

        self.conv1 = EvolvingConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, density=density)
        self.bn1 = EpisodeBatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  3, stride=1, density=density)
        self.layer2 = self._make_layer(128, 4, stride=2, density=density)
        self.layer3 = self._make_layer(256, 6, stride=2, density=density)
        self.layer4 = self._make_layer(512, 3, stride=2, density=density)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes, num_blocks, stride, density):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(EvolvingBottleneck(self.in_planes, planes, s, density=density))
            self.in_planes = planes * EvolvingBottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class EvolvingProtoNet(nn.Module):
    def __init__(self, x_dim=3, z_dim=64, density=0.5):
        super(EvolvingProtoNet, self).__init__()
        self.encoder = EvolvingResNet50(input_channel=x_dim, density=density)
        # ResNet50 outputs 512 * 4 = 2048 features
        self.evolving_head = EvolvingLinear(2048, z_dim, density=density)

    def forward(self, x):
        features = self.encoder(x)
        out = self.evolving_head(features)
        return out

    def evolve_all(self, prune_threshold=0.01, growth_ratio=0.05):
        total_pruned = 0
        total_grown = 0
        modules_count = 0
        changed_masks = {}

        for name, module in self.named_modules():
            if isinstance(module, EvolvingBase):
                p, g, changed = module.evolve(prune_threshold, growth_ratio)
                total_pruned += p
                total_grown += g
                modules_count += 1
                changed_masks[module.weight] = changed

        return total_pruned, total_grown, modules_count, changed_masks


class OmniglotBoosterTaskSampler:
    def __init__(self, train=True, n_way=5, k_shot=1, n_query=15):
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        transform = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            lambda x: 1.0 - x,
            lambda x: x.repeat(3, 1, 1)
        ])
        self.ds = Omniglot('../data', background=train, download=True, transform=transform)
        self.indices = {}
        for idx in range(len(self.ds)):
            l = self.ds._flat_character_images[idx][1]
            if l not in self.indices:
                self.indices[l] = []
            self.indices[l].append(idx)
        self.classes = list(self.indices.keys())

    def get_episode(self):
        c_idxs = np.random.choice(self.classes, self.n_way, replace=False)
        s_list, q_list = [], []
        for c in c_idxs:
            needed = self.k_shot + self.n_query
            replace = True if len(self.indices[c]) < needed else False
            imgs = np.random.choice(self.indices[c], needed, replace=replace)
            s_list.extend(imgs[:self.k_shot])
            q_list.extend(imgs[self.k_shot:])

        s_imgs = torch.stack([self.ds[i][0] for i in s_list])
        q_imgs = torch.stack([self.ds[i][0] for i in q_list])
        s_labels = torch.arange(self.n_way).long().view(-1, 1).repeat(1, self.k_shot).view(-1)
        q_labels = torch.arange(self.n_way).long().view(-1, 1).repeat(1, self.n_query).view(-1)
        return s_imgs, s_labels, q_imgs, q_labels
