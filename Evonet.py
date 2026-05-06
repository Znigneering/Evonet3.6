import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot
import numpy as np
import math

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
            # Prune small-magnitude active weights
            active_weights = torch.abs(self.weight.data)
            mask_prune = (active_weights < prune_threshold) & (self.mask == 1)
            self.mask[mask_prune] = 0
            self.weight.data[mask_prune] = 0.0
            pruned_count = int(mask_prune.sum().item())

            # Grow: select inactive positions with largest gradient signal.
            # STE in forward() routes real gradients to inactive weights, so this
            # actually selects meaningful candidates (fixes gradient mask deadlock).
            grown_count = 0
            # torch.bool was added in PyTorch 1.2; use byte() for old-PyTorch
            # (Python 3.6) compatibility.  0/1 byte tensor is identical semantically.
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
                        mask_grow.view(-1)[effective_indices] = 1  # 1 not True: byte tensor compat
                        grown_count = len(effective_indices)

            # Return changed positions so the caller can surgically reset Adam state.
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
        # Straight-through estimator: output is correctly zero for inactive weights,
        # but autograd sees d(effective_weight)/d(weight) = 1 everywhere.
        # This provides a real gradient signal for inactive positions without
        # contaminating the forward computation (fixes gradient mask deadlock).
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

class EvolvingBlock(nn.Module):
    def __init__(self, in_planes, out_planes, density=0.5):
        super(EvolvingBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, density=density)
        self.bn1 = nn.BatchNorm2d(out_planes, track_running_stats=False)
        self.conv2 = conv3x3(out_planes, out_planes, density=density)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)
        self.conv3 = conv3x3(out_planes, out_planes, density=density)
        self.bn3 = nn.BatchNorm2d(out_planes, track_running_stats=False)

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, density=density),
                nn.BatchNorm2d(out_planes, track_running_stats=False),
            )

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        out = self.maxpool(out)

        return out

class EvolvingResNet12(nn.Module):
    def __init__(self, output_size=640, channels=[64, 160, 320, 640], input_channel=3, density=0.5):
        super(EvolvingResNet12, self).__init__()

        self.layer1 = EvolvingBlock(input_channel, channels[0], density)
        self.layer2 = EvolvingBlock(channels[0], channels[1], density)
        self.layer3 = EvolvingBlock(channels[1], channels[2], density)
        self.layer4 = EvolvingBlock(channels[2], channels[3], density)

        self.output_size = output_size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
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
        self.encoder = EvolvingResNet12(output_size=640, input_channel=x_dim, density=density)
        self.evolving_head = EvolvingLinear(640, z_dim, density=density)

    def forward(self, x):
        features = self.encoder(x)
        out = self.evolving_head(features)
        return out

    def evolve_all(self, prune_threshold=0.01, growth_ratio=0.05):
        total_pruned = 0
        total_grown = 0
        modules_count = 0
        changed_masks = {}  # param tensor -> boolean changed-position mask

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

        # ds[i] returns (image_tensor, label_int); [0] extracts the image tensor
        s_imgs = torch.stack([self.ds[i][0] for i in s_list])
        q_imgs = torch.stack([self.ds[i][0] for i in q_list])
        s_labels = torch.arange(self.n_way).long().view(-1, 1).repeat(1, self.k_shot).view(-1)
        q_labels = torch.arange(self.n_way).long().view(-1, 1).repeat(1, self.n_query).view(-1)
        return s_imgs, s_labels, q_imgs, q_labels
