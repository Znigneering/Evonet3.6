import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot
import numpy as np
import math
from tqdm import tqdm
import sys
import os

try:
    from additive_boosting_tim import AdditiveBoostingForest
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from additive_boosting_tim import AdditiveBoostingForest

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================================
# 1. Model Definitions (Must match the checkpoint structure EXACTLY)
# =========================================================================

class EvolvingBase(nn.Module):
    def __init__(self, density):
        super(EvolvingBase, self).__init__()
        self.density = density

    def init_mask(self):
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
            pruned_count = mask_prune.sum().item()

        if self.weight.grad is None: 
            self.weight.data *= self.mask
            return pruned_count, 0

        with torch.no_grad():
            grad_mag = torch.abs(self.weight.grad)
            grad_mag[self.mask == 1] = 0 
            
            max_grow = int(self.weight.numel() * growth_ratio)
            if max_grow > 0:
                top_grads, top_indices = torch.topk(grad_mag.view(-1), k=max_grow)
                effective_indices = top_indices[top_grads > 1e-4]
                
                self.mask.view(-1)[effective_indices] = 1
                self.weight.data.view(-1)[effective_indices] = 0 
                grown_count = len(effective_indices)
            else:
                grown_count = 0
                
        self.weight.data *= self.mask
        return pruned_count, grown_count


class EvolvingLinear(EvolvingBase):
    def __init__(self, in_features, out_features, density=0.2):
        super(EvolvingLinear, self).__init__(density)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.init_mask()

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class EvolvingConv2d(EvolvingBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, density=0.5):
        super(EvolvingConv2d, self).__init__(density)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.register_buffer('mask', torch.ones(out_channels, in_channels, kernel_size, kernel_size))
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        self.init_mask()

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding)


def conv3x3(in_planes, out_planes, stride=1, density=0.5):
    return EvolvingConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, density=density)

def conv1x1(in_planes, out_planes, stride=1, density=0.5):
    return EvolvingConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, density=density)

class EvolvingBlock(nn.Module):
    def __init__(self, in_planes, out_planes, density=0.5):
        super(EvolvingBlock, self).__init__()
        
        self.conv1 = conv3x3(in_planes, out_planes, density=density)
        self.bn1 = nn.BatchNorm2d(out_planes,track_running_stats=False)
        self.conv2 = conv3x3(out_planes, out_planes, density=density)
        self.bn2 = nn.BatchNorm2d(out_planes,track_running_stats=False)
        self.conv3 = conv3x3(out_planes, out_planes, density=density)
        self.bn3 = nn.BatchNorm2d(out_planes,track_running_stats=False)

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, density=density),
                nn.BatchNorm2d(out_planes,momentum=1e-3),
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
        
        out += identity
        out = self.relu(out)
        
        out = self.maxpool(out)
        return out

class EvolvingResNet12(nn.Module):
    def __init__(self, output_size, channels=[64, 160, 320, 640], input_channel=3, density=0.5):
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
    def __init__(self, x_dim=1, z_dim=64,density = 0.5):
        super(EvolvingProtoNet, self).__init__()
        self.encoder = EvolvingResNet12(output_size=640, input_channel=3, density=density)
        self.evolving_head = EvolvingLinear(640, z_dim, density=density)

    def forward(self, x):
        features = self.encoder(x) 
        out = self.evolving_head(features)
        return out

    def evolve_all(self, prune_threshold=0.01, growth_ratio=0.05):
        total_pruned = 0
        total_grown = 0
        modules_count = 0
        for name, module in self.named_modules():
            if isinstance(module, EvolvingBase):
                p, g = module.evolve(prune_threshold, growth_ratio)
                total_pruned += p
                total_grown += g
                modules_count += 1
        return total_pruned, total_grown, modules_count

# =========================================================================
# 2. Data Sampler
# =========================================================================

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
            if l not in self.indices: self.indices[l] = []
            self.indices[l].append(idx)
        self.classes = list(self.indices.keys())

    def get_episode(self):
        c_idxs = np.random.choice(self.classes, self.n_way, replace=False)
        s_list, q_list = [], []
        # Support and Query indices
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

def proto_loss(support, query, n_way, k_shot, n_query):
    dim = support.size(1)
    protos = support.view(n_way, k_shot, dim).mean(1)
    
    n = query.size(0)
    m = protos.size(0)
    dists = torch.pow(query, 2).sum(1, keepdim=True).expand(n, m) + \
            torch.pow(protos, 2).sum(1, keepdim=True).expand(m, n).t()
    dists.addmm_(query, protos.t(), beta=1, alpha=-2)
    logits = -dists
    
    target = torch.arange(n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target = target.reshape(-1).to(logits.device)
    
    loss = F.cross_entropy(logits, target)
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc

# =========================================================================
# 3. Main Experiment
# =========================================================================

def main():
    print(f"Using device: {DEVICE}")
    checkpoint_path = "./Baseline/evo_omniglot/evonet_test.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print("Loading Baseline Model...")
    # Load non-weights_only=True because we define classes here
    try:
        model = torch.load(checkpoint_path, map_location=DEVICE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # Freeze model params
    for param in model.parameters():
        param.requires_grad = False

    # Test parameters
    N_way_test = 5 
    K_shot = 5
    N_query = 15
    n_test_episodes = 10

    print(f"Evaluating on {n_test_episodes} episodes ({N_way_test}-way {K_shot}-shot)...")
    
    test_sampler = OmniglotBoosterTaskSampler(train=False, n_way=N_way_test, k_shot=K_shot, n_query=N_query)

    correct_proto = 0
    correct_booster = 0
    total_predictions = 0
    
    pbar = tqdm(range(n_test_episodes), desc="Testing")
    
    for _ in pbar:
        # Get Data
        s_imgs, s_lbls, q_imgs, q_lbls = test_sampler.get_episode()
        s_imgs, s_lbls = s_imgs.to(DEVICE), s_lbls.to(DEVICE)
        q_imgs, q_lbls = q_imgs.to(DEVICE), q_lbls.to(DEVICE)
        
        # --- A. Baseline ProtoNet ---
        with torch.no_grad():
            all_raw = torch.cat([s_imgs, q_imgs])
            # 1. Full output for Baseline validation
            all_emb = model(all_raw)
            s_emb = all_emb[:N_way_test*K_shot]
            q_emb = all_emb[N_way_test*K_shot:]
            
            _, acc = proto_loss(s_emb, q_emb, N_way_test, K_shot, N_query)
            correct_proto += acc.item() * len(q_lbls)
            
            # 2. Encoder output for Boosting (User Request)
            all_feats = model.encoder(all_raw)
            s_feats = all_feats[:N_way_test*K_shot]
            q_feats = all_feats[N_way_test*K_shot:]

        # --- B. Transductive Boosting (TIM) ---
        z_s_np = s_feats.cpu().numpy().astype(np.float64)
        z_q_np = q_feats.cpu().numpy().astype(np.float64)
        y_s_np = s_lbls.cpu().numpy().astype(np.int64)
        y_q_np = q_lbls.cpu().numpy().astype(np.int64)
        
        episode_data = (z_s_np, y_s_np, z_q_np, y_q_np, None)
        
        # Adjusted parameters for stability
        forest = AdditiveBoostingForest(
            n_rounds=20,          
            n_epochs=1, 
            learning_rate=0.01, 
            pop_size=10,       
            generations=2,     
            temperature=10.0,
            sample_ratio=1.0   
        )
        
        # Redirect stdout to suppress booster per-round printing
        sys.stdout = open(os.devnull, 'w')
        forest.fit([episode_data])
        sys.stdout = sys.__stdout__
        
        forest.register_prototypes(z_s_np, y_s_np)
        preds_cls, _ = forest.predict(z_q_np)
        
        # Calculate Booster Acc
        correct_booster += np.sum(preds_cls == y_q_np)
        
        total_predictions += len(q_lbls)
        
        acc_p = correct_proto / total_predictions
        acc_b = correct_booster / total_predictions
        
        pbar.set_postfix(Proto=f"{acc_p:.4f}", Booster=f"{acc_b:.4f}")

    print(f"\nFinal Results ({n_test_episodes} episodes):")
    print(f"ProtoNet Accuracy: {correct_proto/total_predictions:.4f}")
    print(f"Booster   Accuracy: {correct_booster/total_predictions:.4f}")
    print(f"Improvement:       {(correct_booster - correct_proto)/total_predictions * 100:.2f}%")

if __name__ == "__main__":
    main()
