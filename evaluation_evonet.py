# Must be set before any import that touches OpenMP (torch, numpy, etc.).
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys

try:
    from Evonet import EvolvingProtoNet, OmniglotBoosterTaskSampler
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Evonet import EvolvingProtoNet, OmniglotBoosterTaskSampler

# 明确告诉旧版 PyTorch，我们要用的是第 0 号显卡
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.prototypes = None

    def process_support_set(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        # 严格遵守训练时的逻辑：单独处理 Support
        support_features = self.backbone(support_images)
        n_way = int(support_labels.max().item()) + 1
        self.prototypes = torch.zeros(n_way, support_features.shape[1], device=support_features.device)
        for c in range(n_way):
            self.prototypes[c] = support_features[support_labels == c].mean(dim=0)

    def forward(self, query_images: torch.Tensor) -> torch.Tensor:
        # 严格遵守训练时的逻辑：单独处理 Query
        query_features = self.backbone(query_images)
        
        # 使用极其稳定的欧式距离算法
        distances = torch.pow(query_features.unsqueeze(1) - self.prototypes.unsqueeze(0), 2).sum(dim=2)
        return -distances

def proto_loss(s_emb, q_emb, n_way, k_shot, n_query):
    prototypes = s_emb.view(n_way, k_shot, -1).mean(dim=1)

    n = q_emb.size(0)
    m = prototypes.size(0)
    distances = (
        torch.pow(q_emb, 2).sum(1, keepdim=True).expand(n, m)
        + torch.pow(prototypes, 2).sum(1, keepdim=True).expand(m, n).t()
        - 2.0 * torch.mm(q_emb, prototypes.t())
    )

    log_probs = F.log_softmax(-distances, dim=1)
    # .long(): old PyTorch returns FloatTensor from arange; nll_loss needs LongTensor
    labels = (
        torch.arange(n_way).view(-1, 1)
        .repeat(1, n_query)
        .view(-1)
        .long()
        .to(s_emb.device)
    )
    loss = F.nll_loss(log_probs, labels)
    acc = (log_probs.argmax(dim=1) == labels).float().mean()
    return loss, acc


# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"Using device: {DEVICE}")

checkpoint_path = "./Baseline/evo_omniglot/evonet_test.pth"
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

print("Loading Baseline Model...")
# 使用和你 local 跑通时一模一样的参数配置！
backbone = EvolvingProtoNet(x_dim=3, z_dim=64, density=0.5) 
model = PrototypicalNetworks(backbone)
state_dict = torch.load(checkpoint_path, map_location=DEVICE)

# 恢复使用 strict=False，这才是正确的！因为我们要忽略不必要的 BN 参数
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)

# ==================== 终极魔法：不要用 model.eval() ====================
# 强行保持 train 模式，逼迫旧版 PyTorch 乖乖计算当前 batch 的均值和方差！
model.train() 
# =======================================================================

for param in model.parameters():
    param.requires_grad = False

# ── Evaluation parameters ─────────────────────────────────────────────────────
N_way_test = 20
K_shot = 1
N_query = 1
n_test_episodes = 600

print(f"Evaluating on {n_test_episodes} episodes ({N_way_test}-way {K_shot}-shot)...")

test_sampler = OmniglotBoosterTaskSampler(
    train=False, n_way=N_way_test, k_shot=K_shot, n_query=N_query
)

correct = 0
total = 0

pbar = tqdm(range(n_test_episodes), desc="Testing")

for _ in pbar:
    s_imgs, s_lbls, q_imgs, q_lbls = test_sampler.get_episode()
    s_imgs = s_imgs.to(DEVICE)
    q_imgs = q_imgs.to(DEVICE)
    q_lbls = q_lbls.to(DEVICE)

    with torch.no_grad():
        # ======= 恢复你最原始、最正确的代码（拼接计算） =======
        all_raw = torch.cat([s_imgs, q_imgs], dim=0)
        
        # 因为处于 model.train()，这里会完美触发 Transductive BN！
        all_emb = model.backbone(all_raw) 
        
        s_emb = all_emb[:N_way_test * K_shot]
        q_emb = all_emb[N_way_test * K_shot:]
        
        # 使用你原本写好的 proto_loss 
        _, acc = proto_loss(s_emb, q_emb, N_way_test, K_shot, N_query)
        correct += acc.item() * len(q_lbls)
        total += len(q_lbls)

    pbar.set_postfix(acc=f"{correct / total:.4f}")

print(f"\nFinal Results ({n_test_episodes} episodes, {N_way_test}-way {K_shot}-shot):")
print(f"EvoNet ProtoNet Accuracy: {correct / total:.4f} ({correct / total * 100:.2f}%)")