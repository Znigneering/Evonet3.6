import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import random 
import numpy as np

# Ensure Evonet can be imported from the current directory
try:
    from Evonet import EvolvingProtoNet, OmniglotBoosterTaskSampler
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Evonet import EvolvingProtoNet, OmniglotBoosterTaskSampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.prototypes = None

    def process_support_set(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        support_features = self.backbone(support_images)
        n_way = int(support_labels.max().item()) + 1
        
        self.prototypes = torch.stack([
            support_features[support_labels == c].mean(dim=0) for c in range(n_way)
        ])

    def forward(self, query_images: torch.Tensor) -> torch.Tensor:
        query_features = self.backbone(query_images)
        n = query_features.size(0)
        m = self.prototypes.size(0)
        
        distances = torch.pow(query_features, 2).sum(1, keepdim=True).expand(n, m) + \
                    torch.pow(self.prototypes, 2).sum(1, keepdim=True).expand(m, n).t()
        
        distances = distances - 2 * torch.mm(query_features, self.prototypes.t())
        
        return -distances

def main():
    # Seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 1. Configuration
    K_SHOT = 1
    N_QUERY = 15 # Better batch size for stable gradients
    N_WAY_TRAIN = 60
    N_TRAIN_EPISODES = 5000 
    LEARNING_RATE = 1e-3 # Optimized LR for Adam
    PRUNE_THRESHOLD = 0.001
    EVOLVE_INTERVAL = 500 
    VALIDATE_INTERVAL = 500 
    SAVE_PATH = "Baseline/evo_omniglot/evonet_test.pth"
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"Using device: {DEVICE}")

    # 2. Data Samplers
    print("Initializing Omniglot Samplers...")
    train_sampler = OmniglotBoosterTaskSampler(train=True, n_way=N_WAY_TRAIN, k_shot=K_SHOT, n_query=N_QUERY)
    val_sampler = OmniglotBoosterTaskSampler(train=False, n_way=5, k_shot=K_SHOT, n_query=N_QUERY)

    # 3. Model Setup
    print("Initializing EvolvingProtoNet...")
    backbone = EvolvingProtoNet(x_dim=3, z_dim=64, density=0.5) 
    model = PrototypicalNetworks(backbone).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    print(f"Starting Training for {N_TRAIN_EPISODES} episodes...")
    model.train()
    pbar = tqdm(range(N_TRAIN_EPISODES), desc="Training")
    
    best_acc = 0.0

    for i in pbar:
        s_img, s_lbl, q_img, q_lbl = train_sampler.get_episode()
        s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
        q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)
        
        # Forward pass
        model.process_support_set(s_img, s_lbl)
        scores = model(q_img)
        loss = criterion(scores, q_lbl)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        
        # Evolution Step
        if (i + 1) % EVOLVE_INTERVAL == 0:
            pruned, grown, _ = backbone.evolve_all(prune_threshold=PRUNE_THRESHOLD)
            
            # Critical: Reset optimizer to drop momentum for pruned/regrown weights
            current_lr = scheduler.get_last_lr()[0]
            optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

            print(f"\n[Evolution] Pruned: {pruned}, Grown: {grown}")

        # Post-Step Masking enforcement
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, 'mask'):
                    module.weight.data *= module.mask

        # Logging
        if i % 10 == 0:
            acc = (scores.argmax(dim=1) == q_lbl).float().mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        # Validation
        if (i + 1) % VALIDATE_INTERVAL == 0:
            print("\nValidating...")
            model.eval()
            val_accs = []
            with torch.no_grad():
                for _ in range(100): # 100 validation episodes
                    vs_img, vs_lbl, vq_img, vq_lbl = val_sampler.get_episode()
                    model.process_support_set(vs_img.to(DEVICE), vs_lbl.to(DEVICE))
                    v_scores = model(vq_img.to(DEVICE))
                    v_acc = (v_scores.argmax(1) == vq_lbl.to(DEVICE)).float().mean().item()
                    val_accs.append(v_acc)
            
            current_val_acc = np.mean(val_accs)
            print(f"Validation Accuracy: {current_val_acc*100:.2f}%")
            
            if current_val_acc > best_acc:
                best_acc = current_val_acc
                print(f"--> New best model! Saving to {SAVE_PATH}")
                torch.save(model.state_dict(), SAVE_PATH)
            
            model.train()

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()