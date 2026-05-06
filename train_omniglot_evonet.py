import os
import sys
import torch
from torch import nn
from tqdm import tqdm
import random 
import numpy as np

try:
    from Evonet50 import EvolvingProtoNet, OmniglotBoosterTaskSampler
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from Evonet50 import EvolvingProtoNet, OmniglotBoosterTaskSampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.prototypes = None

    def process_support_set(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        support_features = self.backbone(support_images)
        n_way = int(support_labels.max().item()) + 1
        self.prototypes = torch.zeros(n_way, support_features.shape[1], device=support_features.device)
        for c in range(n_way):
            self.prototypes[c] = support_features[support_labels == c].mean(dim=0)

    def forward(self, query_images: torch.Tensor) -> torch.Tensor:
        query_features = self.backbone(query_images)
        # PyTorch 0.4.0 compatible squared euclidean distance
        distances = torch.pow(query_features.unsqueeze(1) - self.prototypes.unsqueeze(0), 2).sum(dim=2)
        return -distances

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Configuration
    K_SHOT = 1
    N_QUERY = 1 # Standard for Omniglot
    N_TRAIN_EPISODES = 10000
    N_VAL_EPISODES = 10
    LEARNING_RATE = 1e-3
    prune_threshold = 0.01
    EVOLVE_INTERVAL = 50 # Evolve every N episodes
    VALIDATE_INTERVAL = 500 # Validate every N episodes
    SAVE_PATH = "Baseline/evo_omniglot/evonet50_test.pth"
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f"Using device: {DEVICE}")

    # 2. Data Samplers
    print("Initializing Omniglot Samplers...")
    train_sampler = OmniglotBoosterTaskSampler(
        train=True,
        n_way=80,
        k_shot=K_SHOT,
        n_query=N_QUERY
    )
    
    val_sampler = OmniglotBoosterTaskSampler(
        train=False,
        n_way=20,
        k_shot=K_SHOT,
        n_query=N_QUERY
    )

    # 3. Model Setup
    print("Initializing EvolvingProtoNet...")
    # EvoNet expects input_channel=3 (hardcoded in class for now)
    backbone = EvolvingProtoNet(x_dim=3, z_dim=640,density=1) 
    
    model = PrototypicalNetworks(backbone).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    print(f"Starting Training for {N_TRAIN_EPISODES} episodes...")
    
    model.train()
    pbar = tqdm(range(N_TRAIN_EPISODES), desc="Training")
    
    running_loss = 0.0
    running_acc = 0.0
    
    for i in pbar:
        # Load episode
        s_img, s_lbl, q_img, q_lbl = train_sampler.get_episode()
        s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
        q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)
        
        # Forward
        model.process_support_set(s_img, s_lbl)
        scores = model(q_img)
        
        loss = criterion(scores, q_lbl)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()
        scheduler.step()
        
        preds = scores.argmax(dim=1)
        acc = (preds == q_lbl).float().mean()
        
        running_loss += loss.item()
        running_acc += acc.item()
        
        # Evolve (Prune/Regrow)
        if (i + 1) % EVOLVE_INTERVAL == 0:
            pruned, grown, count, _ = backbone.evolve_all(prune_threshold=prune_threshold)
        
        # Stats
        if i % 10 == 0:
            avg_loss = running_loss / 10 if i > 0 else loss.item()
            avg_acc = running_acc / 10 if i > 0 else acc.item()
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            running_loss = 0.0
            running_acc = 0.0
        
        if i % VALIDATE_INTERVAL == 0:
            # Validation
            print("Validating...")
            model.eval()
            total_correct = 0
            total_count = 0
            
            with torch.no_grad():
                for _ in tqdm(range(50), desc="Validation"):
                    s_img, s_lbl, q_img, q_lbl = val_sampler.get_episode()
                    s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
                    q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)
                    
                    model.process_support_set(s_img, s_lbl)
                    scores = model(q_img)
                    preds = scores.argmax(dim=1)
                    
                    total_correct += (preds == q_lbl).sum().item()
                    total_count += len(q_lbl)

            acc_val = total_correct / total_count
            print(f"Validation Accuracy: {acc_val*100:.2f}%")
            model.train()

    # 5. Validation / Testing
    print("Validating...")
    model.eval()
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for _ in tqdm(range(N_VAL_EPISODES), desc="Validation"):
            s_img, s_lbl, q_img, q_lbl = val_sampler.get_episode()
            s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
            q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)
            
            model.process_support_set(s_img, s_lbl)
            scores = model(q_img)
            preds = scores.argmax(dim=1)
            
            total_correct += (preds == q_lbl).sum().item()
            total_count += len(q_lbl)
            
    acc_val = total_correct / total_count
    print(f"Validation Accuracy: {acc_val*100:.2f}%")
    
    # 6. Save
    print(f"Saving model to {SAVE_PATH}")
    torch.save(model.state_dict(), SAVE_PATH) 
    print("Done!")

if __name__ == "__main__":
    main()
