import os
import sys
import random
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Evonet50 import EvolvingProtoNet
from omni_image_sampler import OmniImageTaskSampler

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
        distances = torch.pow(query_features.unsqueeze(1) - self.prototypes.unsqueeze(0), 2).sum(dim=2)
        return -distances


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Configuration
    N_WAY_TRAIN   = 20
    N_WAY_VAL     = 5
    K_SHOT        = 1
    N_QUERY       = 1
    N_TRAIN_EPISODES  = 100000
    N_VAL_EPISODES    = 1000
    LEARNING_RATE     = 1e-3
    prune_threshold   = 0.001
    EVOLVE_INTERVAL   = 1000
    VALIDATE_INTERVAL = 5000
    _HERE      = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT  = os.path.join(_HERE, '..', 'data', 'OmnImage84_100')
    SAVE_PATH = "Baseline/evo_omniimage/evonet50_omniimage_100k-epoch_1e-3-lr_1000-evo-interval.pth"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"Using device: {DEVICE}")

    # Data samplers
    train_sampler = OmniImageTaskSampler(
        root=DATA_ROOT,
        n_way=N_WAY_TRAIN,
        k_shot=K_SHOT,
        n_query=N_QUERY,
        split='train',
        verbose=True,
    )
    val_sampler = OmniImageTaskSampler(
        root=DATA_ROOT,
        n_way=N_WAY_VAL,
        k_shot=K_SHOT,
        n_query=N_QUERY,
        split='test',
        verbose=True,
    )

    # Model
    print("Initializing EvoNet50...")
    backbone = EvolvingProtoNet(x_dim=3, z_dim=64, density=1)
    model = PrototypicalNetworks(backbone).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"Starting training for {N_TRAIN_EPISODES} episodes...")
    best_acc = 0.0
    running_loss = 0.0
    running_acc  = 0.0
    model.train()
    pbar = tqdm(range(N_TRAIN_EPISODES), desc="Training")

    for i in pbar:
        s_img, s_lbl, q_img, q_lbl = train_sampler.get_episode()
        s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
        q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)

        model.process_support_set(s_img, s_lbl)
        scores = model(q_img)
        loss = criterion(scores, q_lbl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        preds = scores.argmax(dim=1)
        acc = (preds == q_lbl).float().mean()
        running_loss += loss.item()
        running_acc  += acc.item()

        # Evolve
        if (i + 1) % EVOLVE_INTERVAL == 0:
            pruned, grown, count, _ = backbone.evolve_all(prune_threshold=prune_threshold)

        # Progress
        if i % 10 == 0:
            avg_loss = running_loss / 10 if i > 0 else loss.item()
            avg_acc  = running_acc  / 10 if i > 0 else acc.item()
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            running_loss = 0.0
            running_acc  = 0.0

        # Periodic validation
        if i % VALIDATE_INTERVAL == 0:
            print(f"\nValidating at episode {i}...")
            model.eval()
            total_correct = 0
            total_count   = 0
            with torch.no_grad():
                for _ in tqdm(range(50), desc="Validation", leave=False):
                    s_img, s_lbl, q_img, q_lbl = val_sampler.get_episode()
                    s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
                    q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)
                    model.process_support_set(s_img, s_lbl)
                    preds = model(q_img).argmax(dim=1)
                    total_correct += (preds == q_lbl).sum().item()
                    total_count   += len(q_lbl)
            acc_val = total_correct / total_count
            print(f"Validation Accuracy: {acc_val*100:.2f}%")
            print(f"Saving model to checkpoint:{SAVE_PATH}")
            torch.save(model.state_dict(), SAVE_PATH)
            model.train()

    # Final validation
    print("Final validation...")
    model.eval()
    total_correct = 0
    total_count   = 0
    with torch.no_grad():
        for _ in tqdm(range(N_VAL_EPISODES), desc="Final Validation"):
            s_img, s_lbl, q_img, q_lbl = val_sampler.get_episode()
            s_img, s_lbl = s_img.to(DEVICE), s_lbl.to(DEVICE)
            q_img, q_lbl = q_img.to(DEVICE), q_lbl.to(DEVICE)
            model.process_support_set(s_img, s_lbl)
            preds = model(q_img).argmax(dim=1)
            total_correct += (preds == q_lbl).sum().item()
            total_count   += len(q_lbl)
    acc_final = total_correct / total_count
    print(f"Final Validation Accuracy: {acc_final*100:.2f}%")

    print(f"Saving model to {SAVE_PATH}")
    torch.save(model.state_dict(), SAVE_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
