import os
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class OmniImageTaskSampler:
    def __init__(self, root='../data/OmnImage84_100', n_way=5, k_shot=1, n_query=15, split='train', split_ratio=0.8, verbose=True):
        """
        Task Sampler for OmniImage84_100 dataset.
        
        Args:
            root (str): Root directory containing class subdirectories.
            n_way (int): Number of classes per episode.
            k_shot (int): Number of support samples per class.
            n_query (int): Number of query samples per class.
            split (str): 'train', 'test', 'val' or 'all'.
            split_ratio (float): Ratio of classes to use for training (default 0.8).
            verbose (bool): Whether to print dataset stats.
        """
        self.root = root
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found at {root}")
            
        # 1. Discover Classes
        self.all_classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        
        if len(self.all_classes) == 0:
             raise ValueError(f"No class directories found in {root}")

        # 2. Split Logic (Deterministic based on sort order)
        # Assuming classes are somewhat randomized or specific order doesn't matter much for general functionality, 
        # or that alphabetical sort separates them consistently.
        # Ideally, we shuffle with a fixed seed if we want random classes.
        
        # Using a fixed shuffle for consistency across runs
        rng = random.Random(42)
        shuffled_classes = list(self.all_classes)
        rng.shuffle(shuffled_classes)
        
        num_train = int(len(shuffled_classes) * split_ratio)
        
        if split == 'train':
            self.classes = shuffled_classes[:num_train]
        elif split == 'test' or split == 'val':
            # Use the rest for test/val (simplifying for now, can add specific val split if needed)
            self.classes = shuffled_classes[num_train:]
        else: # 'all'
            self.classes = shuffled_classes
            
        if verbose:
            print(f"OmniImageTaskSampler ({split}): Loaded {len(self.classes)} classes from {len(self.all_classes)} total.")
        
        # 3. Cache Image Paths
        # We store paths to avoid hitting FS every episode
        self.images_by_class = {}
        empty_classes = []
        
        # Scanning classes
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        for c in self.classes:
            c_dir = os.path.join(root, c)
            images = []
            for fname in os.listdir(c_dir):
                if os.path.splitext(fname)[1].lower() in valid_extensions:
                    images.append(os.path.join(c_dir, fname))
            
            if len(images) >= (k_shot + n_query):
                self.images_by_class[c] = images
            else:
                if verbose and len(images) > 0:
                    print(f"Warning: Class {c} has only {len(images)} images, skipping (need {k_shot + n_query}).")
                empty_classes.append(c)
        
        # Filter valid classes
        self.classes = [c for c in self.classes if c in self.images_by_class]
        
        if len(self.classes) < n_way:
            raise ValueError(f"Not enough valid classes ({len(self.classes)}) for {n_way}-way classification.")

        # 4. Transforms
        self.transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            # Normalize to match approximate statistics if needed, usually good for pre-trained models
            # Here we just output [0, 1] tensor for flexibility, or standardized stats?
            # Standard ImageNet normalization:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_episode(self):
        """
        Samples a task/episode.
        Returns:
            s_imgs: (N*K, 3, 84, 84)
            s_labels: (N*K)
            q_imgs: (N*Q, 3, 84, 84)
            q_labels: (N*Q)
        """
        sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        s_imgs_list = []
        s_labels_list = []
        q_imgs_list = []
        q_labels_list = []
        
        for label_idx, c in enumerate(sampled_classes):
            # Sample images
            # Replace=False usually
            all_imgs = self.images_by_class[c]
            # Ensure we don't error if somehow list mutated, but we checked len in init
            if len(all_imgs) < self.k_shot + self.n_query:
                 # Should not happen given init check, but could safeguard
                 raise RuntimeError(f"Class {c} has insufficient images during sampling!")
                 
            chosen_imgs = np.random.choice(all_imgs, self.k_shot + self.n_query, replace=False)
            
            for i, img_path in enumerate(chosen_imgs):
                # Load
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Fallback? Just retry? For now raise.
                    raise e
                
                if i < self.k_shot:
                    s_imgs_list.append(img_tensor)
                    s_labels_list.append(label_idx)
                else:
                    q_imgs_list.append(img_tensor)
                    q_labels_list.append(label_idx)
                    
        # Stack
        s_imgs = torch.stack(s_imgs_list)
        s_labels = torch.tensor(s_labels_list, dtype=torch.long)
        q_imgs = torch.stack(q_imgs_list)
        q_labels = torch.tensor(q_labels_list, dtype=torch.long)
        
        return s_imgs, s_labels, q_imgs, q_labels

if __name__ == "__main__":
    print("Testing OmniImageTaskSampler...")
    try:
        # Test with defaults (assumes data/OmnImage84_100 exists)
        sampler = OmniImageTaskSampler(n_way=5, k_shot=1, n_query=5, split='train', verbose=True)
        
        print("Sampling one episode...")
        s_x, s_y, q_x, q_y = sampler.get_episode()
        
        print("Episode Shapes:")
        print(f"Support X: {s_x.shape}") # Expected: [5, 3, 84, 84]
        print(f"Support Y: {s_y.shape}") # Expected: [5]
        print(f"Query X:   {q_x.shape}") # Expected: [25, 3, 84, 84]
        print(f"Query Y:   {q_y.shape}") # Expected: [25]
        
        print("\nClasses sampled (Indices):", s_y.unique())
        print("Success!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
