import random
import numpy as np
import sys
import time
import warnings
from numba import jit, float64, int64

# ==========================================
# 0. Global Spec
# ==========================================
sys.setrecursionlimit(20000)
warnings.filterwarnings('ignore')

# ==========================================
# 1. Numba Utils
# ==========================================
@jit(nopython=True, cache=True, fastmath=True)
def numba_l2_norm_and_div(x):
    n, d = x.shape
    out = np.empty_like(x)
    for i in range(n):
        sum_sq = 0.0
        for j in range(d):
            sum_sq += x[i, j] ** 2
        norm = np.sqrt(sum_sq) + 1e-8
        for j in range(d):
            out[i, j] = x[i, j] / norm
    return out

@jit(nopython=True, cache=True, fastmath=True)
def numba_calculate_loss_tim(z_s, y_s_idx, z_q, q_y_idx, n_cls, temperature):
    """
    TIM Loss:
    Minimize Conditional Entropy (H_cond) - Maximize Marginal Entropy (H_marg)
    Loss = H_cond - H_marg
    
    Note: q_y_idx is UNUSED for calculation but kept for signature compatibility.
    TIM is unsupervised on Query set labels.
    """
    D = z_s.shape[1]
    n_q = z_q.shape[0]
    scale = temperature * 2.0 
    
    # --- 1. Compute Prototypes from Support Set ---
    # Using simple mean prototype for now
    protos = np.zeros((n_cls, D), dtype=np.float64)
    counts = np.zeros(n_cls, dtype=np.float64)
    
    for i in range(z_s.shape[0]):
        cls_k = y_s_idx[i]
        protos[cls_k] += z_s[i]
        counts[cls_k] += 1.0
        
    for k in range(n_cls):
        if counts[k] > 0: protos[k] /= counts[k]
        # L2 Normalize Prototypes
        norm = 0.0
        for d in range(D): norm += protos[k, d]**2
        norm = np.sqrt(norm) + 1e-8
        protos[k] /= norm

    # --- 2. Compute Logits and Softmax on Query Set ---
    # We need full probabilities for Entropy
    probs = np.zeros((n_q, n_cls), dtype=np.float64)
    
    # Store logits temporarily to compute softmax stably
    for i in range(n_q):
        max_logit = -1e30
        row_logits = np.zeros(n_cls, dtype=np.float64)
        
        for k in range(n_cls):
            dot_val = 0.0
            for d in range(D): dot_val += z_q[i, d] * protos[k, d]
            logit = dot_val * scale
            row_logits[k] = logit
            if logit > max_logit: max_logit = logit
        
        sum_exp = 0.0
        for k in range(n_cls): 
            val = np.exp(row_logits[k] - max_logit)
            probs[i, k] = val
            sum_exp += val
            
        # Normalize to get Probabilities
        for k in range(n_cls):
            probs[i, k] /= sum_exp

    # --- 3. Compute Conditional Entropy (H(Y|X)) ---
    # H_cond = - (1/N) * sum_i sum_k P_ik * log(P_ik)
    h_cond = 0.0
    for i in range(n_q):
        row_h = 0.0
        for k in range(n_cls):
            p = probs[i, k]
            # Avoid log(0)
            if p > 1e-12:
                row_h += p * np.log(p)
        h_cond -= row_h
    h_cond /= n_q

    # --- 4. Compute Marginal Entropy (H(Y)) ---
    # P_k = (1/N) * sum_i P_ik
    # H_marg = - sum_k P_k * log(P_k)
    h_marg = 0.0
    for k in range(n_cls):
        p_marg_k = 0.0
        for i in range(n_q):
            p_marg_k += probs[i, k]
        p_marg_k /= n_q
        
        if p_marg_k > 1e-12:
            h_marg -= p_marg_k * np.log(p_marg_k)

    # --- 5. Total Loss ---
    # Minimize H_cond, Maximize H_marg => Minimize (H_cond - H_marg)
    # To keep loss positive, we use (H_cond + (log(N) - H_marg))
    return h_cond - h_marg + np.log(float(n_cls))


# ==========================================
# 2. GP Nodes (Same)
# ==========================================
def protected_div(left, right):
    return np.divide(left, np.where(np.abs(right) < 1e-3, 1.0, right))

OPERATORS = {
    '+':    {'func': np.add,      'arity': 2},
    '-':    {'func': np.subtract, 'arity': 2},
    '*':    {'func': np.multiply, 'arity': 2},
    '/':    {'func': protected_div, 'arity': 2},
    'abs':  {'func': np.abs,      'arity': 1},
    'max':  {'func': np.maximum,  'arity': 2},
}

class Node:
    __slots__ = ('value', 'left', 'right', 'node_type', '_signature')
    def __init__(self, value, left=None, right=None, node_type='op'):
        self.value = value
        self.left = left
        self.right = right
        self.node_type = node_type
        self._signature = None 

    def get_signature(self):
        if self._signature is None:
            if self.node_type == 'terminal': self._signature = (0, self.value, None, None)
            else:
                l_sig = self.left.get_signature()
                r_sig = self.right.get_signature() if self.right else None
                self._signature = (1, self.value, l_sig, r_sig)
        return self._signature

    def evaluate(self, X, memo=None):
        if memo is None: return self._compute(X, None)
        sig = self.get_signature()
        if sig in memo: return memo[sig]
        result = self._compute(X, memo)
        memo[sig] = result 
        return result

    def _compute(self, X, memo):
        if self.node_type == 'terminal':
            if isinstance(self.value, int): return X[:, self.value]
            else: return np.full(X.shape[0], self.value)
        elif self.node_type == 'op':
            op = OPERATORS[self.value]
            if op['arity'] == 1: return op['func'](self.left.evaluate(X, memo))
            else: return op['func'](self.left.evaluate(X, memo), self.right.evaluate(X, memo))
    
    def clone(self):
        if self.node_type == 'terminal': return Node(self.value, node_type='terminal')
        return Node(self.value, self.left.clone() if self.left else None, self.right.clone() if self.right else None, self.node_type)

def generate_random_tree(depth, max_depth, n_features):
    if depth == max_depth or (depth > 0 and random.random() < 0.1):
        if random.random() < 0.6: return Node(random.randint(0, n_features - 1), node_type='terminal')
        return Node(random.uniform(-1, 1), node_type='terminal')
    
    op = random.choice(list(OPERATORS.keys()))
    if OPERATORS[op]['arity'] == 1: return Node(op, left=generate_random_tree(depth+1, max_depth, n_features))
    return Node(op, left=generate_random_tree(depth+1, max_depth, n_features), right=generate_random_tree(depth+1, max_depth, n_features))


# ==========================================
# 3. Additive Boosting Forest (TIM Version)
# ==========================================
class AdditiveBoostingForest:
    def __init__(self, n_rounds=50, n_epochs=5, learning_rate=0.1, pop_size=30, 
                 generations=2, temperature=10.0, sample_ratio=0.1,
                 refine_loops_epoch=2, refine_loops_final=20):
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs 
        self.learning_rate = learning_rate
        self.pop_size = pop_size
        self.generations = generations
        self.temperature = float(temperature)
        self.sample_ratio = sample_ratio
        self.refine_loops_epoch = refine_loops_epoch
        self.refine_loops_final = refine_loops_final
        
        self.rounds_history = []  
        self.prototypes = None         
        self.pillar_class_probs = None 
        self.idx_to_class = {}
        self.n_dims = None

    def fit(self, episodes):
        n_features = episodes[0][0].shape[1]
        self.n_dims = n_features
        n_episodes = len(episodes)
        batch_size = max(1, int(n_episodes * self.sample_ratio))
        rounds_per_epoch = max(1, self.n_rounds // self.n_epochs)
        
        print(f" [Lazy-Update Mode (TIM Loss): Enabled.]")
        print(f" [Config] {self.n_rounds} Rounds, Batch={batch_size}, Epochs={self.n_epochs}")

        # 1. Mega-Batch Init
        all_supp_list, all_query_list = [], []
        supp_slices, query_slices = [], []
        supp_ptr, query_ptr = 0, 0
        episodes_meta = [] 

        for ep in episodes:
            s_X, s_y, q_X, q_y, _ = ep
            all_supp_list.append(s_X); all_query_list.append(q_X)
            n_s, n_q = s_X.shape[0], q_X.shape[0]
            supp_slices.append((supp_ptr, supp_ptr + n_s))
            query_slices.append((query_ptr, query_ptr + n_q))
            supp_ptr += n_s; query_ptr += n_q
            unique_cls = np.unique(s_y)
            s_y_idx = np.searchsorted(unique_cls, s_y).astype(np.int64)
            q_y_idx = np.searchsorted(unique_cls, q_y).astype(np.int64)
            episodes_meta.append((s_y_idx, q_y_idx, len(unique_cls)))

        mega_supp_X = np.concatenate(all_supp_list, axis=0)
        mega_query_X = np.concatenate(all_query_list, axis=0)
        
        # 2. Residual Init (Global State)
        current_supp_emb = mega_supp_X.astype(np.float64)
        current_query_emb = mega_query_X.astype(np.float64)
        
        current_lr = self.learning_rate
        
        # Lazy Update Queue
        pending_trees = [] 
        pending_lrs = []

        # 3. Boosting Loop
        for r in range(self.n_rounds):
            
            # --- A. Sample Batch ---
            batch_indices = np.random.choice(n_episodes, batch_size, replace=False)
            
            batch_s_indices = []
            batch_q_indices = []
            local_supp_slices = []
            local_query_slices = []
            local_s_ptr, local_q_ptr = 0, 0
            
            for e_idx in batch_indices:
                s_len = supp_slices[e_idx][1] - supp_slices[e_idx][0]
                q_len = query_slices[e_idx][1] - query_slices[e_idx][0]
                batch_s_indices.extend(range(supp_slices[e_idx][0], supp_slices[e_idx][1]))
                batch_q_indices.extend(range(query_slices[e_idx][0], query_slices[e_idx][1]))
                local_supp_slices.append((local_s_ptr, local_s_ptr + s_len))
                local_query_slices.append((local_q_ptr, local_q_ptr + q_len))
                local_s_ptr += s_len
                local_q_ptr += q_len
            
            # Batch Data
            batch_supp_X = mega_supp_X[batch_s_indices]
            batch_query_X = mega_query_X[batch_q_indices]
            
            # Current Base
            batch_curr_s_emb = current_supp_emb[batch_s_indices].copy()
            batch_curr_q_emb = current_query_emb[batch_q_indices].copy()

            # --- B. Lazy Update (Catch-up) ---
            if len(pending_trees) > 0:
                l_memo_s = {(0, i, None, None): batch_supp_X[:, i] for i in range(n_features)}
                l_memo_q = {(0, i, None, None): batch_query_X[:, i] for i in range(n_features)}
                
                for t_list, lr in zip(pending_trees, pending_lrs):
                    # Support
                    inc_s = np.stack([t.evaluate(batch_supp_X, memo=l_memo_s) for t in t_list], axis=1)
                    batch_curr_s_emb += lr * inc_s
                    # Query
                    inc_q = np.stack([t.evaluate(batch_query_X, memo=l_memo_q) for t in t_list], axis=1)
                    batch_curr_q_emb += lr * inc_q

            # --- C. GP Evolution ---
            population = [
                [generate_random_tree(0, 3, n_features) for _ in range(self.n_dims)]
                for _ in range(self.pop_size)
            ]
            
            best_gen_loss = 1e9
            best_gen_ind = None

            local_memo_s = {(0, i, None, None): batch_supp_X[:, i] for i in range(n_features)}
            local_memo_q = {(0, i, None, None): batch_query_X[:, i] for i in range(n_features)}

            for gen in range(self.generations):
                loss_list = []
                for individual in population:
                    inc_s = np.stack([t.evaluate(batch_supp_X, memo=local_memo_s) for t in individual], axis=1)
                    inc_q = np.stack([t.evaluate(batch_query_X, memo=local_memo_q) for t in individual], axis=1)

                    temp_z_s = numba_l2_norm_and_div(batch_curr_s_emb + current_lr * inc_s)
                    temp_z_q = numba_l2_norm_and_div(batch_curr_q_emb + current_lr * inc_q)

                    ind_loss = 0.0
                    for k, e_idx in enumerate(batch_indices):
                        s_start, s_end = local_supp_slices[k]
                        q_start, q_end = local_query_slices[k]
                        s_y_idx, q_y_idx, n_cls = episodes_meta[e_idx]
                        
                        # --- USE TIM LOSS HERE ---
                        # NOTE: Still pass q_y_idx for signature, but TIM doesn't use it!
                        ind_loss += numba_calculate_loss_tim(
                            temp_z_s[s_start:s_end], s_y_idx, 
                            temp_z_q[q_start:q_end], q_y_idx, 
                            int(n_cls), self.temperature
                        )
                    loss_list.append(ind_loss / batch_size)

                loss_arr = np.array(loss_list)
                sorted_idx = np.argsort(loss_arr)
                
                if loss_arr[sorted_idx[0]] < best_gen_loss:
                    best_gen_loss = loss_arr[sorted_idx[0]]
                    best_gen_ind = [t.clone() for t in population[sorted_idx[0]]]

                # Reproduce
                top_k = int(self.pop_size * 0.25)
                new_pop = []
                for k in range(top_k):
                    new_pop.append([t.clone() for t in population[sorted_idx[k]]])
                while len(new_pop) < self.pop_size:
                    new_pop.append([generate_random_tree(0, 2, n_features) for _ in range(self.n_dims)])
                population = new_pop
                
            del local_memo_s, local_memo_q

            # --- D. Store & Queue ---
            self.rounds_history.append({'trees': best_gen_ind, 'lr': current_lr})
            pending_trees.append(best_gen_ind)
            pending_lrs.append(current_lr)
            
            print(f"Round {r+1:03d}: Loss(TIM)={best_gen_loss:.4f} | LR={current_lr:.5f} | Pending={len(pending_trees)}")

            current_lr = max(0.005, current_lr * 0.9999) 

            # --- E. Epoch Check & Global Sync ---
            is_epoch_end = ((r + 1) % rounds_per_epoch == 0)
            is_last_round = ((r + 1) == self.n_rounds)
            
            if is_epoch_end or is_last_round:
                # 1. Sync
                if len(pending_trees) > 0:
                    print(f" [Syncing] Global State (Flushing {len(pending_trees)} rounds)...")
                    chunk_size = 50000 
                    
                    for start_idx in range(0, mega_supp_X.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, mega_supp_X.shape[0])
                        sub_X = mega_supp_X[start_idx:end_idx]
                        lm = {(0, i, None, None): sub_X[:, i] for i in range(n_features)}
                        
                        accum_inc = np.zeros_like(current_supp_emb[start_idx:end_idx])
                        for t_list, lr in zip(pending_trees, pending_lrs):
                            accum_inc += lr * np.stack([t.evaluate(sub_X, memo=lm) for t in t_list], axis=1)
                        current_supp_emb[start_idx:end_idx] += accum_inc
                    
                    for start_idx in range(0, mega_query_X.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, mega_query_X.shape[0])
                        sub_X = mega_query_X[start_idx:end_idx]
                        lm = {(0, i, None, None): sub_X[:, i] for i in range(n_features)}
                        
                        accum_inc = np.zeros_like(current_query_emb[start_idx:end_idx])
                        for t_list, lr in zip(pending_trees, pending_lrs):
                            accum_inc += lr * np.stack([t.evaluate(sub_X, memo=lm) for t in t_list], axis=1)
                        current_query_emb[start_idx:end_idx] += accum_inc
                    
                    pending_trees = []
                    pending_lrs = []

                # 2. Refine
                refine_loops = self.refine_loops_final if is_last_round else self.refine_loops_epoch
                if refine_loops > 0:
                    print(f" [Refining] ({refine_loops} passes) ---")
                    
                    ref_batch_size = max(batch_size, 50)
                    ref_indices = np.random.choice(n_episodes, min(n_episodes, ref_batch_size), replace=False)
                    
                    r_s_idx, r_q_idx = [], []
                    r_s_slices, r_q_slices = [], []
                    rp_s, rp_q = 0, 0
                    for e_idx in ref_indices:
                         s_l = supp_slices[e_idx][1] - supp_slices[e_idx][0]
                         q_l = query_slices[e_idx][1] - query_slices[e_idx][0]
                         r_s_idx.extend(range(supp_slices[e_idx][0], supp_slices[e_idx][1]))
                         r_q_idx.extend(range(query_slices[e_idx][0], query_slices[e_idx][1]))
                         r_s_slices.append((rp_s, rp_s + s_l))
                         r_q_slices.append((rp_q, rp_q + q_l))
                         rp_s += s_l; rp_q += q_l
                    
                    r_supp_X = mega_supp_X[r_s_idx]
                    r_query_X = mega_query_X[r_q_idx]
                    r_curr_s_emb = current_supp_emb[r_s_idx].copy()
                    r_curr_q_emb = current_query_emb[r_q_idx].copy()
                    
                    hist_outputs = []
                    lm_s = {(0, i, None, None): r_supp_X[:, i] for i in range(n_features)}
                    lm_q = {(0, i, None, None): r_query_X[:, i] for i in range(n_features)}
                    
                    for record in self.rounds_history:
                        out_s = np.stack([t.evaluate(r_supp_X, memo=lm_s) for t in record['trees']], axis=1)
                        out_q = np.stack([t.evaluate(r_query_X, memo=lm_q) for t in record['trees']], axis=1)
                        hist_outputs.append((out_s, out_q))
                    
                    for loop_i in range(refine_loops):
                        for h_i, record in enumerate(self.rounds_history):
                            old_lr = record['lr']
                            comp_s, comp_q = hist_outputs[h_i]
                            
                            r_curr_s_emb -= old_lr * comp_s
                            r_curr_q_emb -= old_lr * comp_q
                            
                            candidates = [old_lr * 0.8, old_lr, old_lr * 1.2]
                            best_lr = old_lr
                            min_l = 1e9
                            
                            for c_lr in candidates:
                                t_z_s = numba_l2_norm_and_div(r_curr_s_emb + c_lr * comp_s)
                                t_z_q = numba_l2_norm_and_div(r_curr_q_emb + c_lr * comp_q)
                                l_val = 0.0
                                for k, e_idx in enumerate(ref_indices):
                                    ss, se = r_s_slices[k]; qs, qe = r_q_slices[k]
                                    sy, qy, nc = episodes_meta[e_idx]
                                    # --- USE TIM LOSS HERE ---
                                    l_val += numba_calculate_loss_tim(
                                        t_z_s[ss:se], sy, t_z_q[qs:qe], qy, int(nc), self.temperature
                                    )
                                if l_val < min_l: min_l = l_val; best_lr = c_lr
                            
                            record['lr'] = best_lr
                            r_curr_s_emb += best_lr * comp_s
                            r_curr_q_emb += best_lr * comp_q
                    
                    print(" [Applying] Refined LRs to Global State...")
                    current_supp_emb[:] = mega_supp_X
                    current_query_emb[:] = mega_query_X
                    
                    chunk_size = 50000
                    for start_idx in range(0, mega_supp_X.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, mega_supp_X.shape[0])
                        sub_X = mega_supp_X[start_idx:end_idx]
                        lm = {(0, i, None, None): sub_X[:, i] for i in range(n_features)}
                        acc = np.zeros_like(current_supp_emb[start_idx:end_idx])
                        for record in self.rounds_history:
                            acc += record['lr'] * np.stack([t.evaluate(sub_X, memo=lm) for t in record['trees']], axis=1)
                        current_supp_emb[start_idx:end_idx] += acc

                    for start_idx in range(0, mega_query_X.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, mega_query_X.shape[0])
                        sub_X = mega_query_X[start_idx:end_idx]
                        lm = {(0, i, None, None): sub_X[:, i] for i in range(n_features)}
                        acc = np.zeros_like(current_query_emb[start_idx:end_idx])
                        for record in self.rounds_history:
                            acc += record['lr'] * np.stack([t.evaluate(sub_X, memo=lm) for t in record['trees']], axis=1)
                        current_query_emb[start_idx:end_idx] += acc

    def apply_forest(self, X):
        total_output = X.astype(np.float64)
        local_memo = {}
        for i in range(X.shape[1]):
            local_memo[(0, i, None, None)] = X[:, i]

        for record in self.rounds_history:
            trees = record['trees']
            lr = record['lr'] 
            inc = np.stack([t.evaluate(X, memo=local_memo) for t in trees], axis=1)
            total_output += lr * inc
            
        return total_output

    def register_prototypes(self, support_X, support_y):
        embedding_raw = self.apply_forest(support_X)
        embedding = numba_l2_norm_and_div(embedding_raw)
        
        unique_classes = np.unique(support_y)
        unique_classes.sort()
        self.idx_to_class = {i: c for i, c in enumerate(unique_classes)}
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        
        n_classes = len(unique_classes)
        D = embedding.shape[1]
        self.prototypes = np.zeros((n_classes, D))
        
        for i, c in enumerate(unique_classes):
             self.prototypes[i] = np.mean(embedding[support_y == c], axis=0)
        
        p_norm = np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8
        self.prototypes = self.prototypes / p_norm
        
        logits = (embedding @ self.prototypes.T) * (self.temperature * 2)
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        support_pillar_probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        
        y_indices = np.array([class_to_idx[y] for y in support_y])
        y_onehot = np.zeros((len(support_y), n_classes))
        y_onehot[np.arange(len(support_y)), y_indices] = 1.0
        
        raw_weights = support_pillar_probs.T @ y_onehot
        self.pillar_class_probs = raw_weights / (np.sum(raw_weights, axis=1, keepdims=True) + 1e-10)

    def predict(self, query_X):
        if self.prototypes is None: raise ValueError("Wait for registration")
        
        embedding_raw = self.apply_forest(query_X)
        z_query = numba_l2_norm_and_div(embedding_raw)
        logits = (z_query @ self.prototypes.T) * (self.temperature * 2)
        
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        final_probs = probs @ self.pillar_class_probs
        
        best_idx = np.argmax(final_probs, axis=1)
        return np.array([self.idx_to_class[i] for i in best_idx]), np.max(final_probs, axis=1)


if __name__ == "__main__":
    print("Testing TIM Loss Function...")
    # Dummy data
    n_features = 20
    d_out = n_features
    # Support: 5 way, 5 shot
    n_cls = 5
    n_shot = 5
    n_query = 15
    
    # Random embeddings
    z_s = np.random.randn(n_cls * n_shot, d_out).astype(np.float64)
    y_s = np.repeat(np.arange(n_cls), n_shot).astype(np.int64)
    
    z_q = np.random.randn(n_cls * n_query, d_out).astype(np.float64)
    y_q = np.repeat(np.arange(n_cls), n_query).astype(np.int64) # Not used in TIM
    
    # Normalize z_s, z_q
    z_s = numba_l2_norm_and_div(z_s)
    z_q = numba_l2_norm_and_div(z_q)
    
    start = time.time()
    loss = numba_calculate_loss_tim(z_s, y_s, z_q, y_q, n_cls, 10.0)
    end = time.time()
    
    print(f"Loss: {loss}")
    print(f"Time: {end - start:.6f} sec")
    
    print("\nStarting Forest integration test...")
    # Mock Episode
    episode = (
        np.random.randn(n_cls * n_shot, n_features),
        y_s,
        np.random.randn(n_cls * n_query, n_features),
        y_q,
        None
    )
    
    forest = AdditiveBoostingForest(n_rounds=2, n_epochs=1, pop_size=5, generations=1)
    forest.fit([episode])
    print("Fit complete.")
