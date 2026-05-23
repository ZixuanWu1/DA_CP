import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Dict

# Sklearn & Imbalanced-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, log_loss, hamming_loss, recall_score, precision_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    warnings.warn("XGBoost not installed.")

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Optional LAKCP
try:
    from LatentKernCP.lakcp import LAKCP
except Exception as e:
    LAKCP = None
    warnings.warn(f"LAKCP not available: {e}")

warnings.simplefilter("ignore", category=ConvergenceWarning)

# -----------------------------
# Configuration
# -----------------------------
class CDAConfig:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('cvae_batch_size', 128)
        self.context_dim = kwargs.get('cvae_context_dim', 16)
        self.latent_dim = kwargs.get('cvae_latent_dim', 16)
        self.hidden_dim = kwargs.get('cvae_hidden_dim', 64)
        self.lr = kwargs.get('cvae_lr', 1e-3)
        self.epochs = kwargs.get('cvae_epochs', 50)
        self.beta_kl = kwargs.get('cvae_beta_kl', 1.0)
        self.gen_k = kwargs.get('cvae_gen_k', 5)
        self.gen_tau = kwargs.get('cvae_gen_tau', 0.1)
        self.lambda_badness = kwargs.get('lambda_badness', 0.6)
        self.rho_budget = kwargs.get('rho_budget', 0)
        self.alpha = kwargs.get('alpha', 0.1)

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def create_powerset_labels(y_df: pd.DataFrame) -> Tuple[np.ndarray, LabelEncoder]:
    """Combines binary targets into single class ID."""
    y_str = y_df.astype(str).agg('_'.join, axis=1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    return y_enc, le

def compute_stable_rank(X):

    """

    Computes Stable Rank: ||X||_F^2 / ||X||_2^2

    Measure of data diversity/effective rank.

    """

    if len(X) == 0: return 0.0

    # Ensure numpy array

    X_np = np.array(X)

    if X_np.ndim == 1: 

        X_np = X_np.reshape(-1, 1)

        

    # Compute singular values (SVD)

    # s values are sorted descending

    try:

        s = np.linalg.svd(X_np, compute_uv=False)

        if s[0] == 0: return 0.0

        

        frobenius_norm_sq = np.sum(s**2)

        spectral_norm_sq = s[0]**2

        

        return frobenius_norm_sq / (spectral_norm_sq + 1e-10)

    except Exception:

        return 0.0


# -----------------------------
# CVAE Models
# -----------------------------
class ContextNet(nn.Module):
    def __init__(self, in_dim: int, context_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, context_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class CVAE(nn.Module):
    def __init__(self, x_dim: int, context_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.x_dim, self.context_dim, self.latent_dim, self.hidden_dim = x_dim, context_dim, latent_dim, hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + context_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, x_dim)
        )
    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=-1))
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar

def train_cvae(X_data: np.ndarray, cfg: CDAConfig, device: torch.device) -> Tuple[CVAE, ContextNet]:
    if len(X_data) < 2: return None, None
    X_t = torch.tensor(X_data, dtype=torch.float32)
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    
    context_net = ContextNet(X_data.shape[1], cfg.context_dim).to(device)
    model = CVAE(X_data.shape[1], cfg.context_dim, cfg.latent_dim, cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(context_net.parameters()), lr=cfg.lr)
    
    model.train()
    for _ in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            c = context_net(xb)
            x_hat, mu, logvar = model(xb, c)
            recon = F.mse_loss(x_hat, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + cfg.beta_kl * kl
            opt.zero_grad(); loss.backward(); opt.step()
    return model, context_net

@torch.no_grad()
def generate_for_seed(seed_x: np.ndarray, model: CVAE, context_net: ContextNet, K: int, tau: float, device: torch.device) -> np.ndarray:
    x = torch.tensor(seed_x[None, :], dtype=torch.float32, device=device)
    c = context_net(x).repeat(K, 1)
    z = torch.randn(K, model.latent_dim, device=device) * tau
    return model.decode(z, c).cpu().numpy()

# -----------------------------
# Quality Assessment & A-hat
# -----------------------------
class GeometricQualityAssessor:
    def __init__(self, references: np.ndarray, k: int = 10):
        self.refs = references
        self.k = min(k, len(references))
        if self.k > 0:
            self.nn_model = NearestNeighbors(n_neighbors=self.k).fit(self.refs)
            all_knn_dists, _ = self.nn_model.kneighbors(self.refs)
            self.median_dist = np.median(np.mean(all_knn_dists, axis=1))
        else:
            self.nn_model = None

    def compute_A(self, x_generated: np.ndarray, x_seed: np.ndarray) -> float:
        if self.nn_model is None: return 0.0
        # KNN similarity
        dists, _ = self.nn_model.kneighbors(x_generated.reshape(1, -1))
        avg_dist = np.mean(dists)
        s_knn = float(np.exp(-avg_dist / (self.median_dist + 1e-8)))
        # Cosine similarity to seed
        sim = cosine_similarity(x_generated.reshape(1, -1), x_seed.reshape(1, -1))[0, 0]
        s_cosine = float((sim + 1) / 2)
        return np.sqrt(s_knn * s_cosine)

def train_ahat_predictor(X_synth, scores_A, X_real, seed, cfg):
    """Trains the regressor to predict Quality A."""
    # Combine synthetic (with varying A) and real (A=1.0)
    if len(X_synth) == 0: return None
    
    X_train = pd.concat([X_synth, X_real], ignore_index=True)
    y_train = np.concatenate([scores_A, np.ones(len(X_real))])
    
    if getattr(cfg, 'ahat_model', 'gbrt') == 'xgb' and xgb is not None:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=seed)
    else:
        model = GradientBoostingRegressor(random_state=seed)
        
    model.fit(X_train, y_train)
    return model

# -----------------------------
# Conformal Prediction
# -----------------------------
def splitconformal_quantile(vals: np.ndarray, alpha: float) -> float:
    n = len(vals)
    if n == 0: return -np.inf
    q_idx = int(np.ceil((n + 1) * (1.0 - alpha)))
    q_idx = min(max(1, q_idx), n)
    return float(np.sort(vals)[q_idx - 1])

def find_min_s(A_hat, A_true, lam, rho):
    # Find smallest threshold s on A_hat such that we don't accept too many bad points (A < lam)
    candidates = np.sort(np.unique(A_hat))
    for s in candidates:
        accepted = (A_hat >= s)
        bad_count = np.sum(accepted & (A_true < lam))
        if bad_count <= rho:
            return s
    return np.max(A_hat) + 1e-6

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_multilabel(y_true: np.ndarray, y_prob_list: List[np.ndarray]) -> dict:
    """
    Evaluates multi-label performance. 
    Saves F1, Recall, Precision for EVERY individual task.
    """
    # 1. Handle DataFrame vs Array to get names
    if isinstance(y_true, pd.DataFrame):
        col_names = y_true.columns.tolist()
        y_true_np = y_true.values
    else:
        y_true_np = y_true
        col_names = [f"task_{i}" for i in range(y_true_np.shape[1])]
        
    y_prob = np.column_stack([p[:, 1] for p in y_prob_list])
    y_pred = (y_prob > 0.5).astype(int)

    results = {}
    
    # 2. Global Metrics
    try: results["roc_auc_macro"] = roc_auc_score(y_true_np, y_prob, average='macro')
    except: results["roc_auc_macro"] = np.nan
        
    results["hamming_loss"] = hamming_loss(y_true_np, y_pred)
    results["precision_macro"] = precision_score(y_true_np, y_pred, average='macro', zero_division=0)
    results["recall_macro"] = recall_score(y_true_np, y_pred, average='macro', zero_division=0)
    
    losses = []
    for i in range(y_true_np.shape[1]):
        if len(np.unique(y_true_np[:, i])) > 1:
            losses.append(log_loss(y_true_np[:, i], y_prob[:, i]))
    results["log_loss"] = np.mean(losses) if losses else np.nan

    # 3. Per-Task Metrics (Saving ALL of them)
    f1_list, recall_list, prec_list, auc_list = [], [], [], []
    
    for i, task_name in enumerate(col_names):
        # Clean task name (e.g. "target_RAZRIV" -> "RAZRIV")
        clean_name = task_name.replace("target_", "")
        
        # F1
        f1 = f1_score(y_true_np[:, i], y_pred[:, i], zero_division=0)
        results[f"{clean_name}_f1"] = f1
        f1_list.append(f1)
        
        # Recall
        rec = recall_score(y_true_np[:, i], y_pred[:, i], zero_division=0)
        results[f"{clean_name}_recall"] = rec
        recall_list.append(rec)
        
        # Precision
        prec = precision_score(y_true_np[:, i], y_pred[:, i], zero_division=0)
        results[f"{clean_name}_precision"] = prec
        prec_list.append(prec)

        # AUC
        try:
            if len(np.unique(y_true_np[:, i])) > 1:
                auc = roc_auc_score(y_true_np[:, i], y_prob[:, i])
            else: auc = np.nan
        except: auc = np.nan
        results[f"{clean_name}_auc"] = auc
        if not np.isnan(auc): auc_list.append(auc)

    return results

# -----------------------------
# Main Logic
# -----------------------------
def run_for_seed(cfg, seed, cvae_cfg):
    set_seed(seed)
    print(f"\n--- Seed {seed} ---")
    
    # 1. Load & Prep
    df = pd.read_csv(cfg.data_path)
    target_cols = [c for c in df.columns if c.startswith('target_')]
    if not target_cols: raise ValueError("No targets found.")
    
    X = df.drop(columns=target_cols)
    Y = df[target_cols]
    
    # Label Powerset
    y_lp, le = create_powerset_labels(Y)
    
    # 2. Splits: Train / Calib / Test
    # Filter tiny classes
    counts = pd.Series(y_lp).value_counts()
    valid_cls = counts[counts > 2].index
    mask = pd.Series(y_lp).isin(valid_cls)
    X, Y, y_lp = X[mask], Y[mask], y_lp[mask]
    
    # Split Test
    X_temp, X_test, Y_temp, Y_test, y_lp_temp, _ = train_test_split(
        X, Y, y_lp, test_size=0.2, stratify=y_lp, random_state=seed
    )
    # Split Train / Calib
    X_train, X_cal, Y_train, Y_cal, y_lp_train, y_lp_cal = train_test_split(
        X_temp, Y_temp, y_lp_temp, test_size=0.2, stratify=y_lp_temp, random_state=seed
    )
    
    print(f"   Train: {len(X_train)}, Calib: {len(X_cal)}, Test: {len(X_test)}")
    
    # 3. Setup Synthetic Data Generation
    # We define 'Minority' as any label combo that is NOT the majority class.
    # We aim to balance all classes to the count of the majority class.
    
    maj_class = pd.Series(y_lp_train).value_counts().idxmax()
    max_count = pd.Series(y_lp_train).value_counts().max()
    
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_cal_sc = scaler.transform(X_cal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Storage for generated data
    synth_data_unfiltered = [] # List of (X, Y, A_score, Seed_Vector)
    
    # Dictionary to store CVAE models per powerset class
    models_per_class = {} 
    qa_per_class = {} # Quality Assessors
    
    # Train CVAEs and Generate Initial Pool
    unique_classes = np.unique(y_lp_train)
    for cls in unique_classes:
        if cls == maj_class: continue
        
        # Get Real Samples for this class
        mask_cls = (y_lp_train == cls)
        X_cls = X_train_sc[mask_cls]
        Y_cls_row = Y_train[mask_cls].iloc[0] # All same labels
        
        if len(X_cls) < 5: continue # Skip tiny classes for CVAE
        
        # Train CVAE
        model, ctx = train_cvae(X_cls, cvae_cfg, device)
        if model is None: continue
        
        models_per_class[cls] = (model, ctx)
        
        # Setup QA for this class (Assess against real samples of THIS class)
        qa = GeometricQualityAssessor(X_cls)
        qa_per_class[cls] = qa
        
        # Generate Pool (Oversample heavily to allow filtering)
        n_needed = max_count - len(X_cls)
        n_gen = n_needed * 3 # generate 3x to filter later
        
        gen_sc = []
        seeds_sc = []
        
        # Generate loop
        while len(gen_sc) < n_gen:
            idx = np.random.choice(len(X_cls))
            seed_vec = X_cls[idx]
            batch = generate_for_seed(seed_vec, model, ctx, cvae_cfg.gen_k, cvae_cfg.gen_tau, device)
            for g in batch:
                gen_sc.append(g)
                seeds_sc.append(seed_vec)
                
        gen_sc = np.array(gen_sc[:n_gen])
        seeds_sc = np.array(seeds_sc[:n_gen])
        gen_raw = scaler.inverse_transform(gen_sc)
        seeds_raw = scaler.inverse_transform(seeds_sc)
        
        # Compute A scores immediately
        a_scores = np.array([qa.compute_A(g, s) for g, s in zip(gen_sc, seeds_sc)]) # Compute in scaled space? Usually QA is better in scaled space.
        
        # Store
        for i in range(n_gen):
            synth_data_unfiltered.append({
                'X': gen_raw[i],
                'Y': Y_cls_row.values, # Assign the label of the class
                'A': a_scores[i],
                'cls': cls
            })

    # Convert to DataFrame
    if not synth_data_unfiltered:
            print("   No synthetic data generated (classes too small).")
            filters = {"Baseline": None, "ROS": "ros", "SMOTE": "smote"}
            scores_A = np.array([])
            scores_Ahat = np.array([])
    else:
        df_synth = pd.DataFrame([d['X'] for d in synth_data_unfiltered], columns=X_train.columns)
        df_synth_y = pd.DataFrame([d['Y'] for d in synth_data_unfiltered], columns=Y_train.columns)
        scores_A = np.array([d['A'] for d in synth_data_unfiltered])
        synth_classes = np.array([d['cls'] for d in synth_data_unfiltered])
        
        # 4. Train A-hat Predictor
        # We train one regressor on all synthetic data features -> A score
        print("   Training A-hat predictor...")
        ahat_model = train_ahat_predictor(df_synth, scores_A, X_train, seed, cfg)
        scores_Ahat = ahat_model.predict(df_synth)
        
        # 5. Compute CP / LAKCP Thresholds using Calibration Set
        # For each minority class in Calib, generate synthetic data and see how bad it is
        S_scores = []
        Z_calib_seeds = [] # For LAKCP features
        
        for cls in np.unique(y_lp_cal):
            if cls not in models_per_class: continue
            
            mask_cal = (y_lp_cal == cls)
            X_cal_cls = X_cal_sc[mask_cal]
            model, ctx = models_per_class[cls]
            qa = qa_per_class[cls]
            
            # For each calib seed, generate small batch
            for seed_vec in X_cal_cls:
                batch_sc = generate_for_seed(seed_vec, model, ctx, 5, cvae_cfg.gen_tau, device)
                batch_raw = scaler.inverse_transform(batch_sc)
                batch_df = pd.DataFrame(batch_raw, columns=X_train.columns)
                
                # True A (using QA relative to seed)
                a_true = np.array([qa.compute_A(g, seed_vec) for g in batch_sc])
                # Predicted A
                a_pred = ahat_model.predict(batch_df)
                
                # S score for this seed
                s_i = find_min_s(a_pred, a_true, cvae_cfg.lambda_badness, cvae_cfg.rho_budget)
                S_scores.append(s_i)
                Z_calib_seeds.append(seed_vec) # Keep scaled seed for LAKCP
        
        S_scores = np.array(S_scores)
        q_cp = splitconformal_quantile(S_scores, cvae_cfg.alpha)
        
        # LAKCP Cutoffs (Optional)
        lakcp_cutoffs = None
        if LAKCP is not None and len(Z_calib_seeds) > 10:
            # We need Z_aug for the synthetic data we want to filter
            Z_aug = scaler.transform(df_synth.values) # Synthetic data in scaled space
            lakcp = LAKCP(alpha=cvae_cfg.alpha, randomize=True)
            # Use PCA for kernel features if dim is high
            pca = PCA(n_components=min(10, X_train.shape[1])).fit(X_train_sc)
            Z_cal_pca = pca.transform(np.array(Z_calib_seeds))
            Z_aug_pca = pca.transform(Z_aug)
            
            # Dummy Y for LAKCP fit (not used for S-score logic but required by API)
            dummy_y = np.zeros((len(Z_cal_pca), 1))
            lakcp_cutoffs, _ = lakcp.fit(Z_cal_pca, dummy_y, S_scores, Z_aug_pca, np.zeros((len(Z_aug), 1)))
            lakcp_cutoffs = np.array(lakcp_cutoffs).flatten()

        # 6. Define Filters
        filters = {
            "Baseline": None,
            "ROS": "ros",
            "SMOTE": "smote", # ADDED SMOTE
            "Unfiltered": np.ones(len(df_synth), dtype=bool),
            "A_Filter": (scores_A >= cvae_cfg.lambda_badness),
            "Ahat_Filter": (scores_Ahat >= cvae_cfg.lambda_badness),
            "CP_Filter": (scores_Ahat >= q_cp)
        }
        if lakcp_cutoffs is not None:
            filters["LAKCP_Filter"] = (scores_Ahat >= lakcp_cutoffs)

    # 7. Run Comparisons
    results = []
    
    # Base classifier factory
    def get_clf():
        if getattr(cfg, 'classifier', 'xgboost') == 'rf':
            est = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight='balanced')
        else:
            est = xgb.XGBClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        return MultiOutputClassifier(est)

    print(f"   Evaluating {len(filters)} strategies...")
    
    for name, mask in filters.items():
        if name == "ROS":
            ros = RandomOverSampler(random_state=seed)
            idx_res, _ = ros.fit_resample(np.arange(len(X_train)).reshape(-1,1), y_lp_train)
            X_curr = X_train.iloc[idx_res.flatten()]
            Y_curr = Y_train.iloc[idx_res.flatten()]
        elif name == "SMOTE":
            # SMOTE works on Label Powerset (Multiclass)
            # Use k_neighbors=1 to handle small classes
            sm = SMOTE(random_state=seed, k_neighbors=1)
            try:
                X_res, y_lp_res = sm.fit_resample(X_train, y_lp_train)
                # Reconstruct Multi-Label Y from Powerset IDs
                y_str_res = le.inverse_transform(y_lp_res)
                # Parse strings "0_1_0..." back to DataFrame
                Y_res_vals = np.array([list(map(int, s.split('_'))) for s in y_str_res])
                
                X_curr = pd.DataFrame(X_res, columns=X_train.columns)
                Y_curr = pd.DataFrame(Y_res_vals, columns=Y_train.columns)
            except Exception as e:
                print(f"      [Warning] SMOTE failed: {e}. Falling back to Baseline.")
                X_curr, Y_curr = X_train, Y_train
        elif name == "Baseline":
            X_curr, Y_curr = X_train, Y_train

        else:
            # Filter synthetic data
            if mask is None or sum(mask) == 0:
                print(f"      Strategy {name} resulted in 0 synthetic samples.")
                X_curr, Y_curr = X_train, Y_train
            else:
                X_syn_filt = df_synth[mask]
                Y_syn_filt = df_synth_y[mask]
                
                # We generated 3x, so we might need to downsample back to balance
                # (Simple approach: just append all valid good samples)
                X_curr = pd.concat([X_train, X_syn_filt], ignore_index=True)
                Y_curr = pd.concat([Y_train, Y_syn_filt], ignore_index=True)

        # --- NEW: Compute Added Count and Diversity ---
        n_added = len(X_curr) - len(X_train)
        diversity = compute_stable_rank(X_curr)

        # Train
        clf = get_clf()
        clf.fit(X_curr, Y_curr)
        
        # Predict
        y_prob_list = clf.predict_proba(X_test)
        met = evaluate_multilabel(Y_test.values, y_prob_list)
        # Save Metrics
        met['Strategy'] = name
        met['Seed'] = seed
        met['n_added'] = n_added
        met['diversity_stable_rank'] = diversity
        results.append(met)
        print(f"      {name:<15} LogLoss: {met['log_loss']:.4f} | Hamming: {met['hamming_loss']:.4f}")

    return pd.DataFrame(results)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./outputs')
    p.add_argument('--seeds', type=str, default='42')
    p.add_argument('--classifier', type=str, default='xgboost')
    p.add_argument('--cvae_true', type=int, default=1)
    p.add_argument('--cvae_epochs', type=int, default=100)
    p.add_argument('--cvae_batch_size', type=int, default=128)
    p.add_argument('--cvae_latent_dim', type=int, default=16)
    p.add_argument('--cvae_context_dim', type=int, default=32)
    p.add_argument('--cvae_hidden_dim', type=int, default=64)
    p.add_argument('--cvae_lr', type=float, default=1e-3)
    p.add_argument('--cvae_beta_kl', type=float, default=1.0)
    p.add_argument('--cvae_gen_k', type=int, default=50)
    p.add_argument('--cvae_gen_tau', type=float, default=1.0)
    
    # CVAE / CP Params
    p.add_argument('--lambda_badness', type=float, default=0.7)
    p.add_argument('--rho_budget', type=int, default=1)
    p.add_argument('--alpha', type=float, default=0.1)
    return p.parse_args()

def main():
    cfg = parse_args()
    ensure_dir(cfg.out_dir)
    cvae_cfg = CDAConfig(**vars(cfg))
    
    seeds = [int(s.strip()) for s in cfg.seeds.split(',') if s.strip()]
    all_res = []
    
    for seed in tqdm(seeds, desc="Seeds"):
        try:
            res_df = run_for_seed(cfg, seed, cvae_cfg)
            seed_dir = os.path.join(cfg.out_dir, f"seed_{seed}")
            ensure_dir(seed_dir)
            
            # Save the per-seed results DF
            res_df.to_csv(os.path.join(seed_dir, "results.csv"), index=False)
            all_res.append(res_df)
        except Exception as e:
            print(f"Seed {seed} Fail: {e}")
            import traceback
            traceback.print_exc()

    if all_res:
        final = pd.concat(all_res, ignore_index=True)
        final.to_csv(os.path.join(cfg.out_dir, "multilabel_full_results.csv"), index=False)
        print("\n=== Summary ===")
        print(final.groupby('Strategy').mean(numeric_only=True))

if __name__ == "__main__":
    main()