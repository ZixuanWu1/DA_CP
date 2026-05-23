import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List

# Sklearn & Imbalanced-Learn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, precision_score, recall_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import RandomOverSampler, SMOTE

import math
from numpy.linalg import svd, eigvalsh


# Torch (for CVAE and CE metrics)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.simplefilter("ignore", category=ConvergenceWarning)

# Optional LAKCP
try:
    from LatentKernCP.lakcp import LAKCP
except Exception as e:
    LAKCP = None
    warnings.warn(f"LAKCP not available: {e}")

# -----------------------------
# Configuration Class
# -----------------------------
class CDAConfig:
    """Configuration holder for CVAE and pipeline parameters."""
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('cvae_batch_size', 128)
        self.context_dim = kwargs.get('cvae_context_dim', 16)
        self.latent_dim = kwargs.get('cvae_latent_dim', 16)
        self.hidden_dim = kwargs.get('cvae_hidden_dim', 64)
        self.lr = kwargs.get('cvae_lr', 1e-3)
        self.epochs = kwargs.get('cvae_epochs', 100)
        self.beta_kl = kwargs.get('cvae_beta_kl', 1.0)
        self.gen_k = kwargs.get('cvae_gen_k', 5)
        self.gen_tau = kwargs.get('cvae_gen_tau', 0.1)
        self.lambda_badness = kwargs.get('lambda_badness', 0.6)
        self.rho_budget = kwargs.get('rho_budget', 0)

# -----------------------------
# Utility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stratified_cap(df: pd.DataFrame, label_col: str, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y = df[label_col].values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=n/len(df), random_state=random_state)
    idx_take = next(splitter.split(df, y))[1]
    return df.iloc[idx_take].reset_index(drop=True)

def prepare_splits(data_path: str, seed: int, seed_dir: str, max_train=0, max_calib=0, max_test=0,save = False):
    df = pd.read_csv(data_path)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df['Class'])
    train_df, calib_df = train_test_split(
        train_val_df, test_size=0.25, random_state=seed, stratify=train_val_df['Class'])
    train_df = stratified_cap(train_df, 'Class', max_train, seed)
    calib_df = stratified_cap(calib_df, 'Class', max_calib, seed)
    test_df  = stratified_cap(test_df,  'Class', max_test,  seed)
    
    if save: 
        split_dir = os.path.join(seed_dir, "splits")
        ensure_dir(split_dir)
        train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        calib_df.to_csv(os.path.join(split_dir, "calib.csv"), index=False)
        test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    return train_df, calib_df, test_df


def compute_diversity_metrics(X_df: pd.DataFrame, pairwise_sample:int=2000, ridge:float=1e-8) -> dict:
    """
    Diversity measures on X_df (features only).
    We standardize first for geometry-based metrics, and also use correlation for scale invariance.
    """
    X = X_df.values.astype(float)
    n, d = X.shape
    # Standardize
    scaler_tmp = StandardScaler(with_mean=True, with_std=True)
    Z = scaler_tmp.fit_transform(X)  # zero-mean, unit-variance

    # --- Stable rank (scale invariant under scalar multiples)
    # SVD on Z
    # Use full_matrices=False for speed
    try:
        s = svd(Z, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        # fallback: tiny jitter
        s = svd(Z + 1e-12*np.random.randn(*Z.shape), full_matrices=False, compute_uv=False)
    if len(s) == 0:
        stable_rank = 0.0
    else:
        s2 = (s ** 2)
        stable_rank = float(s2.sum() / (s2.max() + 1e-12))

    # --- Eigen-spectrum entropy on CORRELATION (scale-invariant)
    # correlation matrix of standardized Z is simply (1/(n-1)) Z^T Z → its eigenvalues sum to d
    # We use the correlation (not covariance) explicitly:
    C = np.corrcoef(Z, rowvar=False)  # d x d, unit diagonal
    # Numerical ridge for stability if needed
    if not np.all(np.isfinite(C)):
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = C + ridge * np.eye(d)
    # symmetric → use eigvalsh
    evals = eigvalsh(C)
    evals = np.clip(evals, a_min=1e-12, a_max=None)
    # normalize eigenvalues to a probability vector
    p = evals / evals.sum()
    spectral_entropy = float(-(p * np.log(p)).sum())  # nats; divide by log(d) if you want [0,1]
    spectral_entropy_norm = float(spectral_entropy / math.log(d)) if d > 1 else 0.0

    # --- logdet(correlation) — measures generalized variance volume (scale-invariant)
    logdet_corr = float(np.log(evals).sum())  # since evals are eigvals of C

    # --- Mean pairwise distance (on standardized space), sub-sampled for cost
    if n > 1:
        idx = np.random.choice(n, size=min(n, pairwise_sample), replace=False)
        Zs = Z[idx]
        # compute pairwise distances efficiently
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        G = Zs @ Zs.T
        diag = np.sum(Zs * Zs, axis=1, keepdims=True)
        D2 = diag + diag.T - 2.0 * G
        D = np.sqrt(np.maximum(D2, 0.0))
        # use upper triangle (exclude zeros on diagonal)
        iu = np.triu_indices(len(Zs), k=1)
        mean_pairwise_dist = float(D[iu].mean()) if iu[0].size > 0 else 0.0
        std_pairwise_dist  = float(D[iu].std(ddof=1)) if iu[0].size > 1 else 0.0
    else:
        mean_pairwise_dist, std_pairwise_dist = 0.0, 0.0

    # --- Average feature-wise variance BEFORE standardization (for intuition)
    # (Not scale-invariant; reported just for context)
    mean_feature_var_raw = float(X.var(axis=0).mean())

    return {
        "div_stable_rank": stable_rank,
        "div_spectral_entropy": spectral_entropy,
        "div_spectral_entropy_norm": spectral_entropy_norm,  # in [0,1]
        "div_logdet_corr": logdet_corr,
        "div_mean_pairwise_dist_std": mean_pairwise_dist,
        "div_std_pairwise_dist_std": std_pairwise_dist,
        "div_mean_feature_var_raw": mean_feature_var_raw,
        "div_n": int(n),
        "div_d": int(d),
    }

def _parse_hidden_layers(s: str):
    try:
        return tuple(int(x.strip()) for x in s.split(',') if x.strip())
    except Exception:
        return (256, 128)
    
# -----------------------------
# CVAE for minority generation
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
        enc_in, dec_in = x_dim + context_dim, latent_dim + context_dim
        self.encoder = nn.Sequential(nn.Linear(enc_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(dec_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, x_dim))
    def encode(self, x, c): h = self.encoder(torch.cat([x, c], dim=-1)); return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar): std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z, c): return self.decoder(torch.cat([z, c], dim=-1))
    def forward(self, x, c): mu, logvar = self.encode(x, c); z = self.reparameterize(mu, logvar); x_hat = self.decode(z, c); return x_hat, mu, logvar

def cvae_loss(x, x_hat, mu, logvar, beta_kl: float = 1.0):
    recon = F.mse_loss(x_hat, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta_kl * kl, recon.detach(), kl.detach()

def train_cvae(X_fraud: np.ndarray, cfg: CDAConfig, device: torch.device) -> Tuple[CVAE, ContextNet]:
    X_t = torch.tensor(X_fraud, dtype=torch.float32)
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    context_net = ContextNet(X_fraud.shape[1], cfg.context_dim).to(device)
    model = CVAE(x_dim=X_fraud.shape[1], context_dim=cfg.context_dim, latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(context_net.parameters()), lr=cfg.lr)
    model.train(); context_net.train()
    for epoch in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(device); c = context_net(xb)
            x_hat, mu, logvar = model(xb, c)
            loss, _, _ = cvae_loss(xb, x_hat, mu, logvar, beta_kl=cfg.beta_kl)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, context_net

@torch.no_grad()
def generate_for_seed(seed_x: np.ndarray, model: CVAE, context_net: ContextNet, K: int, tau: float, device: torch.device) -> np.ndarray:
    x = torch.tensor(seed_x[None, :], dtype=torch.float32, device=device)
    c = context_net(x).repeat(K, 1)
    z = torch.randn(K, model.latent_dim, device=device) * tau
    return model.decode(z, c).cpu().numpy()

# -----------------------------
# Quality Score 'A' and Predictor 'Â'
# -----------------------------
class GeometricQualityAssessor:
    def __init__(self, fraud_references: np.ndarray, k: int = 10):
        if len(fraud_references) == 0: raise ValueError("fraud_references cannot be empty.")
        self.fraud_refs = fraud_references
        self.k = min(k, len(fraud_references))
        self.nn_model = NearestNeighbors(n_neighbors=self.k).fit(self.fraud_refs)
        all_knn_dists, _ = self.nn_model.kneighbors(self.fraud_refs)
        self.median_dist = np.median(np.mean(all_knn_dists, axis=1))
    def _get_knn_similarity(self, x_generated: np.ndarray) -> float:
        dists, _ = self.nn_model.kneighbors(x_generated.reshape(1, -1))
        avg_dist = np.mean(dists)
        return float(np.exp(-avg_dist / (self.median_dist + 1e-8)))
    def _get_cosine_similarity(self, x_generated: np.ndarray, x_seed: np.ndarray) -> float:
        sim = cosine_similarity(x_generated.reshape(1, -1), x_seed.reshape(1, -1))[0, 0]
        return float((sim + 1) / 2)
    def compute_A(self, x_generated: np.ndarray, x_seed: np.ndarray) -> float:
        s_knn = self._get_knn_similarity(x_generated)
        s_cosine = self._get_cosine_similarity(x_generated, x_seed)
        return np.sqrt(s_knn * s_cosine)
    def compute_A_variant(self, x_generated, x_seed, variant="full"):
        """
        Compute Ã under different surrogate quality levels.
        
        Variants:
            "full":       sqrt(kNN_sim * cosine_sim)  — current default
            "cosine":     cosine_sim only
            "knn":        kNN_sim only
            "l2":         exp(-L2_dist / median_dist)
            "noisy_0.1":  full + N(0, 0.1^2), clipped to [0,1]
            "noisy_0.2":  full + N(0, 0.2^2), clipped to [0,1]
            "noisy_0.3":  full + N(0, 0.3^2), clipped to [0,1]
        """
        if variant == "full":
            return self.compute_A(x_generated, x_seed)
        
        elif variant == "cosine":
            return self._get_cosine_similarity(x_generated, x_seed)
        
        elif variant == "knn":
            return self._get_knn_similarity(x_generated)
        
        elif variant == "l2":
            dist = np.linalg.norm(x_generated - x_seed)
            return float(np.exp(-dist / (self.median_dist + 1e-8)))
        
        elif variant.startswith("noisy_"):
            sigma = float(variant.split("_")[1])
            base = self.compute_A(x_generated, x_seed)
            noisy = base + np.random.normal(0, sigma)
            return float(np.clip(noisy, 0.0, 1.0))
        
        else:
            raise ValueError(f"Unknown variant: {variant}")

def train_and_generate_cvae_minority(
    X_train: pd.DataFrame, y_train: pd.Series, n_generate: int, device: torch.device,
    cfg: CDAConfig, feature_cols: list, save_dir: str,
    surrogate_variant="full"
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, CVAE, ContextNet, MinMaxScaler]:
    X_min_df = X_train[y_train == 1].copy()
    if len(X_min_df) < 10:
        warnings.warn("Very few minority samples to train CVAE.")
    
    # Original behavior: MinMax on minority only
    scaler = MinMaxScaler()
    X_min_sc = scaler.fit_transform(X_min_df.values.astype(float))

    # ---- train CVAE in scaled space
    context_net = ContextNet(X_min_sc.shape[1], cfg.context_dim).to(device)
    model = CVAE(x_dim=X_min_sc.shape[1], context_dim=cfg.context_dim,
                 latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim).to(device)

    X_t = torch.tensor(X_min_sc, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(list(model.parameters()) + list(context_net.parameters()), lr=cfg.lr)
    model.train(); context_net.train()
    for epoch in range(cfg.epochs):
        for (xb,) in dl:
            xb = xb.to(device); c = context_net(xb)
            x_hat, mu, logvar = model(xb, c)
            loss, _, _ = cvae_loss(xb, x_hat, mu, logvar, beta_kl=cfg.beta_kl)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); context_net.eval()

    torch.save(model.state_dict(), os.path.join(save_dir, "cvae_model.pt"))
    torch.save(context_net.state_dict(), os.path.join(save_dir, "context_net.pt"))
    print(f"[Save] Saved VAE models to {save_dir}")

    # ---- generation in scaled space then invert
    quality_assessor = GeometricQualityAssessor(X_min_df.values)
    all_generated_sc, all_seeds_sc = [], []
    while len(all_generated_sc) < n_generate:
        seed_idx = np.random.randint(0, len(X_min_sc))
        seed_x_sc = X_min_sc[seed_idx]
        G_batch_sc = generate_for_seed(seed_x_sc, model, context_net, K=cfg.gen_k, tau=cfg.gen_tau, device=device)
        for row_sc in G_batch_sc:
            all_generated_sc.append(row_sc)
            all_seeds_sc.append(seed_x_sc)

    gen_sc = np.array(all_generated_sc[:n_generate])
    seeds_sc = np.array(all_seeds_sc[:n_generate])
    gen = scaler.inverse_transform(gen_sc)
    seeds = scaler.inverse_transform(seeds_sc)

    scores_A = np.array([quality_assessor.compute_A_variant(g, s, variant=surrogate_variant) for g, s in zip(gen, seeds)])
    
    X_synth = pd.DataFrame(gen, columns=feature_cols)
    y_synth = pd.Series(np.ones(len(X_synth), dtype=int))
    return X_synth, y_synth, scores_A, model, context_net, scaler

def train_Ahat_predictor(X_synth: pd.DataFrame, scores_A_synth: np.ndarray, X_train_minority: pd.DataFrame, seed: int):
    if len(X_synth) == 0:
        reg = GradientBoostingRegressor(random_state=seed)
        reg.fit(X_train_minority, np.ones(len(X_train_minority)))
        return reg
    X_train_Ahat = pd.concat([X_synth, X_train_minority], ignore_index=True)
    y_train_Ahat = np.concatenate([scores_A_synth, np.ones(len(X_train_minority))])
    reg = GradientBoostingRegressor(random_state=seed)
    reg.fit(X_train_Ahat, y_train_Ahat)
    return reg

# -----------------------------
# Conformal Prediction Logic
# -----------------------------
def splitconformal_quantile(vals: np.ndarray, alpha: float) -> float:
    vals = np.asarray(vals, dtype=float); n = int(vals.size)
    if n <= 0: return float("-inf")
    q_idx = int(np.ceil((n + 1) * (1.0 - alpha)))
    q_idx = min(max(1, q_idx), n)
    return float(np.sort(vals)[q_idx - 1])

def find_min_s_for_seed(A_hat: np.ndarray, A_true: np.ndarray, lam: float, rho: int) -> float:
    candidate_thresholds = np.sort(np.unique(A_hat)) 
    valid_thresholds = []
    for s in candidate_thresholds:
        accepted_mask = (A_hat >= s)
        bad_accepted_count = np.sum(accepted_mask & (A_true < lam))
        if bad_accepted_count <= rho:
            valid_thresholds.append(s)
    if valid_thresholds:
        return float(min(valid_thresholds))
    else:
        return float(np.max(A_hat) + 1e-6)  # instead of inf
        # inf gives error at LAKCP

def generate_S_scores_for_calib(
    X_cal: pd.DataFrame, y_cal: pd.Series, 
    model: CVAE, context_net: ContextNet, scaler: MinMaxScaler,
    quality_assessor: GeometricQualityAssessor, 
    reg_Ahat: GradientBoostingRegressor,
    feature_cols: list, seed: int, cfg: argparse.Namespace, cvae_cfg: CDAConfig,
    surrogate_variant="full", 
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds_calib_df = X_cal[y_cal == 1]
    if seeds_calib_df.empty:
        return np.array([]), np.array([])
        
    seeds_calib_scaled = scaler.transform(seeds_calib_df.values)
    S_scores = []
    for i in range(len(seeds_calib_df)):
        seed_unscaled = seeds_calib_df.iloc[i].values
        seed_scaled = seeds_calib_scaled[i]
        
        G_batch = generate_for_seed(seed_scaled, model, context_net, K=cvae_cfg.gen_k, tau=cvae_cfg.gen_tau, device=device)
        G_batch = scaler.inverse_transform(G_batch)
        G_batch_df = pd.DataFrame(G_batch, columns=feature_cols)

        A_true_batch = np.array([quality_assessor.compute_A_variant(g, seed_unscaled, variant=surrogate_variant) for g in G_batch])
        A_hat_batch = reg_Ahat.predict(G_batch_df)
        
        s_i = find_min_s_for_seed(A_hat_batch, A_true_batch, lam=cvae_cfg.lambda_badness, rho=cvae_cfg.rho_budget)
        S_scores.append(s_i)

    pca = PCA(n_components=4, random_state=seed).fit(scaler.transform(X_cal.values))
    Z_calib_seeds = pca.transform(seeds_calib_scaled)
    
    return np.array(S_scores), Z_calib_seeds

# -----------------------------
# Plotting
# -----------------------------
def plot_A_vs_Ahat_scatter(A_true_all: np.ndarray, A_hat_all: np.ndarray, save_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("Matplotlib not found. Skipping plot.")
        return
    lo = float(min(A_true_all.min(), A_hat_all.min())) if len(A_true_all) > 0 else 0
    hi = float(max(A_true_all.max(), A_hat_all.max())) if len(A_true_all) > 0 else 1
    plt.figure(figsize=(6, 6))
    plt.scatter(A_true_all, A_hat_all, alpha=0.5, label="Synthetic Point")
    plt.plot([lo, hi], [lo, hi], '--', color='red', label="y=x (Perfect Prediction)")
    plt.xlabel("True Quality Score (A)")
    plt.ylabel("Predicted Quality Score (Â)")
    plt.title("True A vs. Predicted Â on Calibration Seeds")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_hist_S_i(S_vals: List[float], save_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("Matplotlib not found. Skipping plot.")
        return
    S = np.asarray(S_vals, dtype=float)
    S_clean = S[np.isfinite(S)]
    plt.figure(figsize=(7,5))
    plt.hist(S_clean, bins=30, edgecolor="black", alpha=0.75)
    plt.xlabel("Per-seed threshold $S_i$")
    plt.ylabel("Count")
    plt.title("Histogram of per-seed thresholds $S_i$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# -----------------------------
# Evaluation & Main Loop
# -----------------------------
def evaluate(y_true: pd.Series, y_prob: np.ndarray, amount: pd.Series) -> dict:
    """Calculates a comprehensive set of classification metrics."""
    y_pred = (y_prob > 0.5).astype(int)

    # Standard metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix and derived rates
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Financial and count metrics
    detected_mask = ((y_pred == 1) & (y_true.values == 1))
    missed_mask = ((y_pred == 0) & (y_true.values == 1))
    amount_detected = float(amount.values[detected_mask].sum())
    amount_missed = float(amount.values[missed_mask].sum())

    ce_all = F.binary_cross_entropy(torch.tensor(y_prob, dtype=torch.float32), torch.tensor(y_true.values, dtype=torch.float32)).item()
    mask_min = (y_true.values == 1)
    ce_min = F.binary_cross_entropy(torch.tensor(y_prob[mask_min], dtype=torch.float32), torch.tensor(y_true.values[mask_min], dtype=torch.float32)).item() if mask_min.sum() > 0 else np.nan

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "balanced_accuracy": bal_acc,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "frauds_detected": int(detected_mask.sum()),
        "frauds_missed": int(missed_mask.sum()),
        "amount_detected": amount_detected,
        "amount_missed": amount_missed,
        "ce_all": ce_all,
        "ce_minority": ce_min
    }

def make_classifier(kind: str, seed: int, cfg):
    if kind == 'logistic':
        return LogisticRegression(random_state=seed, max_iter=1000, solver='saga', n_jobs=-1)

    elif kind == 'logistic_l1':
        return LogisticRegression(random_state=seed, max_iter=3000, solver='saga',
                                  penalty='l1', C=1.0, n_jobs=-1)

    elif kind == 'svm':
        # Linear SVM + Platt scaling to get predict_proba
        base = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, loss='squared_hinge', random_state=seed)
        )
        return CalibratedClassifierCV(base, method='sigmoid', cv=3)

    elif kind == 'rf':
        return RandomForestClassifier(
            n_estimators=cfg.rf_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            n_jobs=-1,
            random_state=seed
        )

    elif kind == 'mlp':
        hidden = _parse_hidden_layers(cfg.mlp_hidden)
        # MLP benefits from scaling; use pipeline
        return make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=cfg.mlp_alpha,
                max_iter=cfg.mlp_max_iter,
                early_stopping=bool(cfg.mlp_early_stop),
                random_state=seed
            )
        )

    else:
        raise ValueError(f"Unknown classifier kind: {kind}")

def run_for_seed(cfg: argparse.Namespace, seed: int, cvae_cfg: CDAConfig) -> pd.DataFrame:
    set_seed(seed)
    print(f"Start running experiment for seed {seed} !")
    seed_dir = os.path.join(cfg.out_dir, f"seed_{seed}")
    ensure_dir(seed_dir)
    artifact_dir = os.path.join(seed_dir, "artifacts")
    ensure_dir(artifact_dir)

    train_df, calib_df, test_df = prepare_splits(cfg.data_path, seed, seed_dir, cfg.max_train, cfg.max_calib, cfg.max_test)
    
    feature_cols = [c for c in train_df.columns if c.startswith('V')]
    print(f"[Features] Using {len(feature_cols)} features starting with V.")

    X_train, y_train = train_df[feature_cols], train_df['Class']
    X_cal, y_cal = calib_df[feature_cols], calib_df['Class']
    X_test, y_test = test_df[feature_cols], test_df['Class']

    n_min = (y_train == 1).sum(); n_maj = (y_train == 0).sum()
    n_generate = max(0, n_maj - n_min)

    model, context_net, min_max_scaler = None, None, None
    X_synth, y_synth = pd.DataFrame(columns=feature_cols), pd.Series([], dtype=int)
    scores_A = np.array([])
    
    if cfg.cvae_true and n_generate > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_synth, y_synth, scores_A, model, context_net, min_max_scaler = train_and_generate_cvae_minority(
            X_train, y_train, n_generate, device, cvae_cfg, feature_cols, artifact_dir, 
            surrogate_variant=cfg.surrogate_variant
        )
        # X_synth.to_csv(os.path.join(artifact_dir, "X_synth_from_train_seeds.csv"), index=False)
    
    X_train_minority = X_train[y_train == 1]
    reg_Ahat = train_Ahat_predictor(X_synth, scores_A, X_train_minority, seed)
    
    if model is not None and len(X_cal[y_cal==1]) > 0:
        print("[Plotting] Generating data from calibration seeds for A vs Â plot...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_true_calib_gen, A_hat_calib_gen, all_calib_synth_data = [], [], []
        seeds_calib_unscaled_df = X_cal[y_cal == 1]
        seeds_calib_scaled = min_max_scaler.transform(seeds_calib_unscaled_df.values)
        quality_assessor = GeometricQualityAssessor(X_train_minority.values)
        for i in range(len(seeds_calib_unscaled_df)):
            seed_unscaled = seeds_calib_unscaled_df.iloc[i].values
            seed_scaled = seeds_calib_scaled[i]
            generated_batch_sc = generate_for_seed(seed_scaled, model, context_net, K=cvae_cfg.gen_k, tau=cvae_cfg.gen_tau, device=device)
            generated_batch = min_max_scaler.inverse_transform(generated_batch_sc)
            all_calib_synth_data.append(pd.DataFrame(generated_batch, columns=feature_cols))
            scores_A_true = [quality_assessor.compute_A_variant(g, seed_unscaled, variant=cfg.surrogate_variant) for g in generated_batch]
            scores_A_hat = reg_Ahat.predict(pd.DataFrame(generated_batch, columns=feature_cols))
            A_true_calib_gen.extend(scores_A_true)
            A_hat_calib_gen.extend(scores_A_hat)
        
        if all_calib_synth_data:
            X_synth_calib_df = pd.concat(all_calib_synth_data, ignore_index=True)
            # X_synth_calib_df.to_csv(os.path.join(artifact_dir, "X_synth_from_calib_seeds.csv"), index=False)
            plot_path = os.path.join(artifact_dir, "A_vs_Ahat_calib.png")
            plot_A_vs_Ahat_scatter(np.array(A_true_calib_gen), np.array(A_hat_calib_gen), plot_path)
            print(f"[Plotting & Save] Saved calibration artifacts to {artifact_dir}")

    all_results = []
    
    masks = {}
    q_cp, cutoffs = None, None
    if len(X_synth) > 0:
        A_hat_synth = reg_Ahat.predict(X_synth)
        masks['A-Filter'] = (scores_A >= cfg.lambda_badness)
        masks['Ahat-Filter'] = (A_hat_synth >= cfg.lambda_badness)
        
        quality_assessor = GeometricQualityAssessor(X_train_minority.values)
        S_calib_scores, Z_calib_seeds = generate_S_scores_for_calib(
            X_cal, y_cal, model, context_net, min_max_scaler,
            quality_assessor, reg_Ahat, feature_cols, seed, cfg, cvae_cfg,
             surrogate_variant=cfg.surrogate_variant,  
        )
        if S_calib_scores.size > 0:
            pd.Series(S_calib_scores, name="S_i").to_csv(os.path.join(artifact_dir, "S_i_scores.csv"), index=False)
            hist_path = os.path.join(artifact_dir, "S_i_scores_histogram.png")
            plot_hist_S_i(S_calib_scores, hist_path)
            print(f"[Save] Saved S_i scores and histogram to {artifact_dir}")

        
        if S_calib_scores.size > 5:
            q_cp = splitconformal_quantile(S_calib_scores, cfg.alpha)
            masks['CP-Filter'] = (A_hat_synth >= q_cp)
            if LAKCP is not None:
                pca = PCA(n_components=4, random_state=seed).fit(min_max_scaler.transform(X_train.values))
                Z_aug = pca.transform(min_max_scaler.transform(X_synth.values))
                lakcp = LAKCP(alpha=cfg.alpha, randomize=True, verbose=False, use_cv=True)
                cutoffs, _ = lakcp.fit(Z_calib_seeds, np.ones((len(Z_calib_seeds), 1)), S_calib_scores.ravel(), Z_aug, np.ones((len(Z_aug), 1)))
                masks['LAKCP-Filter'] = (A_hat_synth >= np.asarray(cutoffs).ravel())

        pd.DataFrame({k: pd.Series(v) for k, v in masks.items()}).to_csv(os.path.join(artifact_dir, "filter_masks.csv"), index=False)
        print(f"[Save] Saved filter masks to {artifact_dir}")

        thresholds_to_save = {
            'cp_threshold': q_cp,
            'lakcp_cutoffs': cutoffs.tolist() if cutoffs is not None else None
        }
        with open(os.path.join(artifact_dir, "conformal_thresholds.json"), 'w') as f:
            json.dump(thresholds_to_save, f, indent=4)
        print(f"[Save] Saved CP/LAKCP thresholds to {artifact_dir}")

    strategies: List[Tuple[str, any]] = [
        ("Baseline", "baseline"), ("SMOTE", "smote"), ("RandomOverSampler", "ros"), ("VAE Simple", "vae_simple")
    ]
    for name, mask in masks.items():
        strategies.append((f"VAE + {name}", mask))
    
    print(f"\n--- Evaluating {len(strategies)} strategies for Seed {seed} ---")
    for name, method in strategies:
        added_count = 0
        if isinstance(method, str):
            if method == "baseline":
                X_train_final, y_train_final = X_train, y_train
            elif method == "vae_simple":
                if len(X_synth) == 0: continue
                X_train_final = pd.concat([X_train, X_synth], ignore_index=True)
                y_train_final = pd.concat([y_train, pd.Series(np.ones(len(X_synth)))], ignore_index=True)
                added_count = len(X_synth)
            else:
                sampler = SMOTE(random_state=seed) if method == "smote" else RandomOverSampler(random_state=seed)
                X_train_final_np, y_train_final_np = sampler.fit_resample(X_train.values, y_train.values)
                X_train_final = pd.DataFrame(X_train_final_np, columns=feature_cols)
                y_train_final = pd.Series(y_train_final_np)
                added_count = y_train_final.sum() - y_train.sum()
        else: # VAE strategies with a mask
            if len(X_synth) == 0: continue
            X_kept_aug = X_synth.iloc[method]
            y_kept_aug = y_synth.iloc[method]
            X_train_final = pd.concat([X_train, X_kept_aug], ignore_index=True)
            y_train_final = pd.concat([y_train, y_kept_aug], ignore_index=True)
            added_count = len(X_kept_aug)
        
        try:
            div_metrics = compute_diversity_metrics(X_train_final)
        except Exception as e:
            # Guarantee columns exist even if diversity computation fails
            div_metrics = {
                "div_stable_rank": np.nan,
                "div_spectral_entropy": np.nan,
                "div_spectral_entropy_norm": np.nan,
                "div_logdet_corr": np.nan,
                "div_mean_pairwise_dist_std": np.nan,
                "div_std_pairwise_dist_std": np.nan,
                "div_mean_feature_var_raw": np.nan,
                "div_n": len(X_train_final),
                "div_d": X_train_final.shape[1],
                "div_error": str(e),   # optional: helps you see failures
                }

        
        # we do not want to use balanced weighting because we want to address the imbalance through data augmentation
        clf = make_classifier(cfg.classifier, seed, cfg)
        clf.fit(X_train_final, y_train_final)
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_prob, test_df['Amount'])
        
        row = {
            "Seed": seed, "Strategy": name, "AddedMinority": int(added_count),
            "alpha": cfg.alpha, "lambda_badness": cvae_cfg.lambda_badness, 
            "rho_budget": cvae_cfg.rho_budget, "surrogate_variant": cfg.surrogate_variant,
        }
        row.update(metrics)
        row.update(div_metrics)  # << append the diversity readouts
        all_results.append(row)
        print(f"    - Strategy '{name}': Evaluation complete. PR-AUC = {metrics['pr_auc']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(seed_dir, "comparison_results.csv"), index=False)
    return results_df

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./outputs')
    p.add_argument('--seeds', type=str, default='42')
    p.add_argument('--max_train', type=int, default=0)
    p.add_argument('--max_calib', type=int, default=0)
    p.add_argument('--max_test',  type=int, default=0)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--lambda_badness', type=float, default=0.3, help="Badness threshold for true score A in CP-S logic")
    p.add_argument('--rho_budget', type=int, default=0, help="Error budget (max bad points) for CP-S logic")
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
    p.add_argument('--classifier', type=str, default='logistic',
                choices=['logistic', 'svm', 'logistic_l1', 'mlp', 'rf'],
                help="Base classifier to use (default: logistic)")
    p.add_argument('--surrogate_variant', type=str, default='full',
                choices=['full', 'cosine', 'knn', 'l2', 'noisy_0.1', 'noisy_0.2', 'noisy_0.3'],
                help="Variant of surrogate quality score hatA to use in CP-S logic")

    # Random Forest hyperparams
    p.add_argument('--rf_estimators', type=int, default=500)
    p.add_argument('--rf_max_depth', type=int, default=None)
    p.add_argument('--rf_min_samples_leaf', type=int, default=1)
    p.add_argument('--rf_max_features', type=str, default='sqrt')

    # MLP hyperparams
    p.add_argument('--mlp_hidden', type=str, default='256,128',
                help="Comma-separated sizes, e.g. '256,128'")
    p.add_argument('--mlp_alpha', type=float, default=1e-4)
    p.add_argument('--mlp_max_iter', type=int, default=200)
    p.add_argument('--mlp_early_stop', type=int, default=1)
    return p.parse_args()

def main():
    cfg = parse_args()
    ensure_dir(cfg.out_dir)
    cvae_cfg = CDAConfig(**vars(cfg))
    seeds = [int(s.strip()) for s in cfg.seeds.split(',') if s.strip()]
    
    all_results_dfs = []
    for seed in tqdm(seeds, desc="Seeds"):
        try:
            results_df = run_for_seed(cfg, seed, cvae_cfg)
            all_results_dfs.append(results_df)
            print(f"\n--- Results for Seed {seed} ---")
            print(results_df.to_string(index=False))
        except Exception as e:
            warnings.warn(f"[seed {seed}] failed: {e}")
    
    if all_results_dfs:
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        summary_path = os.path.join(cfg.out_dir, "comparison_results_all_seeds.csv")
        final_df.to_csv(summary_path, index=False)
        print(f"\n[SUCCESS] All seeds complete. Full comparison saved to {summary_path}")

        agg_df = final_df.groupby('Strategy').mean().drop(columns='Seed').reset_index()
        print("\n--- Aggregate Results (Mean over all seeds) ---")
        print(agg_df.to_string(index=False))

if __name__ == "__main__":
    main()