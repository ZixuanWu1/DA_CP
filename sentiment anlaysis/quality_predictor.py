from typing import Sequence, Callable, Tuple, Union, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.linear_model import RidgeCV
import numpy as np

_EMBEDDER: SentenceTransformer | None = None


def _get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def embed_texts(texts: Sequence[str], *, device: str | None = None) -> torch.Tensor:
    model = _get_embedder()
    if device:
        model = model.to(device)
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)



def _graph_features(sim_G: np.ndarray, *, close_thresh: float = 0.7) -> np.ndarray:
    """Return (K,4): degree, clustering, PageRank, betweenness."""
    A = (sim_G > close_thresh).astype(int)
    G = nx.from_numpy_array(A)
    K = sim_G.shape[0]
    degree      = np.array([d for _, d in G.degree()]).reshape(K, 1)
    clustering  = np.array(list(nx.clustering(G).values())).reshape(K, 1)
    pagerank    = np.array(list(nx.pagerank(G).values())).reshape(K, 1)
    betweenness = np.array(list(nx.betweenness_centrality(G).values())).reshape(K, 1)
    return np.hstack([degree, clustering, pagerank, betweenness])



def _features_single(
    seed: str,
    summary: str,
    gens: Sequence[str],
    *,
    prob_negative: Callable[[str], float],
    include_graph: bool,
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    gens = list(gens)
    if len(gens) == 0:
        raise ValueError("`gens` must contain at least one string.")

    # -- embeddings ------------------------------------------------------
    embs = embed_texts([seed, summary] + gens, device=device)
    em_seed, em_sum, em_gens = embs[0], embs[1], embs[2:]

    cos_seed = util.cos_sim(em_gens, em_seed).cpu().numpy().ravel()
    cos_sum  = util.cos_sim(em_gens, em_sum ).cpu().numpy().ravel()

    prob_g    = np.array([prob_negative(g) for g in gens])
    pol_shift = prob_g - prob_negative(seed)
    len_ratio = np.array([len(g.split()) / max(len(seed.split()), 1) for g in gens])

    run_mean = em_gens.cumsum(dim=0) / torch.arange(1, len(gens)+1).unsqueeze(1)
    sim_prev = util.cos_sim(em_gens, run_mean).diagonal().cpu().numpy()

    features = [
        cos_sum,
        prob_g,
        pol_shift,
        len_ratio,
        sim_prev,
    ]

    # -- optional graph stats -------------------------------------------
    if include_graph:
        sim_G = util.cos_sim(em_gens, em_gens).cpu().numpy()
        np.fill_diagonal(sim_G, 0.0)
        features.append(_graph_features(sim_G))

    X = np.column_stack(features)
    y       = cos_seed * prob_g
    return X, y


# ---------------------------------------------------------------------------
# 3. Public API — single or batch
# ---------------------------------------------------------------------------

def build_gen_features(
    seeds: Union[str, Sequence[str]],
    summaries: Union[str, Sequence[str]],
    gens_nested: Union[Sequence[str], Sequence[Sequence[str]]],
    *,
    prob_negative: Callable[[str], float],
    include_graph: bool = True,
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute features/targets; supports nested gens and graph stats.

    Parameters
    ----------
    include_graph : bool, default True
        If True, append 4 graph‑theoretic statistics per generation.
    """
    # -------- single‑block --------------------------------------------
    if isinstance(seeds, str):
        if isinstance(gens_nested, Sequence) and (len(gens_nested) == 0 or isinstance(gens_nested[0], str)):
            return _features_single(seeds, summaries, gens_nested, prob_negative=prob_negative, include_graph=include_graph, device=device)
        raise ValueError("Single‑block mode expects a flat list of strings for `gens_nested`.")

    # -------- batch mode ----------------------------------------------
    seeds_list     = list(seeds)
    summaries_list = list(summaries)
    gens_blocks    = list(gens_nested)

    if not (len(seeds_list) == len(summaries_list) == len(gens_blocks)):
        raise ValueError("`seeds`, `summaries`, `gens_nested` must align in length.")

    X_parts, y_parts = [], []
    for s, summ, gens in zip(seeds_list, summaries_list, gens_blocks):
        X_i, y_i = _features_single(s, summ, gens, prob_negative=prob_negative, include_graph=include_graph, device=device)
        X_parts.append(X_i)
        y_parts.append(y_i)

    return np.vstack(X_parts), np.hstack(y_parts)


def train_predictor(X, y):


  alphas = np.logspace(-4, 4, 20)

  # Define models
  models = {
      "Linear Regression": LinearRegression(),
      "Polynomial Ridge Regression (deg=2)": make_pipeline(
      PolynomialFeatures(degree=2),
      RidgeCV(alphas=alphas, cv=5) ), # 5-fold CV
      "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
  }

  # Fit models
  for name, model in models.items():
      model.fit(X, y)

  return models


##### Computing the score ####

def compute_S_i(
    A_hat_i: np.ndarray,
    A_i: np.ndarray,
    lam: float,
    rho: int
) -> float:
    # Sort all unique thresholds (descending) from predicted scores
    thresholds = np.unique(A_hat_i)[::1]

    for s in thresholds:
        # Select samples G_ijk with score ≥ s
        selected = (A_hat_i >= s)
        selected_gt = A_i[selected]

        # Count how many are false positives (i.e., A_ijk < λ)
        false_positives = np.sum(selected_gt < lam)

        if false_positives <= rho:
            return s

    # If no threshold satisfies the constraint
    return float("inf")
