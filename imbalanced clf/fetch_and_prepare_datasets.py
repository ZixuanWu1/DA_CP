import os
import re
import argparse
import zipfile
import io
import json
import subprocess
from pathlib import Path
from typing import Tuple, Dict, List, Sequence
from sklearn.datasets import fetch_openml

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo




# -----------------------
# Utils
# -----------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def download_file(url: str, dest: Path):
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dest.write_bytes(data)

def unzip_to(path_zip: Path, out_dir: Path):
    with zipfile.ZipFile(path_zip, "r") as zf:
        zf.extractall(out_dir)

def to_bool(x):
    if isinstance(x, str):
        return x.lower() in {"1","true","t","yes","y","m","malignant","fraud","positive","pos"}
    return bool(x)

def drop_obvious_ids(df: pd.DataFrame) -> pd.DataFrame:
    ID_LIKE = {"id", "ID", "Id", "Time", "time", "index", "Unnamed: 32"}
    drop_cols = []
    for c in df.columns:
        if c in ID_LIKE or re.fullmatch(r"id|^id_.*|.*_id$", c, flags=re.I):
            drop_cols.append(c)
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# -----------------------
# Downloaders
# -----------------------
def get_ieee_cis_raw(out_dir: Path) -> Dict[str, Path]:
    """Download IEEE-CIS via Kaggle CLI. Returns dict of csv paths."""
    comp = "ieee-fraud-detection"
    od = ensure_dir(out_dir / "ieee_cis" / "raw")
    zip_path = od / "ieee.zip"
    if not zip_path.exists():
        print("[IEEE-CIS] Downloading via Kaggle CLI...")
        # kaggle competitions download -c ieee-fraud-detection -p <dir>
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(od)],
            check=True
        )
    # The download is multiple zip files; unzip all *.zip inside od
    for z in od.glob("*.zip"):
        unzip_to(z, od)
    paths = {
        "train": next(od.glob("train_transaction.csv")),
        "train_id": next(od.glob("train_identity.csv")),
        "test": next(od.glob("test_transaction.csv")),
        "test_id": next(od.glob("test_identity.csv")),
    }
    return paths

def get_mammography_raw(out_dir: Path) -> Path:
    """
    UCI Mammographic Mass Data Set (mammographic_masses.data).
    Columns: BI-RADS, Age, Shape, Margin, Density, Severity (label).
    """
    od = ensure_dir(out_dir / "mammography" / "raw")
    data_path = od / "mammographic_masses.data"
    if not data_path.exists():
        print("[Mammography] Downloading...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
        download_file(url, data_path)
    return data_path

def get_pima_raw(out_dir: Path) -> Path:
    """
    Pima Indians Diabetes (CSV mirror).
    Columns commonly: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
    BMI, DiabetesPedigreeFunction, Age, Outcome (label)
    """
    od = ensure_dir(out_dir / "pima" / "raw")
    data_path = od / "pima-indians-diabetes.csv"
    if not data_path.exists():
        print("[Pima] Downloading...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        download_file(url, data_path)
    return data_path

def get_breast_cancer_wisconsin_raw(out_dir: Path) -> Path:
    """
    Breast Cancer Wisconsin (Diagnostic) CSV (UCI mirror).
    Columns: id, diagnosis (M/B label), 30 features...
    """
    od = ensure_dir(out_dir / "breast_cancer_wisconsin" / "raw")
    data_path = od / "data.csv"
    if not data_path.exists():
        print("[BreastCancer] Downloading...")
        url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris-breast-cancer-wisconsin.csv"
        # Above is a common mirror; if it breaks, replace with a working CSV URL.
        download_file(url, data_path)
    return data_path

def get_statlog_shuttle_ucimlrepo_raw(out_dir: Path) -> Path:
    """
    UCI Statlog (Shuttle) via ucimlrepo (id=148).
    Saves a single raw CSV containing features + target.
    """
    od = ensure_dir(out_dir / "statlog_shuttle" / "raw")
    csv_path = od / "statlog_shuttle_ucimlrepo_raw.csv"
    if not csv_path.exists():
        print("[Statlog Shuttle] Fetching via ucimlrepo (id=148)...")
        ds = fetch_ucirepo(id=148)
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(csv_path, index=False)
    return csv_path


def get_connect4_ucimlrepo_raw(out_dir: Path) -> Path:
    """
    UCI Connect-4 via ucimlrepo (id=26).
    Saves a single raw CSV containing features + target.
    """
    od = ensure_dir(out_dir / "connect4" / "raw")
    csv_path = od / "connect4_ucimlrepo_raw.csv"
    if not csv_path.exists():
        print("[Connect-4] Fetching via ucimlrepo (id=26)...")
        ds = fetch_ucirepo(id=26)
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(csv_path, index=False)
    return csv_path

def get_maternal_health_ucimlrepo_raw(out_dir: Path) -> Path:
    """
    UCI id=863: Maternal Health Risk.
    Saves a single raw CSV containing features + target.
    """
    od = ensure_dir(out_dir / "maternal_health_risk" / "raw")
    csv_path = od / "maternal_health_risk_ucimlrepo_raw.csv"
    if not csv_path.exists():
        print("[Maternal Health Risk] Fetching via ucimlrepo (id=863)...")
        ds = fetch_ucirepo(id=863)
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(csv_path, index=False)
    return csv_path

def get_mi_complications_ucimlrepo_raw(out_dir: Path, target_col: str = "RAZRIV") -> Path:
    """
    UCI id=579: Myocardial infarction complications.
    Saves a task-specific raw CSV: features + ONE target column.
    """
    od = ensure_dir(out_dir / "mi_complications" / "raw")
    csv_path = od / f"mi_complications_{target_col.lower()}_ucimlrepo_raw.csv"

    if not csv_path.exists():
        print(f"[MI Complications] Fetching via ucimlrepo (id=579), target={target_col}...")
        ds = fetch_ucirepo(id=579)
        X = ds.data.features
        Y = ds.data.targets

        if target_col not in Y.columns:
            raise ValueError(f"target_col='{target_col}' not found. Available targets: {list(Y.columns)[:20]} ...")

        y = Y[target_col].rename(target_col)
        df = pd.concat([X, y], axis=1)
        df.to_csv(csv_path, index=False)

    return csv_path

def get_diabetes130_ucimlrepo_raw(out_dir: Path) -> Path:
    """
    UCI id=296: Diabetes 130-US hospitals (1999-2008).
    Saves raw CSV containing features + target 'readmitted'.
    """
    od = ensure_dir(out_dir / "diabetes130" / "raw")
    csv_path = od / "diabetes130_ucimlrepo_raw.csv"
    if not csv_path.exists():
        print("[Diabetes130] Fetching via ucimlrepo (id=296)...")
        ds = fetch_ucirepo(id=296)
        X = ds.data.features
        y = ds.data.targets  # has column 'readmitted'
        df = pd.concat([X, y], axis=1)
        df.to_csv(csv_path, index=False)
    return csv_path

def get_mi_complications_full_raw(out_dir: Path) -> Path:
    """
    Retrieves the FULL MI Complications dataset using fetch_ucirepo.
    We access 'ds.data.original' to ensure we get all 124 columns (Admission + Days 1-3 + Targets),
    avoiding the default split which might exclude intermediate monitoring features.
    """
    od = ensure_dir(out_dir / "mi_complications" / "raw")
    csv_path = od / "mi_complications_full_raw.csv"

    # Targets required for Multi-label task (Section 2 of Descriptive Statistics) [cite: 3]
    required_targets = [
        "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", 
        "A_V_BLOK", "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN", 
        "REC_IM", "P_IM_STEN"
    ]

    # 1. Check if valid file exists
    if csv_path.exists():
        try:
            df_check = pd.read_csv(csv_path, nrows=5)
            # Verify we have the target columns
            if all(col in df_check.columns for col in required_targets):
                return csv_path
            print("[MI Complications] Cached file missing required columns. Re-fetching...")
            csv_path.unlink()
        except:
            csv_path.unlink()

    print(f"[MI Complications] Fetching full dataset via fetch_ucirepo(id=579)...")
    
    # 2. Fetch using ucimlrepo
    try:
        ds = fetch_ucirepo(id=579)
        
        # KEY FIX: Use .original to get the full 124-column dataset
        if ds.data.original is not None:
            df = ds.data.original
        else:
            # Fallback: Reconstruct if original is missing (unlikely for this dataset ID)
            print("[Warning] ds.data.original is None. Reconstructing from ids+features+targets...")
            parts = []
            if ds.data.ids is not None: parts.append(ds.data.ids)
            if ds.data.features is not None: parts.append(ds.data.features)
            if ds.data.targets is not None: parts.append(ds.data.targets)
            df = pd.concat(parts, axis=1)

        # 3. Verification
        missing = [c for c in required_targets if c not in df.columns]
        if missing:
            raise ValueError(
                f"Fetched data is missing columns: {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )

        df.to_csv(csv_path, index=False)
        print(f"[MI Complications] Saved full raw data to {csv_path.name}")
        return csv_path

    except Exception as e:
        print(f"[Error] fetch_ucirepo failed: {e}")
        raise
    
# -----------------------
# Standardizers
# -----------------------
ID_LIKE = {"id", "ID", "Id", "Time", "time", "index", "Unnamed: 32"}

def drop_obvious_ids(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for c in df.columns:
        if c in ID_LIKE or re.fullmatch(r"id|^id_.*|.*_id$", c, flags=re.I):
            drop_cols.append(c)
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def one_hot(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if c not in exclude and df[c].dtype == "object"]
    if not cat_cols:
        return df
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc = ohe.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    return pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

def detect_label(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    Heuristics by common label names and types. Returns (label_name, y01).
    """
    candidates = [
        "isFraud", "Class", "Severity", "Outcome", "diagnosis", "target", "label", "Label"
    ]
    for c in candidates:
        if c in df.columns:
            y = df[c]
            # Map to 0/1
            if y.dtype == "O":
                y01 = y.map(lambda s: 1 if str(s).strip().upper() in {"1","M","MALIGNANT","YES","TRUE","FRAUD","POS"} else 0)
            else:
                # numeric → threshold at >0 or ==1
                if set(pd.unique(y.dropna())) <= {0,1}:
                    y01 = y.astype(int)
                else:
                    y01 = (y > 0).astype(int)
            return c, y01
    # fallback: last column
    c = df.columns[-1]
    y = df[c]
    if y.dtype == "O":
        y01 = y.map(lambda s: 1 if str(s).strip().upper() in {"1","M","MALIGNANT","YES","TRUE","FRAUD","POS"} else 0)
    else:
        y01 = (y > 0).astype(int)
    return c, y01

def clean_mammography(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    cols = ["BI_RADS", "Age", "Shape", "Margin", "Density", "Severity"]
    df = pd.read_csv(path, header=None, names=cols, na_values="?")
    df = df.dropna().reset_index(drop=True)
    label_col, y01 = "Severity", df["Severity"].astype(int)  # 0 = benign, 1 = malignant in UCI
    df = df.drop(columns=[label_col])
    df = one_hot(df, exclude=[])
    df = drop_obvious_ids(df)
    df["y"] = y01
    X_cols = [c for c in df.columns if c != "y"]
    clean_path = ensure_dir(out_dir / "mammography").joinpath("mammography_clean.csv")
    df.to_csv(clean_path, index=False)
    report = (label_col, X_cols)
    return df, report[0], report[1]

def clean_pima(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    df = pd.read_csv(path, header=None, names=cols)
    label_col, y01 = "Outcome", df["Outcome"].astype(int)
    df = df.drop(columns=[label_col])
    df = drop_obvious_ids(df)
    df["y"] = y01
    X_cols = [c for c in df.columns if c != "y"]
    clean_path = ensure_dir(out_dir / "pima").joinpath("pima_clean.csv")
    df.to_csv(clean_path, index=False)
    return df, label_col, X_cols

def clean_breast_cancer(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(path)
    # Try common schema: id, diagnosis (M/B), then features
    if "diagnosis" not in df.columns:
        # Try flexible match
        cand = [c for c in df.columns if c.lower() == "diagnosis"]
        if cand:
            df.rename(columns={cand[0]: "diagnosis"}, inplace=True)

    # drop obvious ids
    df = drop_obvious_ids(df)

    label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[0]
    y = df[label_col]
    if y.dtype == "O":
        y01 = y.map(lambda s: 1 if str(s).strip().upper().startswith("M") else 0)
    else:
        y01 = (y > 0).astype(int)

    df = df.drop(columns=[label_col])
    df = one_hot(df, exclude=[])
    df["y"] = y01
    X_cols = [c for c in df.columns if c != "y"]
    clean_path = ensure_dir(out_dir / "breast_cancer_wisconsin").joinpath("bcw_clean.csv")
    df.to_csv(clean_path, index=False)
    return df, label_col, X_cols



def clean_ieee(
    train_path: str | Path,
    out_dir: str | Path,
    na_col_thresh: float = 0.80,   # drop columns with >80% missing
    chunksize: int = 500_000,
    join_identity: bool = False,   # optional: join train_identity on TransactionID
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Best-practice IEEE-CIS cleaning, but label column is 'Class' (not 'y').
    """
    train_path = Path(train_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ieee_cis_clean.csv"

    df_tx = pd.read_csv(train_path)
    start_rows, start_cols = df_tx.shape
    print(f"[START] train_transaction: rows={start_rows:,}, cols={start_cols:,}")

    if join_identity:
        id_path = train_path.parent / "train_identity.csv"
        if id_path.exists():
            df_id = pd.read_csv(id_path)
            keep_id_cols = ["TransactionID"] + [c for c in df_id.columns if re.fullmatch(r"M\d+", c)]
            keep_id_cols += [c for c in df_id.columns if pd.api.types.is_numeric_dtype(df_id[c])]
            keep_id_cols = list(dict.fromkeys(keep_id_cols))
            df_id_small = df_id.loc[:, [c for c in keep_id_cols if c in df_id.columns]]
            before_join = df_tx.shape[0]
            df_tx = df_tx.merge(df_id_small, on="TransactionID", how="left")
            print(f"[Join] identity rows= {len(df_id):,}, kept cols= {len(df_id_small.columns):,}. "
                  f"Rows unchanged after join: {before_join == len(df_tx)}")
        else:
            print("[Join] train_identity.csv not found; skipping join.")

    cols = df_tx.columns.tolist()
    if "isFraud" not in cols:
        raise ValueError("Expected 'isFraud' in train_transaction.csv")
    if "TransactionAmt" in cols:
        amt_col = "TransactionAmt"
    elif "TransactionAmount" in cols:
        amt_col = "TransactionAmount"
    else:
        raise ValueError("Expected TransactionAmt/TransactionAmount in train_transaction.csv")

    # rename amount & label → Class
    df_tx = df_tx.rename(columns={amt_col: "Amount"})
    df_tx["Class"] = df_tx["isFraud"].astype("int8")  # << label is 'Class'

    # Keep numeric + binary-looking object columns
    keep_numeric = [c for c in df_tx.columns if pd.api.types.is_numeric_dtype(df_tx[c])]
    bin_obj_cols = []
    for c in df_tx.columns:
        if c in {"isFraud", "Class"}:
            continue
        if df_tx[c].dtype == "O":
            vals = pd.unique(df_tx[c].dropna().astype(str).str.strip())
            if len(vals) == 2:
                bin_obj_cols.append(c)

    to_drop_objects = [c for c in df_tx.columns if (df_tx[c].dtype == "O" and c not in bin_obj_cols)]
    df_tx = df_tx.drop(columns=to_drop_objects, errors="ignore")

    for c in bin_obj_cols:
        vals = pd.unique(df_tx[c].dropna().astype(str).str.strip())
        if len(vals) == 2:
            lo, hi = sorted(vals)
            mapping = {lo: 0, hi: 1}
            df_tx[c] = df_tx[c].astype(str).str.strip().map(mapping).astype("float32")

    print(f"[Keep] numeric={len(keep_numeric):,}, "
          f"binary_obj_mapped={len(bin_obj_cols):,}, "
          f"dropped_object_cols={len(to_drop_objects):,}")

    numeric_cols = [c for c in df_tx.columns if pd.api.types.is_numeric_dtype(df_tx[c])]
    v_cols = [c for c in df_tx.columns if re.fullmatch(r"V\d+", c)]
    must_keep = {"Amount", "Class"}
    numeric_cols = list(dict.fromkeys([*numeric_cols, *v_cols, *must_keep]))

    df_tx = df_tx.loc[:, [c for c in numeric_cols if c in df_tx.columns]]
    print(f"[Columns] after numeric+binary selection: cols={df_tx.shape[1]:,}")

    # Missingness: exclude 'Class' from the decision
    feats = [c for c in df_tx.columns if c != "Class"]
    miss_ratio = df_tx[feats].isna().mean()
    keep_feats = [c for c in feats if miss_ratio[c] <= (1.0 - na_col_thresh)]
    drop_feats = [c for c in feats if c not in keep_feats]
    df_tx = df_tx.loc[:, keep_feats + ["Class"]]
    print(f"[Col NA] threshold={na_col_thresh:.0%} keep={len(keep_feats):,}, drop={len(drop_feats):,}")

    # Drop rows with any NA (after pruning sparse columns)
    before_rows = len(df_tx)
    df_tx = df_tx.dropna(axis=0, how="any")
    removed_rows = before_rows - len(df_tx)
    print(f"[Row NA] removed={removed_rows:,} ({removed_rows / before_rows:.2%}) → rows={len(df_tx):,}")

    # Final tidy types
    df_tx["Class"] = df_tx["Class"].astype("int8")
    if "Amount" in df_tx.columns:
        df_tx["Amount"] = pd.to_numeric(df_tx["Amount"], errors="coerce").astype("float32")

    for c in df_tx.columns:
        if c != "Class" and pd.api.types.is_float_dtype(df_tx[c]):
            df_tx[c] = df_tx[c].astype("float32")
        if c != "Class" and pd.api.types.is_integer_dtype(df_tx[c]):
            if df_tx[c].dropna().isin({0, 1}).all():
                df_tx[c] = df_tx[c].astype("int8")
            else:
                df_tx[c] = df_tx[c].astype("int16")

    # Order columns: Class first, then Amount, then V*, then others
    ordered = ["Class"] + [c for c in ["Amount"] if c in df_tx.columns]
    ordered += sorted([c for c in df_tx.columns if re.fullmatch(r"V\d+", c)], key=lambda x: int(x[1:]))
    ordered += [c for c in df_tx.columns if c not in set(ordered)]
    df_tx = df_tx.loc[:, ordered]

    # Sanity checks
    assert start_rows > 0 and start_cols > 0, "Empty input?"
    assert "Class" in df_tx.columns, "Label column missing"
    assert "Amount" in df_tx.columns, "Amount missing"
    assert set(df_tx.dtypes.unique()) <= {np.dtype("int8"), np.dtype("int16"), np.dtype("float32")}, \
        f"Unexpected dtypes: {df_tx.dtypes.value_counts()}"

    df_tx.to_csv(out_csv, index=False)
    print(f"[DONE] Saved → {out_csv} with rows={len(df_tx):,}, cols={df_tx.shape[1]:,}")

    sample = df_tx.head(5).copy()
    feature_cols = [c for c in df_tx.columns if c != "Class"]
    return sample, "Class", feature_cols

def clean_ieee2(
    train_path: str | Path,
    out_dir: str | Path,
    na_col_thresh: float = 0.80,   # drop columns with >80% missing
    join_identity: bool = True,    # notebooks often merge identity
    topk_email: int = 20,          # keep top-K email domains, rest -> "other"
    topk_deviceinfo: int = 30,     # keep top-K device info, rest -> "other"
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Notebook-style IEEE-CIS cleaner:
      - Keep feature families: card*, addr*, dist*, C*, D*, M*, V*, ProductCD, DeviceType, DeviceInfo, P_emaildomain, R_emaildomain
      - Drop ID-ish columns (TransactionID, *id*, etc.)
      - Drop Amount
      - Label named 'y' from isFraud
      - One-hot a small set of categoricals with rare-category bucketing
    """
    train_path = Path(train_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ieee_cis_clean.csv"

    # --- Load transaction and (optionally) identity ---
    tx = pd.read_csv(train_path)
    if "isFraud" not in tx.columns:
        raise ValueError("Expected 'isFraud' in train_transaction.csv")
    if join_identity:
        id_path = train_path.parent / "train_identity.csv"
        if id_path.exists():
            ident = pd.read_csv(id_path)
            # Only keep a small number of low-cardinality, useful identity cols
            keep_id_cats = [
                "DeviceType", "DeviceInfo"
            ]
            keep_id_nums = [c for c in ident.columns if c.startswith("id_") and ident[c].dtype != "O"]
            ident_small = ident[["TransactionID"] + [c for c in keep_id_cats if c in ident.columns] + keep_id_nums]
            tx = tx.merge(ident_small, on="TransactionID", how="left")
        # else: silently skip

    # --- Build allowlist of feature groups (typical notebook set) ---
    keep_groups = []
    keep_groups += [c for c in tx.columns if c.startswith("card")]
    keep_groups += [c for c in tx.columns if c.startswith("addr")]
    keep_groups += [c for c in tx.columns if c.startswith("dist")]
    keep_groups += [c for c in tx.columns if re.fullmatch(r"C\d+", c)]
    keep_groups += [c for c in tx.columns if re.fullmatch(r"D\d+", c)]
    keep_groups += [c for c in tx.columns if re.fullmatch(r"M\d+", c)]
    keep_groups += [c for c in tx.columns if re.fullmatch(r"V\d+", c)]

    # small, important categoricals used in many notebooks
    small_cats = [c for c in ["ProductCD", "P_emaildomain", "R_emaildomain"] if c in tx.columns]
    # identity categoricals (if joined)
    id_cats = [c for c in ["DeviceType", "DeviceInfo"] if c in tx.columns]

    # label
    y = tx["isFraud"].astype("int8").rename("y")

    # --- Drop IDs and Amount ---
    drop_id_like = {"TransactionID"}
    drop_id_like |= {c for c in tx.columns if re.search(r"(^|_)id(_|$)", c, flags=re.I)}  # drops id_*, *_id*
    drop_cols = (drop_id_like | {"isFraud", "Amount"})
    keep_cols = sorted(set(keep_groups + small_cats + id_cats))
    keep_cols = [c for c in keep_cols if c not in drop_cols]  # ensure

    # --- Slice and handle missingness columnwise ---
    X = tx.loc[:, keep_cols].copy()

    # Drop columns with too much missing
    miss_ratio = X.isna().mean()
    keep_cols2 = [c for c in keep_cols if miss_ratio[c] <= (1.0 - na_col_thresh)]
    X = X.loc[:, keep_cols2]

    # --- Rare-category bucketing + one-hot for small categoricals ---
    def topk_bucket(s: pd.Series, k: int):
        vc = s.value_counts(dropna=True)
        keep_vals = set(vc.head(k).index.tolist())
        return s.where(s.isin(keep_vals), other="other")

    # Apply top-k bucketing
    cat_cols = [c for c in (small_cats + id_cats) if c in X.columns]
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype("string")
        if "P_emaildomain" in X.columns:
            X["P_emaildomain"] = topk_bucket(X["P_emaildomain"], topk_email)
        if "R_emaildomain" in X.columns:
            X["R_emaildomain"] = topk_bucket(X["R_emaildomain"], topk_email)
        if "DeviceInfo" in X.columns:
            X["DeviceInfo"] = topk_bucket(X["DeviceInfo"], topk_deviceinfo)

        # One-hot just these few cats; keep numeric as-is
        X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns], dummy_na=False)

    # --- Coerce numerics compactly; drop remaining-NA rows (after OHE there should be fewer) ---
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            # should be only OHE columns left as uint8
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # Cast dtypes for compactness
    for c in X.columns:
        if pd.api.types.is_integer_dtype(X[c]):
            # one-hot becomes uint8/int64; compress
            if X[c].min() >= 0 and X[c].max() <= 255:
                X[c] = X[c].astype("uint8")
            else:
                X[c] = X[c].astype("int16")
        elif pd.api.types.is_float_dtype(X[c]):
            X[c] = X[c].astype("float32")

    # Align y
    y = y.loc[X.index].astype("int8")

    # Final frame
    df_out = pd.concat([X, y], axis=1)
    df_out.to_csv(out_csv, index=False)

    feature_cols = [c for c in df_out.columns if c != "y"]
    sample = df_out.head(5).copy()
    print(f"[DONE] Saved → {out_csv} with rows={len(df_out):,}, cols={len(df_out.columns):,}")
    return sample, "y", feature_cols

def get_thyroid_openml_raw(out_dir: Path) -> Path:
    """
    OpenML 'thyroid' (id=38). We'll save the raw frame to CSV.
    Target column typically 'Class' with values like 'negative', 'hyperthyroid', ...
    """
    od = ensure_dir(out_dir / "thyroid" / "raw")
    csv_path = od / "thyroid_openml.csv"
    if not csv_path.exists():
        print("[Thyroid] Downloading from OpenML (id=38)...")
        ds = fetch_openml(data_id=38, as_frame=True)
        df = ds.frame
        df.to_csv(csv_path, index=False)
    return csv_path
import pandas as pd
from pathlib import Path
from typing import Tuple, List

TRUTHY = {"1","t","true","yes","y","pos","positive","malignant","hyperthyroid","hyper","fraud"}
FALSY  = {"0","f","false","no","n","neg","negative","benign","hypothyroid","hypo","non-fraud"}

def _is_binary_series(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna().astype(str).str.strip())
    return len(vals) == 2

def _map_binary_series(s: pd.Series, colname: str) -> pd.Series:
    """Map a binary categorical series to {0,1} with sensible defaults."""
    x = s.astype(str).str.strip()
    xl = x.str.lower()

    # Special-case sex
    if colname.lower() in {"sex","gender"}:
        return xl.map(lambda v: 1 if v in {"m","male"} else (0 if v in {"f","female"} else pd.NA)).astype("Int8")

    # Truthy/Falsy heuristics
    uniq = pd.unique(xl.dropna())
    if set(uniq) <= TRUTHY.union(FALSY):
        return xl.map(lambda v: 1 if v in TRUTHY else (0 if v in FALSY else pd.NA)).astype("Int8")

    # Fallback: deterministic mapping by sorted order
    # larger (lexicographically) → 1, smaller → 0
    u_sorted = sorted([u for u in uniq if pd.notna(u)])
    if len(u_sorted) == 2:
        lo, hi = u_sorted[0], u_sorted[1]
        print(f"[thyroid] Binary column '{colname}': mapping '{hi}'→1, '{lo}'→0 (fallback).")
        return xl.map({hi: 1, lo: 0}).astype("Int8")

    # Not binary; return as-is
    return s

def clean_thyroid(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Start-from-scratch cleaner for OpenML Thyroid (id=38):
      - y := 1{Class != 'negative'}
      - remove referral_source (or any column starting with it)
      - drop sex_nan if present
      - for ANY binary categorical feature, convert to {0,1}
      - keep non-binary categoricals as-is (no OHE)
    """
    df_raw = pd.read_csv(path)

    # ----- Label y -----
    label_col = "Class" if "Class" in df_raw.columns else df_raw.columns[-1]
    y_raw = df_raw[label_col]
    if y_raw.dtype == "O":
        y = (
            y_raw.fillna("negative").astype(str).str.strip().str.lower()
               .map(lambda s: 0 if s == "negative" else 1)
               .astype("Int8")
        )
    else:
        y = (y_raw.astype(float) != 0).astype("Int8")

    X = df_raw.drop(columns=[label_col])

    # ----- Drop referral_source*, sex_nan, TBG and TBG_measured -----
    drop_cols = [
        c for c in X.columns
        if c.lower() == "referral_source" or c.lower().startswith("referral_source")
        or c.lower() == "tbg" or c.lower() == "tbg_measured"
        or c.lower() == "sex_nan"
    ]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # ----- Convert binary categoricals to 0/1 -----
    for c in X.columns:
        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(X[c]):
            continue

        # Treat object/categorical
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c]):
            if _is_binary_series(X[c]):
                X[c] = _map_binary_series(X[c], c)
            else:
                # leave multi-class categoricals unchanged
                pass


   # ----- Coerce obvious numeric columns -----
    for num_like in ["age","TSH","T3","TT4","T4U","FTI"]:
        if num_like in X.columns:
            X[num_like] = pd.to_numeric(X[num_like], errors="coerce")

    # (NEW) If any columns are still object after all conversions, try numeric coerce where possible
    for c in X.columns:
        if X[c].dtype == object:
            # try to coerce numerics hidden as strings; if not numeric, leave as-is
            X[c] = pd.to_numeric(X[c], errors="ignore")

    # (NEW) Combine features + label FIRST, then drop NaNs consistently
    df_all = X.copy()
    df_all["y"] = y

    # (NEW) Normalize inf -> NaN, then drop rows having ANY NaN in any column
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # OPTIONAL: enforce pure numeric features for methods like KNN
    X = df_all.drop(columns=["y"])
    # cast Int8 binaries to plain int8 (no NA remains after dropna)
    for c in X.columns:
        if str(X[c].dtype) == "Int8":
            X[c] = X[c].astype("int8")

    y = df_all["y"].astype("int8")

    # Save
    out = Path(out_dir) / "thyroid"
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "thyroid_clean.csv"
    pd.concat([X, y], axis=1).to_csv(save_path, index=False)

    X_cols = [c for c in X.columns]
    print(f"[Thyroid] Cleaned: X={len(X_cols)} features, n={len(X)} rows → {save_path.name}")
    return pd.concat([X, y], axis=1), "y", X_cols


def get_phoneme_openml_raw(out_dir: Path) -> Path:
    """
    OpenML Phoneme (id=1489). Target column is 'class' (nasal/oral or 0/1 depending on version).
    """
    od = ensure_dir(out_dir / "phoneme" / "raw")
    csv_path = od / "phoneme_openml.csv"
    if not csv_path.exists():
        print("[Phoneme] Downloading from OpenML (id=1489)...")
        ds = fetch_openml(data_id=1489, as_frame=True)
        df = ds.frame
        df.to_csv(csv_path, index=False)
    return csv_path

def clean_phoneme(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(path)
    # Target column
    label_col = "class" if "class" in df.columns else df.columns[-1]
    y = df[label_col]

    # --- Map label to 0/1 ---
    if y.dtype == "O":
        s = y.astype(str).str.lower().str.strip()
        if set(s.unique()) <= {"nasal", "oral"}:
            y01 = s.map({"oral": 0, "nasal": 1}).astype("int8")
        else:
            first = sorted(s.unique())[0]
            y01 = (s != first).astype("int8")
    else:
        # numeric: assume values {1,2} → convert to {0,1}
        uniq = sorted(pd.unique(y.dropna()))
        if set(uniq) == {1, 2}:
            y01 = (y.astype(int) - 1).astype("int8")   # maps 1→0, 2→1
        else:
            # fallback: treat >0 as 1
            y01 = (y.astype(float) > 0).astype("int8")

    # --- Features ---
    X = df.drop(columns=[label_col])
    X = drop_obvious_ids(X)
    # phoneme features are already numeric, but just in case:
    X = one_hot(X, exclude=[])
    X["y"] = y01
    X_cols = [c for c in X.columns if c != "y"]

    # Save
    clean_path = ensure_dir(out_dir / "phoneme").joinpath("phoneme_clean.csv")
    X.to_csv(clean_path, index=False)

    print(f"[Phoneme] Cleaned: {len(X)} rows, {len(X_cols)} features → phoneme_clean.csv")
    return X, "y", X_cols

def clean_statlog_shuttle_ucimlrepo(path: Path, out_dir: Path, pos_class: int = 1) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Clean Shuttle from ucimlrepo raw CSV:
      - y = 1{Class == pos_class} else 0
      - all features numeric
    Saves: out_dir/statlog_shuttle/statlog_shuttle_clean.csv
    """
    df = pd.read_csv(path)

    # target column name can vary; ucimlrepo typically uses something like 'class'
    # we’ll take the last column as target robustly
    label_col = df.columns[-1]
    y_raw = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")

    X = df.drop(columns=[label_col]).copy()
    X = drop_obvious_ids(X)

    # coerce all features numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")

    y = (y_raw == int(pos_class)).astype("int8")

    df_out = X.copy()
    df_out["y"] = y
    df_out["y"] = 1 - df_out["y"] # flip so that 1 be minority class
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)

    clean_path = ensure_dir(out_dir / "statlog_shuttle").joinpath("statlog_shuttle_clean.csv")
    df_out.to_csv(clean_path, index=False)

    X_cols = [c for c in df_out.columns if c != "y"]
    print(f"[Statlog Shuttle] Cleaned: rows={len(df_out):,}, X={len(X_cols)} → {clean_path.name}")
    return df_out, label_col, X_cols


def clean_connect4_ucimlrepo(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Clean Connect-4 from ucimlrepo raw CSV:
      - y = 1{Class == 'lose'} else 0
      - one-hot encode board-position features
    Saves: out_dir/connect4/connect4_clean.csv
    """
    df = pd.read_csv(path)

    label_col = df.columns[-1]
    y_raw = df[label_col].astype(str).str.strip().str.lower()
    y = (y_raw == "draw").astype("int8")

    X = df.drop(columns=[label_col]).copy()
    X = drop_obvious_ids(X)

    # one-hot all categorical positions
    X = one_hot(X, exclude=[])

    df_out = X.copy()
    df_out["y"] = y
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)

    clean_path = ensure_dir(out_dir / "connect4").joinpath("connect4_clean.csv")
    df_out.to_csv(clean_path, index=False)

    X_cols = [c for c in df_out.columns if c != "y"]
    print(f"[Connect-4] Cleaned: rows={len(df_out):,}, X={len(X_cols)} → {clean_path.name}")
    return df_out, label_col, X_cols


# ---------- MNIST 7 vs Others ----------
def get_mnist7_openml_raw(out_dir: Path) -> Path:
    """
    OpenML mnist_784 (id=554). Saves the raw frame to CSV.
    Columns: pixel1..pixel784, class (digits '0'..'9').
    """
    od = ensure_dir(out_dir / "mnist7" / "raw")
    csv_path = od / "mnist_784.csv"
    if not csv_path.exists():
        print("[MNIST7] Downloading from OpenML (mnist_784, id=554)...")
        ds = fetch_openml(name="mnist_784", version=1, as_frame=True)
        df = ds.frame  # 70k x 785
        df.to_csv(csv_path, index=False)
    return csv_path


def clean_mnist7(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Build a binary dataset: y=1 if class=='7' else 0. Keep 784 pixel features.
    Saves to data/mnist7/mnist7_clean.csv
    """
    df = pd.read_csv(path)
    if "class" not in df.columns:
        # OpenML uses 'class' for the digit label
        # if not present (unlikely), assume last column is label
        label_col = df.columns[-1]
    else:
        label_col = "class"

    # map to 0/1 (7 vs others)
    y = (df[label_col].astype(str).str.strip() == "7").astype("int8")

    # features: pixel1..pixel784 (already numeric strings)
    X = df.drop(columns=[label_col]).copy()

    # ensure numeric & compact dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    # drop any rows with NaN (should be none)
    X["y"] = y
    X = X.dropna(axis=0, how="any").reset_index(drop=True)

    # put y first, then pixels in order
    pixel_cols = [c for c in X.columns if c != "y"]
    ordered = ["y"] + pixel_cols
    X = X.loc[:, ordered]

    # save
    out = ensure_dir(out_dir / "mnist7")
    save_path = out / "mnist7_clean.csv"
    X.to_csv(save_path, index=False)

    print(f"[MNIST7] Cleaned: rows={len(X):,}, features={len(pixel_cols)} → {save_path.name}")
    return X, "y", pixel_cols



def clean_mnist49(
    path: Path,
    out_dir: Path,
    imb_rates: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.3,
                                  0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    random_state: int = 42
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Build 4-vs-9 binary dataset from mnist_784.
    y = 1 if digit == 4, y = 0 if digit == 9.
    
    Also generates multiple imbalanced datasets with ratio:
        (# of 4s) / (# of 9s) = imb_rate
    by downsampling the 4s.

    Saves:
      - mnist49/mnist49_clean_full.csv  (all 4s and 9s)
      - mnist49/imb_rate_<r>/mnist49_imb_<r>.csv  for each r in imb_rates
    """
    df = pd.read_csv(path)
    if "class" not in df.columns:
        label_col = df.columns[-1]
    else:
        label_col = "class"

    # keep only 4 and 9
    labels = df[label_col].astype(str).str.strip()
    df = df[labels.isin(["4", "9"])].copy()

    # map labels: 4 -> 1, 9 -> 0
    y = (df[label_col].astype(str).str.strip() == "4").astype("int8")

    # features: drop label column
    X = df.drop(columns=[label_col]).copy()

    # numeric & compact dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")

    X["y"] = y
    X = X.dropna(axis=0, how="any").reset_index(drop=True)

    # put y first
    pixel_cols = [c for c in X.columns if c != "y"]
    ordered = ["y"] + pixel_cols
    X = X.loc[:, ordered]

    # base output dir
    base_out = ensure_dir(out_dir / "mnist49")

    # save full clean dataset
    full_path = base_out / "mnist49_clean_full.csv"
    X.to_csv(full_path, index=False)
    print(f"[MNIST49] Full clean: rows={len(X):,}, features={len(pixel_cols)} → {full_path.name}")

    # split by class
    df_4 = X[X["y"] == 1].reset_index(drop=True)  # digit 4
    df_9 = X[X["y"] == 0].reset_index(drop=True)  # digit 9
    n4, n9 = len(df_4), len(df_9)
    print(f"[MNIST49] #4={n4}, #9={n9}, base ratio #4/#9={n4/n9:.3f}")

    rng = np.random.default_rng(random_state)

    # generate imbalanced versions
    for r in imb_rates:
        # desired # of 4s given all 9s kept
        n4_target = int(round(r * n9))
        n4_target = min(n4_target, n4)  # cannot exceed available 4s

        if n4_target == 0:
            print(f"[MNIST49] Skipping imb_rate={r}: target #4 is 0.")
            continue

        # sample 4s
        idx = rng.choice(n4, size=n4_target, replace=False)
        df_4_sub = df_4.iloc[idx]

        df_imb = pd.concat([df_9, df_4_sub], axis=0)
        df_imb = df_imb.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        out_subdir = ensure_dir(base_out / f"imb_rate_{r:.2f}")
        out_path = out_subdir / f"mnist49_imb_{r:.2f}.csv"
        df_imb.to_csv(out_path, index=False)

        n4_new = (df_imb["y"] == 1).sum()
        n9_new = (df_imb["y"] == 0).sum()
        print(f"[MNIST49] imb_rate={r:.2f}: #4={n4_new}, #9={n9_new}, ratio={n4_new/n9_new:.3f} → {out_path}")

    return X, "y", pixel_cols

def clean_maternal_health_ucimlrepo(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    UCI 863: y=1 if RiskLevel is high (otherwise 0).
    """
    df = pd.read_csv(path)

    label_col = "RiskLevel" if "RiskLevel" in df.columns else df.columns[-1]
    s = df[label_col].astype(str).str.strip().str.lower()

    # common strings in this dataset: "low risk", "mid risk", "high risk"
    y = s.map(lambda v: 1 if "high risk" in v else 0).astype("int8")

    X = df.drop(columns=[label_col]).copy()
    X = drop_obvious_ids(X)

    # ensure numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    df_out = X.copy()
    df_out["y"] = y
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)

    clean_path = ensure_dir(out_dir / "maternal_health_risk").joinpath("maternal_health_risk_clean.csv")
    df_out.to_csv(clean_path, index=False)

    X_cols = [c for c in df_out.columns if c != "y"]
    print(f"[Maternal Health Risk] label={label_col} (high vs rest), rows={len(df_out):,}, X={len(X_cols)} → {clean_path.name}")
    return df_out, "y", X_cols

def clean_ham10000_hmnist(path: Path, out_dir: Path, pos_class: int = 6) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    HAM10000 HMNIST (28x28 RGB) CSV:
      - Input: flattened pixels + 'label' in {0..6}
      - Output: y = 1{label == pos_class} else 0 (default: melanoma, label=6)
      - All features numeric (uint8 if possible)
    Saves: out_dir/ham10000_hmnist/ham10000_hmnist_clean.csv
    HAM10000_LABEL_MAP = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "nv",
    5: "vasc",
    6: "mel",
}
    """
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("Expected column 'label' in HMNIST CSV")

    # label -> binary y
    y_raw = pd.to_numeric(df["label"], errors="coerce")
    y = (y_raw == int(pos_class)).astype("int8")

    # features
    X = df.drop(columns=["label"]).copy()
    X = drop_obvious_ids(X)

    # ensure numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    df_out = X.copy()
    df_out["y"] = y

    # drop bad rows
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)

    # (optional) cast pixels compactly if they look like 0..255
    X_cols = [c for c in df_out.columns if c != "y"]
    try:
        mn = df_out[X_cols].min().min()
        mx = df_out[X_cols].max().max()
        if mn >= 0 and mx <= 255:
            df_out[X_cols] = df_out[X_cols].astype("uint8")
        else:
            df_out[X_cols] = df_out[X_cols].astype("float32")
    except Exception:
        df_out[X_cols] = df_out[X_cols].astype("float32")

    clean_path = ensure_dir(out_dir / "skin-cancer-mnist-ham10000").joinpath("ham10000_hmnist_clean.csv")
    df_out.to_csv(clean_path, index=False)

    X_cols = [c for c in df_out.columns if c != "y"]
    print(f"[HAM10000 HMNIST] pos_class={pos_class}, rows={len(df_out):,}, X={len(X_cols)} → {clean_path.name}")
    return df_out, "y", X_cols
    
def clean_mi_complication_ucimlrepo(
    path: Path,
    out_dir: Path,
    target_col: str = "RAZRIV"
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Clean MI complications (UCI 579):
      - y = 1{target_col == 1} else 0
      - numeric features only
      - impute ALL missing values in X using column-wise median
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path.name}")

    # --- label ---
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    y = (y_raw == 1).astype("int8")

    # --- features ---
    X = df.drop(columns=[target_col]).copy()
    X = drop_obvious_ids(X)

    # coerce numeric + normalize inf
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # --- median impute ALL missing in X ---
    med = X.median(numeric_only=True)   # Series indexed by columns
    X = X.fillna(med)

    # (optional) if any columns were entirely NaN, median is NaN -> fill those with 0
    X = X.fillna(0)

    df_out = X.copy()
    df_out["y"] = y

    clean_path = ensure_dir(out_dir / "mi_complications").joinpath(
        f"mi_complication_{target_col.lower()}_clean.csv"
    )
    df_out.to_csv(clean_path, index=False)

    X_cols = [c for c in df_out.columns if c != "y"]
    print(f"[MI Complication] target={target_col}, rows={len(df_out):,}, X={len(X_cols)} → {clean_path.name}")
    return df_out, "y", X_cols

def clean_diabetes130_ucimlrepo(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Diabetes130 cleaning:
      - drop columns with >50% missing
      - Yes/No -> 0/1 (binary only)
      - strings -> categorical
      - numeric columns: median impute if missing <=50%
    """
    df = pd.read_csv(path, low_memory=False)

    df = df.replace(["?", "MISSING", "Unknown/Invalid", "None"], np.nan)

    # --------------------
    # 1. Separate label
    # --------------------
    if "readmitted" not in df.columns:
        raise ValueError("Expected column 'readmitted' in Diabetes130 dataset")

    # --- filter target ---
    s = df["readmitted"].astype(str).str.strip()
    df = df[s.isin(["NO", "<30"])].copy()

    # --- label ---
    y = (df["readmitted"].astype(str).str.strip() == "<30").astype("int8")
    X = df.drop(columns=["readmitted"]).copy()
    X = drop_obvious_ids(X)

    # Drop obvious IDs if present
    X = X.drop(columns=[c for c in ["encounter_id", "patient_nbr"] if c in X.columns],
               errors="ignore")

    # --------------------
    # 2. Drop columns with >50% missing
    # --------------------
    miss_frac = X.isna().mean()
    drop_cols = miss_frac[miss_frac > 0.5].index.tolist()
    X = X.drop(columns=drop_cols)

    # --------------------
    # 3. Convert Yes/No → 0/1 (binary only)
    # --------------------
    for c in X.columns:
        if X[c].dtype == "object":
            vals = pd.unique(X[c].dropna().astype(str).str.strip())
            if set(vals).issubset({"Yes", "No"}) and len(vals) <= 2:
                X[c] = X[c].map({"No": 0, "Yes": 1}).astype("Int64")

    # --------------------
    # 4. Coerce numeric columns (where possible)
    # --------------------
    for c in X.columns:
        if X[c].dtype == "object":
            coerced = pd.to_numeric(X[c], errors="coerce")
            # accept numeric if most non-missing survive coercion
            nonmiss = X[c].notna().sum()
            if nonmiss > 0 and coerced.notna().sum() / nonmiss >= 0.9:
                X[c] = coerced

    # --------------------
    # 5. Median impute numeric columns (≤50% missing only)
    # --------------------
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if num_cols:
        med = X[num_cols].median(numeric_only=True)
        X[num_cols] = X[num_cols].fillna(med)

    # --------------------
    # 6. Cast remaining strings to categorical
    # --------------------
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")

    # --------------------
    # 7. Final dataset
    # --------------------
    df_out = X.copy()
    df_out["y"] = y.values

    out_path = ensure_dir(out_dir / "diabetes130") / "diabetes130_clean.csv"
    df_out.to_csv(out_path, index=False)

    X_cols = [c for c in df_out.columns if c != "y"]
    print(
        f"[Diabetes130] rows={len(df_out):,}, "
        f"X={len(X_cols)}, "
        f"dropped_cols={len(drop_cols)} → {out_path.name}"
    )

    return df_out, "y", X_cols

def clean_mi_complications_multilabel(path: Path, out_dir: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Cleans the MI Complications dataset for Multi-Label Classification.
    Uses 'Most Features' criteria: Admission + Day 1 + Day 2 + Day 3 inputs.
    """
    df = pd.read_csv(path)
    
    # Define Targets (11 Complications from Section 2) [cite: 3]
    # We exclude LET_IS (Lethal Outcome) from features, but don't predict it here.
    target_cols = [
        "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", 
        "A_V_BLOK", "OTEK_LANC", "RAZRIV", 'DRESSLER', "ZSN", 
        "REC_IM", "P_IM_STEN"
    ]
    
    # Validate
    missing = [t for t in target_cols if t not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required targets: {missing}")

    # 1. Prepare Y (Targets)
    Y = df[target_cols].copy()
    
    # Filter rows with complete target information
    valid_mask = Y.notna().all(axis=1)
    Y = Y[valid_mask].astype(int)
    
    # 2. Prepare X (Features)
    # Drop targets and lethal outcome from inputs
    drop_from_x = target_cols + ["LET_IS"]
    X = df.drop(columns=[c for c in drop_from_x if c in df.columns])
    X = X[valid_mask]
    
    X = drop_obvious_ids(X)

    # 3. Clean Features
    # Note: Day 3 pain relapse (R_AB_3_n) has ~86% missing[cite: 33].
    # Standard cleanup drops columns with >80% missing.
    thresh = 0.80
    miss_frac = X.isna().mean()
    drop_sparse = miss_frac[miss_frac > thresh].index.tolist()
    if drop_sparse:
        print(f"[MI Clean] Dropping sparse columns (>80% missing): {drop_sparse}")
        X = X.drop(columns=drop_sparse)
    
    # Numeric Coercion
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    
    # Median Imputation
    X = X.fillna(X.median())
    
    # 4. Final Assemble
    # Prefix targets with "target_"
    Y.columns = [f"target_{c}" for c in Y.columns]
    df_out = pd.concat([X, Y], axis=1)
    
    clean_path = ensure_dir(out_dir / "mi_complications").joinpath("mi_multilabel_clean.csv")
    df_out.to_csv(clean_path, index=False)
    
    print(f"[MI Multi-label] Cleaned data saved → {clean_path.name}")
    print(f"   Rows: {len(df_out)} | Features: {len(X.columns)} | Targets: {len(Y.columns)}")
    
    return df_out, list(Y.columns), list(X.columns)
# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Download & standardize imbalanced datasets")
    ap.add_argument("--out_dir", type=str, default="./data")
    ap.add_argument("--datasets", type=str, default="ieee,mammography,pima,bcw, thyroid",
                    help="comma list: ieee,mammography,pima,bcw, thyroid")
    ap.add_argument("--skip_download", action="store_true",
                    help="use existing files if already present")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    wanted = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}

    reports = {}

    # IEEE-CIS
    if "ieee" in wanted:
        try:
            if not args.skip_download:
                paths = get_ieee_cis_raw(out_dir)
            else:
                paths = {
                    "train": next((out_dir / "ieee_cis" / "raw").glob("train_transaction.csv")),
                    "train_id": next((out_dir / "ieee_cis" / "raw").glob("train_identity.csv")),
                }
            df, label_col, X_cols = clean_ieee(
                train_path=paths["train"],     # << just the CSV path
                out_dir=out_dir,
                chunksize=200_000
        )
            reports["ieee_cis"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[IEEE-CIS] label={label_col}, X={len(X_cols)} features, n={len(df)} → saved ieee_cis_clean.csv")
        except Exception as e:
            print(f"[IEEE-CIS] Skipped: {e}")
    
    # IEEE-CIS for general pipeline
    if "ieee2" in wanted:
        try:
            if not args.skip_download:
                paths = get_ieee_cis_raw(out_dir)
            else:
                paths = {
                    "train": next((out_dir / "ieee_cis" / "raw").glob("train_transaction.csv")),
                    "train_id": next((out_dir / "ieee_cis" / "raw").glob("train_identity.csv")),
                }
                df, label_col, X_cols = clean_ieee2(
                    train_path=paths["train"],
                    out_dir=out_dir,
                    join_identity=True  # set False if you don’t want to merge identity
                )

            # df_head is just a head(5) sample returned for preview; read the saved CSV if you need the full frame
            reports["ieee_cis"] = {
                "label": label_col,
                "n_samples": int(pd.read_csv(out_dir / "ieee_cis_clean.csv").shape[0]),
                "n_features": len(X_cols)
            }
            print(f"[IEEE-CIS] label={label_col}, X={len(X_cols)} features → saved ieee_cis_clean.csv")
        except Exception as e:
            print(f"[IEEE-CIS] Skipped: {e}")


    # Mammography
    if "mammography" in wanted:
        try:
            p = get_mammography_raw(out_dir) if not args.skip_download else out_dir / "mammography" / "raw" / "mammographic_masses.data"
            df, label_col, X_cols = clean_mammography(p, out_dir)
            reports["mammography"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[Mammography] label={label_col}, X={len(X_cols)}, n={len(df)} → mammography_clean.csv")
        except Exception as e:
            print(f"[Mammography] Skipped: {e}")

    # Pima
    if "pima" in wanted:
        try:
            p = get_pima_raw(out_dir) if not args.skip_download else out_dir / "pima" / "raw" / "pima-indians-diabetes.csv"
            df, label_col, X_cols = clean_pima(p, out_dir)
            reports["pima"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[Pima] label={label_col}, X={len(X_cols)}, n={len(df)} → pima_clean.csv")
        except Exception as e:
            print(f"[Pima] Skipped: {e}")

    # Breast Cancer Wisconsin
    if "bcw" in wanted:
        try:
            p = get_breast_cancer_wisconsin_raw(out_dir) if not args.skip_download else out_dir / "breast_cancer_wisconsin" / "raw" / "data.csv"
            df, label_col, X_cols = clean_breast_cancer(p, out_dir)
            reports["breast_cancer_wisconsin"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[BCW] label={label_col}, X={len(X_cols)}, n={len(df)} → bcw_clean.csv")
        except Exception as e:
            print(f"[BCW] Skipped: {e}")
    
    # Thyroid (OpenML)
    if "thyroid" in wanted:
        try:
            p = get_thyroid_openml_raw(out_dir) if not args.skip_download else out_dir / "thyroid" / "raw" / "thyroid_openml.csv"
            df, label_col, X_cols = clean_thyroid(p, out_dir)
            reports["thyroid"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[Thyroid] label=y, X={len(X_cols)}, n={len(df)} → thyroid_clean.csv")
        except Exception as e:
            print(f"[Thyroid] Skipped: {e}")

    # Phoneme (OpenML)
    if "phoneme" in wanted:
        try:
            p = get_phoneme_openml_raw(out_dir) if not args.skip_download else out_dir / "phoneme" / "raw" / "phoneme_openml.csv"
            df, label_col, X_cols = clean_phoneme(p, out_dir)
            reports["phoneme"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[Phoneme] label=y, X={len(X_cols)}, n={len(df)} → phoneme_clean.csv")
        except Exception as e:
            print(f"[Phoneme] Skipped: {e}")
    
       # MNIST 7 vs Others
    if "mnist7" in wanted:
        try:
            p = get_mnist7_openml_raw(out_dir) if not args.skip_download else out_dir / "mnist7" / "raw" / "mnist_784.csv"
            df, label_col, X_cols = clean_mnist7(p, out_dir)
            reports["mnist7"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[MNIST7] label=y, X={len(X_cols)}, n={len(df)} → mnist7_clean.csv")
        except Exception as e:
            print(f"[MNIST7] Skipped: {e}")
    
    if "mnist49" in wanted:
        try:
            p = get_mnist7_openml_raw(out_dir) if not args.skip_download else out_dir / "mnist7" / "raw" / "mnist_784.csv"
            raw_csv = get_mnist7_openml_raw(out_dir=Path("data"))   # same raw file
            df, label_col, X_cols = clean_mnist49(
                p,
                out_dir,
                imb_rates=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # customize if you want
            )
            reports["mnist49"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
            print(f"[MNIST49] label=y, X={len(X_cols)}, n={len(df)} → mnist49_clean.csv")
        except Exception as e:
            print(f"[MNIST49] Skipped: {e}")

    # Statlog Shuttle
    # Statlog Shuttle (ucimlrepo)
    if "shuttle" in wanted:
        try:
            p = get_statlog_shuttle_ucimlrepo_raw(out_dir) if not args.skip_download else out_dir / "statlog_shuttle" / "raw" / "statlog_shuttle_ucimlrepo_raw.csv"
            df, label_col, X_cols = clean_statlog_shuttle_ucimlrepo(p, out_dir, pos_class=1)
            reports["statlog_shuttle"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
        except Exception as e:
            print(f"[Statlog Shuttle] Skipped: {e}")

    # Connect-4 (ucimlrepo)
    if "connect4" in wanted:
        try:
            p = get_connect4_ucimlrepo_raw(out_dir) if not args.skip_download else out_dir / "connect4" / "raw" / "connect4_ucimlrepo_raw.csv"
            df, label_col, X_cols = clean_connect4_ucimlrepo(p, out_dir)
            reports["connect4"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
        except Exception as e:
            print(f"[Connect-4] Skipped: {e}")
    # Maternal Health Risk (ucimlrepo id=863)
    if "maternal" in wanted:
        try:
            p = get_maternal_health_ucimlrepo_raw(out_dir) if not args.skip_download else \
                out_dir / "maternal_health_risk" / "raw" / "maternal_health_risk_ucimlrepo_raw.csv"
            df, label_col, X_cols = clean_maternal_health_ucimlrepo(p, out_dir)
            reports["maternal_health_risk_863"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
        except Exception as e:
            print(f"[Maternal Health Risk] Skipped: {e}")
    if "ham10000" in wanted:
        try:
            p = (out_dir / "skin-cancer-mnist-ham10000" / "versions" / "2" / "hmnist_28_28_RGB.csv") if args.skip_download else \
                (out_dir / "skin-cancer-mnist-ham10000" / "versions" / "2" / "hmnist_28_28_RGB.csv")
            df, label_col, X_cols = clean_ham10000_hmnist(p, out_dir, pos_class=6)  # 6 = melanoma
            reports["ham10000_hmnist"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
        except Exception as e:
            print(f"[HAM10000 HMNIST] Skipped: {e}")

    if "mic" in wanted:
        try:
            target_col = "RAZRIV"
            p = get_mi_complications_ucimlrepo_raw(out_dir, target_col=target_col) \
                if not args.skip_download else \
                out_dir / "mi_complications" / "raw" / f"mi_complications_{target_col.lower()}_ucimlrepo_raw.csv"

            df, label_col, X_cols = clean_mi_complication_ucimlrepo(p, out_dir, target_col=target_col)
            reports["mi_complications"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
        except Exception as e:
            print(f"[MI complications] Skipped: {e}")

    if "mi_multilabel" in wanted:
            try:
                # 1. Fetch Full Raw Data using fetch_ucirepo
                p = get_mi_complications_full_raw(out_dir)
                # 2. Clean
                clean_mi_complications_multilabel(p, out_dir)
            except Exception as e:
                print(f"[MI Multi-label] Failed: {e}")
                import traceback
                traceback.print_exc()
    if "diabetes130" in wanted:
        try:
            p = get_diabetes130_ucimlrepo_raw(out_dir) if not args.skip_download else \
                out_dir / "diabetes130" / "raw" / "diabetes130_ucimlrepo_raw.csv"
            df, label_col, X_cols = clean_diabetes130_ucimlrepo(p, out_dir)
            reports["diabetes130_no_vs_lt30"] = {"label": label_col, "n_samples": len(df), "n_features": len(X_cols)}
        except Exception as e:
            print(f"[Diabetes130] Skipped: {e}")
            

    # Save a small report
    rep_path = out_dir / "download_report.json"
    rep_path.write_text(json.dumps(reports, indent=2))
    print("\nSummary:")
    print(json.dumps(reports, indent=2))
    print(f"\nWrote report → {rep_path}")

if __name__ == "__main__":
    main()