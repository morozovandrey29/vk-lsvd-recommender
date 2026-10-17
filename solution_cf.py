"""VK-LSVD recommender comparison: Top Popular, ItemKNN, EASE, SLIM, ALS, TruncatedSVD.

Example (fit on train, compare on a labelled hold-out parquet and write recs):
    python solution_cf.py --input_path train.parquet --test_path test.parquet \
        --output_path submission.csv --model ease

The comparison table is written to <output_name>_metrics.csv by default.
EASE is an item-item linear model. Its dense inverse is expensive, so it is fitted
on the most popular --ease_max_items items (default: 1500). ItemKNN is likewise
pruned to a manageable top-item catalog.
"""
from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet
from scipy.sparse import hstack


def _bool_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[name].fillna(False).astype(bool).to_numpy()


def interaction_weights(df: pd.DataFrame) -> np.ndarray:
    """Heuristic confidence for implicit feedback; dislikes are not positive."""
    if "timespent" in df.columns:
        watch = pd.to_numeric(df["timespent"], errors="coerce").fillna(0)
        watch = watch.clip(lower=0, upper=255).to_numpy(dtype=np.float32)
    else:
        watch = np.zeros(len(df), dtype=np.float32)

    weights = 1.0 + 1.5 * (np.log1p(watch) / np.log(256.0))
    weights += 3.0 * _bool_column(df, "like")
    weights += 3.5 * _bool_column(df, "share")
    weights += 3.5 * _bool_column(df, "bookmark")
    weights += 0.5 * _bool_column(df, "click_on_author")
    weights += 0.5 * _bool_column(df, "open_comments")
    weights[_bool_column(df, "dislike")] = 0.0
    return weights.astype(np.float32)


def build_matrices(
    train: pd.DataFrame,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build weighted positives, all-seen mask, binary positives, IDs and popularity.

    `popularity_raw` mirrors the original baseline: count interaction rows per item.
    `popularity_pos` is the number of distinct users with positive interactions.
    """
    required = {"user_id", "item_id"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Required columns are missing from train: {sorted(missing)}")

    work = train.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("The training data is empty after rows without user_id/item_id were removed.")

    user_codes, user_ids = pd.factorize(work["user_id"], sort=False)
    item_codes, item_ids = pd.factorize(work["item_id"], sort=False)
    shape = (len(user_ids), len(item_ids))

    seen = csr_matrix(
        (np.ones(len(work), dtype=np.float32), (user_codes, item_codes)),
        shape=shape,
        dtype=np.float32,
    )
    seen.sum_duplicates()
    seen.data[:] = 1.0

    weights = interaction_weights(work)
    positive = weights > 0
    user_item = csr_matrix(
        (weights[positive], (user_codes[positive], item_codes[positive])),
        shape=shape,
        dtype=np.float32,
    )
    user_item.sum_duplicates()
    user_item.eliminate_zeros()

    binary = user_item.copy().tocsr()
    binary.data[:] = 1.0
    binary.eliminate_zeros()

    popularity_pos = binary.getnnz(axis=0).astype(np.float32)
    popularity_raw = np.bincount(item_codes, minlength=len(item_ids)).astype(np.float32)
    return user_item, seen, binary, user_ids.to_numpy(), item_ids.to_numpy(), popularity_pos, popularity_raw


def _popular_candidate_indices(popularity: np.ndarray, limit: int) -> np.ndarray:
    valid = np.flatnonzero(popularity > 0)
    if valid.size == 0:
        valid = np.arange(len(popularity), dtype=np.int64)
    order = valid[np.argsort(-popularity[valid], kind="stable")]
    return order[: max(1, min(int(limit), len(order)))].astype(np.int64, copy=False)


def fit_itemknn(
    binary: csr_matrix,
    popularity_pos: np.ndarray,
    max_items: int = 5000,
    neighbors: int = 100,
) -> Dict[str, Any]:
    """Fit a cosine item-item KNN model and retain only top neighbours per item."""
    candidate = _popular_candidate_indices(popularity_pos, max_items)
    x = binary[:, candidate].tocsr().astype(np.float32)
    print(f"  ItemKNN: co-occurrence matrix for {len(candidate):,} items...")
    cooc = (x.T @ x).tocsr()
    cooc.setdiag(0.0)
    cooc.eliminate_zeros()

    norms = np.sqrt(np.asarray(x.getnnz(axis=0), dtype=np.float32))
    coo = cooc.tocoo()
    denominator = norms[coo.row] * norms[coo.col]
    values = np.divide(coo.data, denominator, out=np.zeros_like(coo.data), where=denominator > 0)
    sim = csr_matrix((values.astype(np.float32), (coo.row, coo.col)), shape=cooc.shape)
    sim.eliminate_zeros()

    # Prune rows to the strongest neighbours, avoiding a dense similarity matrix.
    out_rows: List[np.ndarray] = []
    out_cols: List[np.ndarray] = []
    out_vals: List[np.ndarray] = []
    for row in range(sim.shape[0]):
        start, end = sim.indptr[row], sim.indptr[row + 1]
        cols = sim.indices[start:end]
        vals = sim.data[start:end]
        if len(vals) > neighbors:
            take = np.argpartition(vals, len(vals) - neighbors)[-neighbors:]
            cols, vals = cols[take], vals[take]
        if len(vals):
            out_rows.append(np.full(len(vals), row, dtype=np.int32))
            out_cols.append(cols.astype(np.int32, copy=False))
            out_vals.append(vals.astype(np.float32, copy=False))
    if out_vals:
        pruned = csr_matrix(
            (np.concatenate(out_vals), (np.concatenate(out_rows), np.concatenate(out_cols))),
            shape=sim.shape,
            dtype=np.float32,
        )
    else:
        pruned = csr_matrix(sim.shape, dtype=np.float32)

    return {"kind": "itemknn", "candidate_indices": candidate, "similarity": pruned}


def fit_ease(
    binary: csr_matrix,
    popularity_pos: np.ndarray,
    max_items: int = 1500,
    regularization: float = 500.0,
) -> Dict[str, Any]:
    """Fit EASE: B = -P / diag(P), P=(X.T X + lambda I)^-1; diagonal(B)=0."""
    candidate = _popular_candidate_indices(popularity_pos, max_items)
    n_items = len(candidate)
    if n_items < 2:
        raise ValueError("EASE requires at least two items with positive interactions.")

    print(f"  EASE: constructing dense {n_items:,} x {n_items:,} Gram matrix...")
    x = binary[:, candidate].tocsr().astype(np.float32)
    gram = (x.T @ x).toarray().astype(np.float64, copy=False)
    diagonal = np.diag_indices_from(gram)
    gram[diagonal] += float(regularization)
    precision = np.linalg.inv(gram)
    precision_diag = np.diag(precision).copy()
    weights = -precision / precision_diag[np.newaxis, :]
    weights[diagonal] = 0.0
    return {"kind": "ease", "candidate_indices": candidate, "weights": weights.astype(np.float32)}



def fit_slim(
    binary: csr_matrix,
    popularity_pos: np.ndarray,
    max_items: int = 500,
    neighbors: int = 100,
    alpha: float = 1e-3,
    l1_ratio: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-3,
    positive: bool = True,
) -> Dict[str, Any]:
    """Fit SLIM with one ElasticNet regression per target item.

    The learned matrix W is oriented so X @ W produces scores: W[i, j]
    describes how much source item i contributes to target item j. The diagonal
    is excluded by omitting target item j from its own regression predictors.
    """
    candidate = _popular_candidate_indices(popularity_pos, max_items)
    x = binary[:, candidate].tocsr().astype(np.float32)
    n_users, n_items = x.shape
    if n_items < 2:
        raise ValueError("SLIM requires at least two items.")
    if alpha <= 0:
        raise ValueError("slim_alpha must be positive.")
    if not 0.0 < l1_ratio <= 1.0:
        raise ValueError("slim_l1_ratio must be in the range (0, 1].")

    print(
        f"  SLIM: {n_items:,} items, {n_users:,} users, "
        f"ElasticNet alpha={alpha:g}, l1_ratio={l1_ratio:g}..."
    )
    x_csc = x.tocsc()
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    values: List[np.ndarray] = []

    # SLIM fits a separate sparse regression for each target item.
    for target in range(n_items):
        y = x_csc[:, target].toarray().ravel()
        if not np.any(y):
            continue
        predictors = hstack(
            (x[:, :target], x[:, target + 1 :]), format="csr", dtype=np.float32
        )
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            positive=positive,
            max_iter=max_iter,
            tol=tol,
            selection="cyclic",
            copy_X=False,
        )
        estimator.fit(predictors, y)
        coef = np.asarray(estimator.coef_, dtype=np.float32)
        nonzero = np.flatnonzero(np.abs(coef) > 1e-10)
        if nonzero.size > neighbors:
            strongest = np.argpartition(np.abs(coef[nonzero]), -neighbors)[-neighbors:]
            nonzero = nonzero[strongest]
        if nonzero.size:
            # Coefficients omit the target predictor, so shift indices after target.
            source = nonzero + (nonzero >= target)
            rows.append(source.astype(np.int32, copy=False))
            cols.append(np.full(len(source), target, dtype=np.int32))
            values.append(coef[nonzero].astype(np.float32, copy=False))
        if (target + 1) % 100 == 0 or target + 1 == n_items:
            print(f"    SLIM: processed {target + 1}/{n_items} target items")

    if values:
        weights = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
            dtype=np.float32,
        )
    else:
        weights = csr_matrix((n_items, n_items), dtype=np.float32)
    weights.setdiag(0.0)
    weights.eliminate_zeros()
    return {
        "kind": "slim",
        "candidate_indices": candidate,
        "weights": weights,
    }


def fit_als(
    user_item: csr_matrix,
    factors: int = 64,
    iterations: int = 20,
    regularization: float = 0.05,
    alpha: float = 20.0,
) -> Dict[str, Any]:
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError as exc:
        raise RuntimeError("The implicit library is not installed; run pip install implicit==0.7.2") from exc

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        random_state=42,
        num_threads=0,
    )
    model.fit(user_item.tocsr().astype(np.float32), show_progress=True)
    return {
        "kind": "factors",
        "name": "ALS",
        "user_factors": np.asarray(model.user_factors, dtype=np.float32),
        "item_factors": np.asarray(model.item_factors, dtype=np.float32),
        "candidate_indices": np.arange(user_item.shape[1], dtype=np.int64),
    }


def fit_svd(user_item: csr_matrix, factors: int = 64) -> Dict[str, Any]:
    n_users, n_items = user_item.shape
    n_components = min(factors, min(n_users, n_items) - 1)
    if n_components < 1:
        raise ValueError("TruncatedSVD requires more users and items.")
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd.fit(user_item)
    item_factors = svd.components_.T.astype(np.float32, copy=False)
    user_factors = np.asarray(user_item @ item_factors, dtype=np.float32)
    return {
        "kind": "factors",
        "name": "TruncatedSVD",
        "user_factors": user_factors,
        "item_factors": item_factors,
        "candidate_indices": np.arange(n_items, dtype=np.int64),
    }


def _seen_indices(seen: csr_matrix, user_idx: int) -> np.ndarray:
    return seen.indices[seen.indptr[user_idx] : seen.indptr[user_idx + 1]]


def _rank_scores(
    scores: np.ndarray,
    excluded: np.ndarray,
    candidate_indices: np.ndarray,
    popularity_tiebreak: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return catalog indices for top-k scores, filtering all train interactions."""
    if k <= 0 or len(scores) == 0:
        return np.empty(0, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float32).copy()
    score[~np.isfinite(score)] = -np.inf
    if excluded.size:
        score[np.isin(candidate_indices, excluded)] = -np.inf

    valid = np.flatnonzero(np.isfinite(score))
    take_n = min(k, len(valid))
    if take_n == 0:
        return np.empty(0, dtype=np.int64)

    # An extremely small popularity perturbation makes equal scores deterministic.
    tie = popularity_tiebreak[candidate_indices]
    max_tie = float(tie.max()) if tie.size else 0.0
    if max_tie > 0:
        score += (tie / max_tie) * np.float32(1e-7)
    if take_n == len(score):
        top_local = valid
    else:
        top_local = np.argpartition(score, len(score) - take_n)[-take_n:]
        top_local = top_local[np.isfinite(score[top_local])]
    top_local = top_local[np.argsort(score[top_local])[::-1]]
    return candidate_indices[top_local].astype(np.int64, copy=False)


def recommend_users(
    model: Dict[str, Any],
    requested_users: Sequence[Hashable],
    user_item: csr_matrix,
    seen: csr_matrix,
    binary: csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    popularity_pos: np.ndarray,
    popularity_raw: np.ndarray,
    k: int = 10,
    batch_size: int = 128,
) -> Dict[Hashable, List[Hashable]]:
    """Generate ranked recommendations; unknown users fall back to Top Popular."""
    user_to_idx = {u: i for i, u in enumerate(user_ids.tolist())}
    all_candidates = np.arange(len(item_ids), dtype=np.int64)
    candidate = model.get("candidate_indices", all_candidates)
    kind = model["kind"]
    result: Dict[Hashable, List[Hashable]] = {}
    max_batch = max(1, min(batch_size, 1_500_000 // max(1, len(candidate))))
    popular_order = np.argsort(-popularity_raw, kind="stable")

    for batch_start in range(0, len(requested_users), max_batch):
        users_batch = requested_users[batch_start : batch_start + max_batch]
        valid_positions: List[int] = []
        valid_user_indices: List[int] = []
        for pos, uid in enumerate(users_batch):
            idx = user_to_idx.get(uid)
            if idx is not None and user_item.indptr[idx + 1] > user_item.indptr[idx]:
                valid_positions.append(pos)
                valid_user_indices.append(idx)

        blocks: Dict[int, np.ndarray] = {}
        if valid_user_indices:
            if kind == "popularity":
                block = np.tile(popularity_raw[candidate], (len(valid_user_indices), 1))
            elif kind == "factors":
                block = model["user_factors"][valid_user_indices] @ model["item_factors"][candidate].T
            elif kind == "itemknn":
                seed = binary[valid_user_indices, :][:, candidate]
                block = (seed @ model["similarity"]).toarray()
            elif kind == "ease":
                seed = binary[valid_user_indices, :][:, candidate]
                block = np.asarray(seed @ model["weights"], dtype=np.float32)
            elif kind == "slim":
                seed = binary[valid_user_indices, :][:, candidate]
                block = (seed @ model["weights"]).toarray()
            else:
                raise ValueError(f"Unknown model type: {kind}")
            for row_no, pos in enumerate(valid_positions):
                blocks[pos] = np.asarray(block[row_no], dtype=np.float32)

        for pos, uid in enumerate(users_batch):
            idx = user_to_idx.get(uid)
            excluded = _seen_indices(seen, idx) if idx is not None else np.empty(0, dtype=np.int64)
            if pos in blocks:
                cand_for_row = candidate
                scores = blocks[pos]
                popularity_for_row = popularity_pos
            else:
                # New users and users with only dislikes have no positive profile.
                cand_for_row = all_candidates
                scores = popularity_raw
                popularity_for_row = popularity_raw

            ranked = _rank_scores(scores, excluded, cand_for_row, popularity_for_row, k)
            result[uid] = [item_ids[int(i)] for i in ranked]
    return result


def _relevant_by_user(test: pd.DataFrame) -> Dict[Hashable, set]:
    if not {"user_id", "item_id"}.issubset(test.columns):
        raise ValueError("To calculate metrics, test must contain user_id and item_id.")
    frame = test
    if "dislike" in frame.columns:
        frame = frame.loc[~frame["dislike"].fillna(False).astype(bool)]
    frame = frame[["user_id", "item_id"]].dropna().drop_duplicates()
    return frame.groupby("user_id")["item_id"].apply(set).to_dict()


def calculate_metrics(
    test: pd.DataFrame,
    recs: Dict[Hashable, List[Hashable]],
    item_ids: np.ndarray,
    k: int = 10,
) -> Dict[str, Any]:
    relevant = _relevant_by_user(test)
    users = list(relevant)
    if not users:
        return {
            f"MAP@{k}": 0.0,
            f"Precision@{k}": 0.0,
            f"Recall@{k}": 0.0,
            f"NDCG@{k}": 0.0,
            "CatalogCoverage": 0.0,
            "EvalUsers": 0,
        }

    ap_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    ndcg_values: List[float] = []
    all_recommended = set()
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))

    for uid in users:
        truth = relevant[uid]
        ranked = recs.get(uid, [])[:k]
        all_recommended.update(ranked)
        hit_count = 0
        precision_sum = 0.0
        dcg = 0.0
        for rank, item in enumerate(ranked, start=1):
            if item in truth:
                hit_count += 1
                precision_sum += hit_count / rank
                dcg += discounts[rank - 1]
        ap_values.append(precision_sum / min(len(truth), k))
        precision_values.append(hit_count / k)
        recall_values.append(hit_count / len(truth))
        ideal_len = min(len(truth), k)
        idcg = float(discounts[:ideal_len].sum())
        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    coverage = len(all_recommended.intersection(set(item_ids.tolist()))) / max(1, len(item_ids))
    return {
        f"MAP@{k}": float(np.mean(ap_values)),
        f"Precision@{k}": float(np.mean(precision_values)),
        f"Recall@{k}": float(np.mean(recall_values)),
        f"NDCG@{k}": float(np.mean(ndcg_values)),
        "CatalogCoverage": float(coverage),
        "EvalUsers": len(users),
    }


def _try_fit(name: str, fitter: Any, models: Dict[str, Dict[str, Any]]) -> None:
    print(f"\nFitting {name}...")
    try:
        models[name] = fitter()
        print(f"  {name}: done")
    except Exception as exc:
        warnings.warn(f"{name} was skipped: {type(exc).__name__}: {exc}")
        print(f"  {name}: FAILED TO FIT ({type(exc).__name__}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="VK-LSVD recommender: collaborative-filtering benchmark")
    parser.add_argument("--input_path", required=True, help="Path to the training Parquet file")
    parser.add_argument("--output_path", required=True, help="Path to the output recommendations CSV")
    parser.add_argument("--test_path", default=None, help="Optional hold-out Parquet file for MAP@K and other metrics")
    parser.add_argument(
        "--model", choices=["top_popular", "popularity", "itemknn", "ease", "slim", "als", "svd", "auto"],
        default="ease", help="Method for the output CSV; auto selects the highest MAP@K on the supplied test file (use a validation split)",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations per user")
    parser.add_argument("--factors", type=int, default=64, help="Latent factor dimension for ALS/SVD")
    parser.add_argument("--iterations", type=int, default=20, help="Number of ALS iterations")
    parser.add_argument("--regularization", type=float, default=0.05, help="ALS regularization")
    parser.add_argument("--alpha", type=float, default=20.0, help="Confidence scaling for implicit ALS")
    parser.add_argument("--ease_max_items", type=int, default=1500, help="Maximum number of popular items used by EASE (dense matrix)")
    parser.add_argument("--ease_reg", type=float, default=500.0, help="EASE regularization")
    parser.add_argument("--slim_max_items", type=int, default=500, help="Maximum number of popular items for SLIM; one regression is fitted per target item")
    parser.add_argument("--slim_neighbors", type=int, default=100, help="Maximum number of strongest coefficients retained per target item in SLIM")
    parser.add_argument("--slim_alpha", type=float, default=1e-3, help="ElasticNet regularization strength for SLIM; tune on validation data")
    parser.add_argument("--slim_l1_ratio", type=float, default=0.05, help="L1 ratio in SLIM ElasticNet: values closer to 1 generally produce sparser models")
    parser.add_argument("--slim_max_iter", type=int, default=1000, help="Maximum ElasticNet iterations per target item in SLIM")
    parser.add_argument("--slim_tol", type=float, default=1e-3, help="ElasticNet convergence tolerance for SLIM; smaller values can increase training time")
    parser.add_argument("--slim_positive", action=argparse.BooleanOptionalAction, default=True, help="Constrain SLIM coefficients to be non-negative; disable with --no-slim_positive")
    parser.add_argument("--knn_max_items", type=int, default=5000, help="Maximum number of items for ItemKNN")
    parser.add_argument("--knn_neighbors", type=int, default=100, help="Number of neighbors retained per item for ItemKNN")
    parser.add_argument("--metrics_path", default=None, help="Output path for the metrics table; defaults to a file next to output_path")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1")
    if args.ease_max_items < 2 or args.knn_max_items < 1 or args.knn_neighbors < 1:
        parser.error("ease_max_items must be >=2; knn_max_items and knn_neighbors must be >=1")
    if args.slim_max_items < 2 or args.slim_neighbors < 1 or args.slim_max_iter < 1:
        parser.error("slim_max_items must be >=2; slim_neighbors and slim_max_iter must be >=1")
    if args.ease_reg <= 0 or args.slim_alpha <= 0:
        parser.error("--ease_reg and --slim_alpha must be positive")
    if not 0.0 < args.slim_l1_ratio <= 1.0 or args.slim_tol <= 0:
        parser.error("--slim_l1_ratio must be in (0, 1]; --slim_tol must be positive")

    try:
        train = pd.read_parquet(args.input_path)
        test = pd.read_parquet(args.test_path) if args.test_path else None
    except ImportError as exc:
        raise SystemExit(
            "Could not read Parquet. Install a Parquet engine, for example: "
            "pip install pyarrow==21.0.0"
        ) from exc

    print(f"Train: {len(train):,} rows, {train['user_id'].nunique():,} users, "
          f"{train['item_id'].nunique():,} clips")
    user_item, seen, binary, user_ids, item_ids, popularity_pos, popularity_raw = build_matrices(train)
    print(f"Matrix: {user_item.shape[0]:,} users x {user_item.shape[1]:,} items; "
          f"positive user-item pairs: {user_item.nnz:,}")

    if test is not None and "user_id" in test.columns:
        request_users = test["user_id"].dropna().drop_duplicates().tolist()
    else:
        request_users = user_ids.tolist()

    # Always include the original-style top-popularity baseline.
    models: Dict[str, Dict[str, Any]] = {
        "top_popular": {
            "kind": "popularity",
            "candidate_indices": np.arange(len(item_ids), dtype=np.int64),
        }
    }
    can_compare = test is not None and {"user_id", "item_id"}.issubset(test.columns)
    names_to_fit = ["itemknn", "ease", "slim", "als", "svd"] if can_compare else []
    if not can_compare:
        if args.model not in ("top_popular", "popularity", "auto"):
            names_to_fit = [args.model]
        elif args.model == "auto":
            names_to_fit = ["ease"]

    fitters = {
        "itemknn": lambda: fit_itemknn(binary, popularity_pos, args.knn_max_items, args.knn_neighbors),
        "ease": lambda: fit_ease(binary, popularity_pos, args.ease_max_items, args.ease_reg),
        "slim": lambda: fit_slim(
            binary, popularity_pos, args.slim_max_items, args.slim_neighbors,
            args.slim_alpha, args.slim_l1_ratio, args.slim_max_iter,
            args.slim_tol, args.slim_positive,
        ),
        "als": lambda: fit_als(user_item, args.factors, args.iterations, args.regularization, args.alpha),
        "svd": lambda: fit_svd(user_item, args.factors),
    }
    for name in names_to_fit:
        if name not in models:
            _try_fit(name, fitters[name], models)

    if can_compare:
        print("\nEvaluating on the hold-out file (all non-dislike interactions are treated as relevant)...")
        metric_rows: List[Dict[str, Any]] = []
        chosen_name: Optional[str] = None
        chosen_recs: Optional[Dict[Hashable, List[Hashable]]] = None
        chosen_map = -1.0
        for name, model in models.items():
            recs = recommend_users(
                model, request_users, user_item, seen, binary, user_ids, item_ids,
                popularity_pos, popularity_raw, k=args.top_k,
            )
            metrics = calculate_metrics(test, recs, item_ids, k=args.top_k)
            row = {"Model": name, **metrics}
            metric_rows.append(row)
            print(
                f"{name:12s} | MAP@{args.top_k}={metrics[f'MAP@{args.top_k}']:.6f} | "
                f"P@{args.top_k}={metrics[f'Precision@{args.top_k}']:.6f} | "
                f"R@{args.top_k}={metrics[f'Recall@{args.top_k}']:.6f} | "
                f"NDCG@{args.top_k}={metrics[f'NDCG@{args.top_k}']:.6f} | "
                f"coverage={metrics['CatalogCoverage']:.3%}"
            )

            # Retain only the requested output's recommendations (or current best for auto).
            if args.model == "auto":
                current_map = float(metrics[f"MAP@{args.top_k}"])
                if current_map > chosen_map:
                    chosen_map, chosen_name, chosen_recs = current_map, name, recs
            elif args.model in (name, "popularity" if name == "top_popular" else name):
                chosen_name, chosen_recs = name, recs

            del recs

        metrics_df = pd.DataFrame(metric_rows).sort_values(f"MAP@{args.top_k}", ascending=False)
        metrics_path = args.metrics_path or os.path.splitext(args.output_path)[0] + "_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to: {metrics_path}")

        positive_test = test
        if "dislike" in positive_test.columns:
            positive_test = positive_test.loc[~positive_test["dislike"].fillna(False).astype(bool)]
        test_pairs = positive_test[["user_id", "item_id"]].dropna().drop_duplicates()
        in_catalog = test_pairs["item_id"].isin(set(item_ids.tolist())).mean() if len(test_pairs) else 0.0
        print(f"Coverage of hold-out items in the training catalog: {in_catalog:.2%}")
        if args.model == "auto" and chosen_name is not None:
            print(f"auto selected {chosen_name} by MAP@{args.top_k} on the supplied hold-out file. "
                  "To avoid selecting a model on the final test set, use a separate validation split.")
    else:
        if args.model in ("top_popular", "popularity"):
            chosen_name = "top_popular"
        elif args.model == "auto":
            chosen_name = "ease"
        else:
            chosen_name = args.model
        chosen_recs = None
        if test is not None:
            print("The hold-out file does not contain both user_id and item_id; metrics cannot be calculated.")

    # Without labelled test, use the explicitly requested method.
    if chosen_name is None:
        chosen_name = "top_popular" if args.model in ("top_popular", "popularity", "auto") else args.model
    if chosen_name not in models:
        raise SystemExit(
            f"Model '{chosen_name}' was not fitted. Check the messages above; "
            "install implicit==0.7.2 for ALS, or choose --model ease/itemknn/svd/top_popular."
        )
    if chosen_recs is None:
        chosen_recs = recommend_users(
            models[chosen_name], request_users, user_item, seen, binary, user_ids, item_ids,
            popularity_pos, popularity_raw, k=args.top_k,
        )

    output_rows = [
        (uid, item_id)
        for uid in request_users
        for item_id in chosen_recs.get(uid, [])
    ]
    output = pd.DataFrame(output_rows, columns=["user_id", "recs"])
    output.to_csv(args.output_path, index=False)
    print(f"\nSelected method for the submission: {chosen_name}")
    print(f"Saved: {args.output_path} ({len(output):,} rows, {len(request_users):,} users)")


if __name__ == "__main__":
    main()
