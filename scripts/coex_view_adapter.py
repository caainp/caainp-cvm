from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from scripts.localize_image import cosine_sim_matrix, encode_image_openclip, load_openclip  # type: ignore
except ImportError:
    from localize_image import cosine_sim_matrix, encode_image_openclip, load_openclip  # type: ignore


REQUIRED_INDEX_COLUMNS = [
    "row_index",
    "view_id",
    "node_id",
    "view_label",
    "image_path",
    "use_for_localization",
]


def _resolve_path(root_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _project_relative(root_dir: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), root_dir).replace(os.sep, "/")


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_view_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_INDEX_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"view_embedding_index.csv missing columns: {missing}")
    expected = list(range(len(df)))
    actual = [int(x) for x in df["row_index"].tolist()]
    if actual != expected:
        raise ValueError("row_index must be contiguous and aligned with embedding rows")
    if df["view_id"].duplicated().any():
        raise ValueError("duplicate view_id found in view_embedding_index.csv")
    return df


def _load_gallery(embeddings_path: Path, index_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(embeddings_path).astype(np.float32)
    index_df = _load_view_index(index_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")
    if embeddings.shape[0] != len(index_df):
        raise ValueError(
            f"Embedding/index row mismatch: embeddings={embeddings.shape[0]}, index={len(index_df)}"
        )
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embedding matrix contains NaN or inf")
    return embeddings, index_df


def _topk(scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0:
        return []
    k = min(max(int(k), 1), int(scores.size))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx.tolist()]


def _view_rows(
    *,
    query_image: str,
    scores: np.ndarray,
    selected_global_rows: np.ndarray,
    index_df: pd.DataFrame,
    topk: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, local_idx in enumerate(_topk(scores, topk), start=1):
        global_row = int(selected_global_rows[local_idx])
        meta = index_df.iloc[global_row]
        rows.append({
            "query_image": query_image,
            "rank": rank,
            "view_id": str(meta["view_id"]),
            "node_id": int(meta["node_id"]),
            "view_score": float(scores[local_idx]),
            "image_path": str(meta["image_path"]),
            "use_for_localization": str(meta["use_for_localization"]).strip().lower(),
        })
    return rows


def _node_rows(
    *,
    query_image: str,
    scores: np.ndarray,
    selected_global_rows: np.ndarray,
    index_df: pd.DataFrame,
    topk: int,
) -> List[Dict[str, Any]]:
    best_by_node: Dict[int, Dict[str, Any]] = {}
    for local_idx, score in enumerate(scores.tolist()):
        global_row = int(selected_global_rows[int(local_idx)])
        meta = index_df.iloc[global_row]
        node_id = int(meta["node_id"])
        score_f = float(score)
        current = best_by_node.get(node_id)
        if current is None or score_f > float(current["node_score"]):
            best_by_node[node_id] = {
                "node_id": node_id,
                "node_score": score_f,
                "best_view_id": str(meta["view_id"]),
                "best_view_score": score_f,
                "best_view_path": str(meta["image_path"]),
            }

    ordered = sorted(best_by_node.values(), key=lambda r: (-float(r["node_score"]), int(r["node_id"])))
    rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(ordered[: max(int(topk), 1)], start=1):
        rows.append({
            "query_image": query_image,
            "rank": rank,
            **row,
        })
    return rows


def localize_query(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    root_dir = Path(args.root_dir).resolve()
    embeddings_path = _resolve_path(root_dir, args.embeddings)
    index_path = _resolve_path(root_dir, args.index)
    query_path = _resolve_path(root_dir, args.query_image)

    embeddings, index_df = _load_gallery(embeddings_path, index_path)

    if args.include_all_views:
        mask = np.ones(len(index_df), dtype=bool)
    else:
        mask = index_df["use_for_localization"].map(_boolish).to_numpy(dtype=bool)
    if not mask.any():
        raise ValueError("No views selected for similarity search")

    selected_rows = np.where(mask)[0].astype(np.int64)
    gallery = embeddings[selected_rows]

    device = _resolve_device(str(args.device))
    model, preprocess = load_openclip(
        model_name=str(args.model),
        pretrained=str(args.pretrained),
        device=device,
    )
    query_embedding = encode_image_openclip(
        model,
        preprocess,
        str(query_path),
        device=device,
        aug_times=int(args.aug_times),
    )
    if query_embedding.shape[0] != gallery.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_embedding.shape[0]}, gallery={gallery.shape[1]}"
        )

    scores = cosine_sim_matrix(query_embedding, gallery).astype(np.float32)
    query_label = _project_relative(root_dir, query_path)
    view_rows = _view_rows(
        query_image=query_label,
        scores=scores,
        selected_global_rows=selected_rows,
        index_df=index_df,
        topk=int(args.view_topk),
    )
    node_rows = _node_rows(
        query_image=query_label,
        scores=scores,
        selected_global_rows=selected_rows,
        index_df=index_df,
        topk=int(args.node_topk),
    )
    return view_rows, node_rows
