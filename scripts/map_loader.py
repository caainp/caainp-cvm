import ast
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class NodeRecord:
    node_id: int
    floor: Optional[str]
    description: Optional[str]
    neighbors: List[int]
    node_type: Optional[str]
    clip_embedding: Optional[np.ndarray]
    extra: Dict[str, Any]


def _parse_neighbors(value: Any) -> List[int]:
    """
    Accepts formats:
    - JSON list string: "[401, 402]"
    - Python list/tuple string: "[401, 402]" or "(401, 402)"
    - Delimited string: "401,402" or "401; 402"
    - Iterable of ints
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    s = str(value).strip()
    if not s:
        return []
    # Try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]
    except Exception:
        pass
    # Try Python literal
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]
    except Exception:
        pass
    # Fallback: split by non-digits
    parts = [p for p in re_split_non_digit(s) if p]
    try:
        return [int(p) for p in parts]
    except Exception:
        logger.warning(f"Failed to parse neighbors from: {value!r}")
        return []


def re_split_non_digit(s: str) -> List[str]:
    import re
    return re.split(r"[^0-9]+", s)


def _parse_embedding(value: Any) -> Optional[np.ndarray]:
    """
    Accepts formats:
    - JSON list string: "[0.1, -0.2, ...]"
    - Python list/tuple string
    - Comma/space-separated floats
    - Already a list/ndarray
    Returns L2-normalized float32 vector when possible.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        vec = value.astype(np.float32)
        return _safe_l2_normalize(vec)
    if isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=np.float32)
        return _safe_l2_normalize(vec)
    s = str(value).strip()
    if not s:
        return None
    # Try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)):
            vec = np.asarray(parsed, dtype=np.float32)
            return _safe_l2_normalize(vec)
    except Exception:
        pass
    # Try Python literal
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            vec = np.asarray(parsed, dtype=np.float32)
            return _safe_l2_normalize(vec)
    except Exception:
        pass
    # Fallback: split by commas/spaces
    try:
        parts = [p for p in s.replace(",", " ").split() if p]
        vec = np.asarray([float(p) for p in parts], dtype=np.float32)
        return _safe_l2_normalize(vec)
    except Exception:
        logger.warning("Failed to parse clip_embedding; returning None")
        return None


def _safe_l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec) + 1e-12)
    return (vec / norm).astype(np.float32)


def load_map_csv(
    csv_path: str,
    id_column: str = "node_id",
    floor_column: str = "floor",
    description_column_candidates: Tuple[str, ...] = ("description", "descriptor"),
    neighbors_column: str = "neighbors",
    type_column: str = "type",
    embedding_column: str = "clip_embedding",
) -> Tuple[nx.Graph, Dict[int, NodeRecord], np.ndarray, List[int]]:
    """
    Load CSV and build:
      - networkx Graph
      - node_id -> NodeRecord
      - embedding matrix (N x D) aligned with node_ids_for_embeddings
      - node_ids_for_embeddings (list)
    Extra columns are kept in NodeRecord.extra.
    """
    df = pd.read_csv(csv_path)
    cols = set(df.columns.astype(str))

    if id_column not in cols:
        raise KeyError(f"Missing required column: {id_column}")
    if neighbors_column not in cols:
        raise KeyError(f"Missing required column: {neighbors_column}")

    # Resolve description/descriptor
    description_col = None
    for cand in description_column_candidates:
        if cand in cols:
            description_col = cand
            break

    graph = nx.Graph()
    node_records: Dict[int, NodeRecord] = {}
    embeddings: List[np.ndarray] = []
    node_ids_for_embeddings: List[int] = []

    for _, row in df.iterrows():
        node_id = int(row[id_column])
        floor_val = row.get(floor_column, None)
        desc_val = row.get(description_col, None) if description_col else None
        type_val = row.get(type_column, None)

        neighbors_val = _parse_neighbors(row.get(neighbors_column, None))
        emb = _parse_embedding(row.get(embedding_column, None)) if embedding_column in cols else None

        # capture extras
        extra: Dict[str, Any] = {}
        for c in cols:
            if c in {id_column, floor_column, neighbors_column, type_column, embedding_column}:
                continue
            if description_col and c == description_col:
                continue
            extra[c] = row.get(c, None)

        rec = NodeRecord(
            node_id=node_id,
            floor=str(floor_val) if pd.notna(floor_val) else None,
            description=str(desc_val) if (desc_val is not None and pd.notna(desc_val)) else None,
            neighbors=neighbors_val,
            node_type=str(type_val) if (type_val is not None and pd.notna(type_val)) else None,
            clip_embedding=emb,
            extra=extra,
        )
        node_records[node_id] = rec
        graph.add_node(node_id, floor=rec.floor, type=rec.node_type, description=rec.description, **rec.extra)

    # add edges (undirected)
    for node_id, rec in node_records.items():
        for nb in rec.neighbors:
            if nb in node_records:
                graph.add_edge(node_id, nb)
            else:
                logger.warning(f"Neighbor {nb} of node {node_id} not found in CSV; skipping edge")

    # build embedding matrix
    for node_id, rec in node_records.items():
        if rec.clip_embedding is not None:
            embeddings.append(rec.clip_embedding)
            node_ids_for_embeddings.append(node_id)

    if embeddings:
        emb_matrix = np.vstack(embeddings).astype(np.float32)
    else:
        emb_matrix = np.zeros((0, 0), dtype=np.float32)

    logger.info(
        f"Loaded {len(node_records)} nodes; {emb_matrix.shape[0]} with embeddings; "
        f"graph edges={graph.number_of_edges()}"
    )
    return graph, node_records, emb_matrix, node_ids_for_embeddings


__all__ = [
    "NodeRecord",
    "load_map_csv",
]

