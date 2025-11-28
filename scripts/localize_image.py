import argparse
import glob
import os
import re
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from loguru import logger

import torch
import open_clip
import cv2
import networkx as nx
try:
    import easyocr  # optional
except Exception:
    easyocr = None

# Support running as a module (-m scripts.localize_image) and as a file (python scripts/localize_image.py)
try:
    from scripts.map_loader import load_map_csv  # type: ignore
except Exception:
    from map_loader import load_map_csv  # type: ignore

# Reduce noisy warnings and avoid symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Optional: disable tokenizers parallelism warning if it appears in some envs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Targeted warning filters (console cleanliness; does not change behavior)
warnings.filterwarnings("ignore", message=".*pinned memory.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*symlink.*", category=UserWarning)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)


def cosine_sim_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """
    query: (D,) or (B,D)
    gallery: (N,D)
    returns: (N,) or (B,N)
    """
    if query.ndim == 1:
        return (query[None, :] @ gallery.T).squeeze(0)
    return query @ gallery.T


def load_openclip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess


def guess_model_for_embed_dim(embed_dim: int) -> Optional[Tuple[str, str]]:
    """
    Heuristic mapping from embedding dimension to a common OpenCLIP model.
    Note: This is a best-effort guess; prefer explicitly aligning train/infer configs.
    """
    mapping: Dict[int, Tuple[str, str]] = {
        512: ("ViT-B-32", "laion2b_s34b_b79k"),
        768: ("ViT-L-14", "laion2b_s32b_b82k"),
        1024: ("ViT-H-14", "laion2b_s32b_b79k"),
    }
    return mapping.get(int(embed_dim))


@torch.inference_mode()
def encode_image_openclip(model, preprocess, image_path: str, device: str = "cpu", aug_times: int = 1) -> np.ndarray:
    """
    Light-weight robustness via test-time augmentation (center crop composed by preprocess).
    If aug_times > 1, we do simple horizontal flip ensemble for robustness.
    """
    img = Image.open(image_path).convert("RGB")
    imgs: List[Image.Image] = [img]
    if aug_times > 1:
        imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    embs: List[np.ndarray] = []
    for im in imgs:
        tensor = preprocess(im).unsqueeze(0).to(device)
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats[0].detach().cpu().numpy().astype(np.float32))
    emb = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(emb)


def topk_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[-1])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def extract_room_numbers_from_text(texts: List[str]) -> List[str]:
    numbers: List[str] = []
    for t in texts:
        for match in re.findall(r"\b\d{2,5}\b", t):
            numbers.append(match)
    # unique preserve order
    seen = set()
    out = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def ocr_hints(image_path: str) -> List[str]:
    if easyocr is None:
        logger.warning("easyocr not available; skipping OCR hints")
        return []
    # Use GPU if available to avoid CPU warning and speed up OCR
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    result = reader.readtext(image_path, detail=0)
    return extract_room_numbers_from_text(result)


def parse_room_range(value: Optional[str]) -> List[Tuple[int, int]]:
    if value is None or not isinstance(value, str) or not value.strip():
        return []
    ranges: List[Tuple[int, int]] = []
    parts = re.split(r"[;,]", value)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # "4101-4110" or single "4101"
        m = re.match(r"^\s*(\d{2,5})(?:\s*-\s*(\d{2,5}))?\s*$", p)
        if m:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            if a <= b:
                ranges.append((a, b))
            else:
                ranges.append((b, a))
    return ranges


def room_number_match_score(
    node_meta: Dict[str, any],
    ocr_nums: List[str],
    description: Optional[str]
) -> float:
    if not ocr_nums:
        return 0.0
    desc = description or ""
    anchor_room = node_meta.get("anchor_room")
    room_range = node_meta.get("room_range")

    ranges = parse_room_range(room_range if isinstance(room_range, str) else str(room_range) if room_range is not None else None)

    score = 0.0
    for n in ocr_nums:
        try:
            vn = int(n)
        except Exception:
            continue
        # exact anchor match
        if anchor_room is not None:
            try:
                if int(str(anchor_room)) == vn:
                    score = max(score, 1.0)
            except Exception:
                pass
        # within range
        for a, b in ranges:
            if a <= vn <= b:
                score = max(score, 0.8)
                break
        # description contains number
        if n in desc:
            score = max(score, 0.6)
    return score


def geometric_verification_score(
    query_image_path: str,
    reference_image_paths: List[str],
) -> float:
    if not reference_image_paths:
        return 0.0
    try:
        sift = cv2.SIFT_create()
    except Exception:
        logger.warning("SIFT unavailable; skipping geometric verification")
        return 0.0

    try:
        query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            return 0.0
        kp_q, des_q = sift.detectAndCompute(query_img, None)
        if des_q is None or len(kp_q) < 8:
            return 0.0
    except Exception:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    best_norm_inliers = 0.0
    tested = 0

    for ref_path in reference_image_paths:
        try:
            ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref is None:
                continue
            kp_r, des_r = sift.detectAndCompute(ref, None)
            if des_r is None or len(kp_r) < 8:
                continue
            matches = bf.knnMatch(des_q, des_r, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) < 8:
                continue
            src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is None:
                continue
            inliers = int(mask.sum())
            # normalize by number of keypoints to keep 0..1-ish
            norm = inliers / (len(good) + 1e-6)
            best_norm_inliers = max(best_norm_inliers, float(norm))
            tested += 1
        except Exception:
            continue
    return best_norm_inliers if tested > 0 else 0.0


def collect_reference_images(node_images_dir: str, node_id: int, limit: int = 4) -> List[str]:
    patterns = [
        os.path.join(node_images_dir, f"{node_id}*.jpg"),
        os.path.join(node_images_dir, f"{node_id}*.png"),
        os.path.join(node_images_dir, f"{node_id}*.jpeg"),
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    # keep deterministic subset
    paths = sorted(paths)[:limit]
    return paths


def graph_prior_score(graph: nx.Graph, prev_node: Optional[int], candidate_node: int, alpha: float = 1.0) -> float:
    if prev_node is None or prev_node not in graph or candidate_node not in graph:
        return 0.0
    try:
        d = nx.shortest_path_length(graph, prev_node, candidate_node)
    except Exception:
        return 0.0
    # convert distance to [0,1]: closer → higher prior
    return float(np.exp(-alpha * float(d)))


def localize_image(
    image_path: str,
    csv_path: str,
    device: str = "cpu",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    topk: int = 5,
    use_ocr: bool = False,
    node_images_dir: Optional[str] = None,
    use_geo: bool = False,
    prev_node: Optional[int] = None,
    w_clip: float = 1.0,
    w_ocr: float = 0.3,
    w_geo: float = 0.4,
    w_prior: float = 0.2,
    auto_match_model: bool = True,
) -> dict:
    # load map
    graph, node_records, emb_matrix, node_ids = load_map_csv(csv_path)
    if emb_matrix.size == 0:
        raise RuntimeError("No clip_embedding found in CSV; cannot localize. Please populate embeddings first.")

    # Ensure model matches CSV embedding dimension if requested
    embed_dim = int(emb_matrix.shape[1]) if emb_matrix.ndim == 2 else 0
    if auto_match_model and embed_dim > 0:
        guessed = guess_model_for_embed_dim(embed_dim)
        if guessed is not None:
            g_model, g_pretrained = guessed
            if (g_model != model_name) or (g_pretrained != pretrained):
                logger.info(f"Embedding dim={embed_dim}; overriding model to {g_model} ({g_pretrained}) for compatibility")
                model_name, pretrained = g_model, g_pretrained
        else:
            logger.warning(f"Unknown embedding dim={embed_dim}; cannot auto-match model. "
                           f"Make sure CSV embeddings and inference model are aligned.")

    # load model + encode query with the (possibly updated) config
    model, preprocess = load_openclip(model_name=model_name, pretrained=pretrained, device=device)
    q = encode_image_openclip(model, preprocess, image_path, device=device, aug_times=2)  # small TTA

    # Final sanity: dimension check
    if q.shape[0] != emb_matrix.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={q.shape[0]} vs CSV={emb_matrix.shape[1]}. "
            f"CSV likely built with a different CLIP model.\n"
            f"Fix options:\n"
            f"  - Re-run with a matching model/pretrained (e.g., {guess_model_for_embed_dim(emb_matrix.shape[1])}), or\n"
            f"  - Rebuild CSV clip_embedding with the current model ({model_name}, {pretrained})."
        )

    # cosine sims
    sims = cosine_sim_matrix(q, emb_matrix)  # (N,)

    # initial top-k by CLIP
    idx0, vals0 = topk_indices(sims, max(topk, 10))  # take a slightly larger pool for re-ranking

    # optional OCR hints
    ocr_adj = np.zeros_like(vals0)
    ocr_nums: List[str] = []
    if use_ocr:
        ocr_nums = ocr_hints(image_path)
        if ocr_nums:
            for j, ni in enumerate(idx0):
                nid = int(node_ids[ni])
                rec = node_records[nid]
                score = room_number_match_score(rec.extra, ocr_nums, rec.description)
                ocr_adj[j] = score

    # optional geometric verification per candidate using node_images_dir
    geo_adj = np.zeros_like(vals0)
    if use_geo and node_images_dir:
        for j, ni in enumerate(idx0):
            nid = int(node_ids[ni])
            ref_paths = collect_reference_images(node_images_dir, nid)
            if ref_paths:
                geo_adj[j] = geometric_verification_score(image_path, ref_paths)

    # optional graph prior from prev_node
    prior_adj = np.zeros_like(vals0)
    if prev_node is not None:
        for j, ni in enumerate(idx0):
            nid = int(node_ids[ni])
            prior_adj[j] = graph_prior_score(graph, prev_node, nid, alpha=0.7)

    # combine scores
    combined = w_clip * vals0 + w_ocr * ocr_adj + w_geo * geo_adj + w_prior * prior_adj

    # final topk
    idx, vals = topk_indices(combined, topk)
    top_nodes = [int(node_ids[idx0[i]]) for i in idx]
    top_scores = [float(vals[i]) for i in range(len(idx))]

    # confidence with temperature scaling on combined top-k
    tau = 0.05
    expv = np.exp((np.array(top_scores) - np.max(top_scores)) / max(tau, 1e-6))
    probs = expv / np.sum(expv)
    confidence = float(probs[0])

    result = {
        "current_node": top_nodes[0],
        "confidence": confidence,
        "candidates": [{"node_id": n, "score": s} for n, s in zip(top_nodes, top_scores)],
        "debug": {
            "ocr_numbers": ocr_nums,
            "weights": {"w_clip": w_clip, "w_ocr": w_ocr, "w_geo": w_geo, "w_prior": w_prior},
        },
    }
    return result
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Query image path")
    ap.add_argument("--csv", required=True, help="CSV path with clip_embedding column")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--use_ocr", action="store_true", help="Use OCR re-ranking")
    ap.add_argument("--node_images_dir", default="node_images", help="Directory of reference node images")
    ap.add_argument("--use_geo", action="store_true", help="Use geometric verification re-ranking")
    ap.add_argument("--prev_node", type=int, default=None, help="Previous node id for graph prior")
    ap.add_argument("--w_clip", type=float, default=1.0)
    ap.add_argument("--w_ocr", type=float, default=0.3)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--w_prior", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device={device}")
    out = localize_image(
        image_path=args.image,
        csv_path=args.csv,
        device=device,
        model_name=args.model,
        pretrained=args.pretrained,
        topk=args.topk,
        use_ocr=args.use_ocr,
        node_images_dir=args.node_images_dir,
        use_geo=args.use_geo,
        prev_node=args.prev_node,
        w_clip=args.w_clip,
        w_ocr=args.w_ocr,
        w_geo=args.w_geo,
        w_prior=args.w_prior,
    )
    print(out)


if __name__ == "__main__":
    main()

