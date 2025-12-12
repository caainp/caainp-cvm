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


def normalize01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def extract_room_numbers_from_text(texts: List[str]) -> List[str]:
    numbers: List[str] = []
    for t in texts:
        # Normalize common OCR confusions (e.g., 'O'/'o'→'0', 'I'/'l'/'|'/'!'→'1', 'S'/'s'→'5', 'B'/'b'→'8')
        norm = (
            t.replace("O", "0").replace("o", "0")
             .replace("I", "1").replace("l", "1").replace("|", "1").replace("!", "1")
             .replace("S", "5").replace("s", "5")
             .replace("B", "8").replace("b", "8")
        )
        # Capture 2~5 digit sequences even when adjacent to non-digit chars (e.g., "4101호")
        for match in re.findall(r"(?<!\d)(\d{2,5})(?!\d)", norm):
            numbers.append(match)
    # unique preserve order
    seen = set()
    out: List[str] = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            out.append(n)
    # Debug logging for OCR parsing
    logger.debug(f"OCR raw texts={texts}, parsed room numbers={out}")
    return out


def ocr_hints(
    image_path: str,
    languages: Optional[List[str]] = None,
    min_confidence: float = 0.4,
    readtext_kwargs: Optional[Dict[str, any]] = None,
) -> List[str]:
    if easyocr is None:
        logger.warning("easyocr not available; skipping OCR hints")
        return []
    # Default to Korean+English when not specified
    langs = languages if (languages and len(languages) > 0) else ["ko", "en"]
    # Use GPU if available to avoid CPU warning and speed up OCR
    reader = easyocr.Reader(langs, gpu=torch.cuda.is_available())
    texts: List[str] = []
    try:
        result = reader.readtext(image_path, detail=1, **(readtext_kwargs or {}))
        for item in result or []:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = str(item[1])
                conf = float(item[2])
                if conf >= float(min_confidence):
                    texts.append(text)
    except Exception:
        try:
            texts = reader.readtext(image_path, detail=0, **(readtext_kwargs or {})) or []
        except Exception:
            texts = []
    return extract_room_numbers_from_text(texts)


def preprocess_image_for_ocr(
    image_bgr: np.ndarray,
    use_contrast: bool = False,
    use_sharpen: bool = False,
    use_adaptive: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
    sharpen_amount: float = 0.7,
    adaptive_block_size: int = 31,
    adaptive_C: int = 5,
) -> np.ndarray:
    """
    Optional contrast (CLAHE), unsharp mask sharpening, adaptive thresholding for OCR.
    Returns processed image (single channel if adaptive thresholding applied).
    """
    img = image_bgr.copy()
    if use_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)))
        L = clahe.apply(L)
        lab = cv2.merge([L, A, B])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if use_sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
        img = cv2.addWeighted(img, 1.0 + float(sharpen_amount), blur, -float(sharpen_amount), 0)
    if use_adaptive:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block = int(adaptive_block_size)
        if block % 2 == 0:
            block += 1
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(adaptive_C)
        )
        return th
    return img


def detect_text_rois(
    image_path: str,
    max_rois: int = 8,
    min_area: int = 500,   # pixels
    max_area: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Heuristic text ROI detector using morphological gradients.
    Returns list of (x, y, w, h) sorted by area desc, truncated to max_rois.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Enhance text-like structures
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    grad = cv2.GaussianBlur(grad, (3, 3), 0)
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close gaps horizontally to merge characters into words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    if max_area is None:
        max_area = int(0.8 * H * W)
    rois: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area:
            continue
        ar = w / float(h + 1e-6)
        # Text lines often have 2 <= aspect ratio <= 20; relax to include plates
        if ar < 1.2 or ar > 25.0:
            continue
        # Keep within image bounds
        x0 = max(0, x - 4); y0 = max(0, y - 4)
        x1 = min(W, x + w + 4); y1 = min(H, y + h + 4)
        rois.append((x0, y0, x1 - x0, y1 - y0))
    # Sort by area desc and deduplicate overlapping boxes
    rois = sorted(rois, key=lambda r: r[2] * r[3], reverse=True)
    filtered: List[Tuple[int, int, int, int]] = []
    def iou(a, b) -> float:
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah; bx2, by2 = bx + bw, by + bh
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter + 1e-6
        return inter / union
    for r in rois:
        if all(iou(r, f) < 0.3 for f in filtered):
            filtered.append(r)
        if len(filtered) >= max_rois:
            break
    return filtered


def ocr_hints_with_roi(
    image_path: str,
    languages: Optional[List[str]] = None,
    max_rois: int = 8,
    readtext_kwargs: Optional[Dict[str, any]] = None,
    preproc_opts: Optional[Dict[str, any]] = None,
    min_confidence: float = 0.4,
    tta_rotate_deg: float = 0.0,
    tta_scales: Optional[List[float]] = None,
    tta_max_aug: int = 6,
) -> List[str]:
    if easyocr is None:
        logger.warning("easyocr not available; skipping OCR hints")
        return []
    # Default to Korean+English when not specified
    langs = languages if (languages and len(languages) > 0) else ["ko", "en"]
    reader = easyocr.Reader(langs, gpu=torch.cuda.is_available())
    # Run once on full image
    texts: List[str] = []
    try:
        img_full = cv2.imread(image_path)
        if img_full is not None and preproc_opts is not None:
            img_full = preprocess_image_for_ocr(img_full, **preproc_opts)
            res_full = reader.readtext(img_full, detail=1, **(readtext_kwargs or {}))
        else:
            res_full = reader.readtext(image_path, detail=1, **(readtext_kwargs or {}))
        for item in res_full or []:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = str(item[1])
                conf = float(item[2])
                if conf >= float(min_confidence):
                    texts.append(text)
    except Exception:
        pass
    # Then on candidate ROIs
    rois = detect_text_rois(image_path, max_rois=max_rois)
    img = cv2.imread(image_path)
    if img is not None:
        for (x, y, w, h) in rois:
            crop = img[y:y+h, x:x+w]
            try:
                if preproc_opts is not None:
                    crop = preprocess_image_for_ocr(crop, **preproc_opts)
                # Base pass
                res = reader.readtext(crop, detail=1, **(readtext_kwargs or {}))
                for item in res or []:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        text = str(item[1]); conf = float(item[2])
                        if conf >= float(min_confidence):
                            texts.append(text)
                # Optional TTA: small rotations and scales
                if tta_rotate_deg and tta_rotate_deg > 0:
                    angles = [-float(tta_rotate_deg), 0.0, float(tta_rotate_deg)]
                else:
                    angles = [0.0]
                scales = tta_scales if (tta_scales and len(tta_scales) > 0) else [1.0]
                aug_count = 0
                Hc, Wc = crop.shape[:2]
                for a in angles:
                    if aug_count >= int(tta_max_aug):
                        break
                    if abs(a) > 1e-6:
                        M = cv2.getRotationMatrix2D((Wc / 2.0, Hc / 2.0), a, 1.0)
                        aug = cv2.warpAffine(crop, M, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    else:
                        aug = crop
                    for s in scales:
                        if aug_count >= int(tta_max_aug):
                            break
                        if abs(s - 1.0) > 1e-6:
                            aug2 = cv2.resize(aug, (int(Wc * float(s)), int(Hc * float(s))), interpolation=cv2.INTER_LINEAR)
                        else:
                            aug2 = aug
                        try:
                            res_tta = reader.readtext(aug2, detail=1, **(readtext_kwargs or {}))
                            for item in res_tta or []:
                                if isinstance(item, (list, tuple)) and len(item) >= 3:
                                    text = str(item[1]); conf = float(item[2])
                                    if conf >= float(min_confidence):
                                        texts.append(text)
                        except Exception:
                            pass
                        aug_count += 1
            except Exception:
                continue
    return extract_room_numbers_from_text(texts)


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
    # Enhanced reranking weights
    w_id: float = 0.6,                 # OCR 숫자와 node_id 직접 일치/접두 일치 보너스
    w_consistency: float = 0.4,        # 그래프 이웃과의 후보 일관성 보너스
    ocr_langs: Optional[List[str]] = None,
    ocr_use_roi: bool = False,
    ocr_max_rois: int = 8,
    # OCR confidence and filtering
    ocr_min_conf: float = 0.4,
    ocr_filter_to_nodes: bool = False,
    ocr_digits_min: int = 3,
    ocr_digits_max: int = 4,
    # OCR TTA
    ocr_augment: bool = False,
    ocr_tta_deg: float = 6.0,
    ocr_tta_scales: Optional[List[float]] = None,
    ocr_tta_max_aug: int = 6,
    # OCR preprocessing flags
    ocr_contrast: bool = False,
    ocr_sharpen: bool = False,
    ocr_adaptive: bool = False,
    ocr_clahe_clip: float = 2.0,
    ocr_clahe_grid: int = 8,
    ocr_sharpen_amount: float = 0.7,
    ocr_adaptive_block: int = 31,
    ocr_adaptive_C: int = 5,
    # EasyOCR readtext tuning
    ocr_text_threshold: Optional[float] = None,
    ocr_low_text: Optional[float] = None,
    ocr_link_threshold: Optional[float] = None,
    ocr_decoder: Optional[str] = None,
    ocr_beam_width: Optional[int] = None,
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

    # initial candidate pool by CLIP (larger pool for robust re-ranking)
    pool_size = max(int(topk) * 10, 50)
    idx0, vals0 = topk_indices(sims, pool_size)
    # Force-inject OCR-matched node_ids so re-ranking can correct CLIP misses
    if use_ocr:
        # quick OCR pass (no ROI, minimal kwargs) to get obvious numbers
        try:
            quick_nums = ocr_hints(
                image_path=image_path,
                languages=ocr_langs,
                min_confidence=float(ocr_min_conf),
                readtext_kwargs=None,
            )
        except Exception:
            quick_nums = []
        if quick_nums:
            quick_nums = [n for n in quick_nums if len(n) >= int(ocr_digits_min) and len(n) <= int(ocr_digits_max)]
            if ocr_filter_to_nodes:
                whitelist = set(str(int(k)) for k in node_ids)
                quick_nums = [n for n in quick_nums if n in whitelist]
            node_id_to_pos: Dict[int, int] = {int(nid): i for i, nid in enumerate(node_ids)}
            present = set(int(node_ids[i]) for i in idx0)
            inject_pos: List[int] = []
            for s in quick_nums:
                try:
                    nid = int(s)
                except Exception:
                    continue
                pos = node_id_to_pos.get(nid)
                if pos is not None and (nid not in present):
                    inject_pos.append(pos)
                    present.add(nid)
            if inject_pos:
                idx0 = np.concatenate([idx0, np.asarray(inject_pos, dtype=idx0.dtype)], axis=0)
                vals0 = np.concatenate([vals0, sims[np.asarray(inject_pos)]], axis=0)

    # optional OCR hints
    ocr_nums: List[str] = []
    if use_ocr:
        preproc_opts = {
            "use_contrast": bool(ocr_contrast),
            "use_sharpen": bool(ocr_sharpen),
            "use_adaptive": bool(ocr_adaptive),
            "clahe_clip_limit": float(ocr_clahe_clip),
            "clahe_tile_grid": int(ocr_clahe_grid),
            "sharpen_amount": float(ocr_sharpen_amount),
            "adaptive_block_size": int(ocr_adaptive_block),
            "adaptive_C": int(ocr_adaptive_C),
        }
        readtext_kwargs: Dict[str, any] = {}
        if ocr_text_threshold is not None:
            readtext_kwargs["text_threshold"] = float(ocr_text_threshold)
        if ocr_low_text is not None:
            readtext_kwargs["low_text"] = float(ocr_low_text)
        if ocr_link_threshold is not None:
            readtext_kwargs["link_threshold"] = float(ocr_link_threshold)
        if ocr_decoder is not None:
            readtext_kwargs["decoder"] = str(ocr_decoder)
        if ocr_beam_width is not None:
            readtext_kwargs["beamWidth"] = int(ocr_beam_width)
        if ocr_use_roi:
            ocr_nums = ocr_hints_with_roi(
                image_path,
                languages=ocr_langs,
                max_rois=ocr_max_rois,
                readtext_kwargs=readtext_kwargs,
                preproc_opts=preproc_opts,
                min_confidence=float(ocr_min_conf),
                tta_rotate_deg=float(ocr_tta_deg) if ocr_augment else 0.0,
                tta_scales=ocr_tta_scales if ocr_augment else None,
                tta_max_aug=int(ocr_tta_max_aug) if ocr_augment else 0,
            )
        else:
            if easyocr is None:
                logger.warning("easyocr not available; skipping OCR hints")
            else:
                # Use non-ROI helper with confidence filter
                ocr_nums = ocr_hints(
                    image_path=image_path,
                    languages=ocr_langs,
                    min_confidence=float(ocr_min_conf),
                    readtext_kwargs=readtext_kwargs,
                )
        # Fallback: if still nothing, relax thresholds and expand ROI count
        if (not ocr_nums) and ocr_use_roi:
            try:
                ocr_nums = ocr_hints_with_roi(
                    image_path,
                    languages=ocr_langs,
                    max_rois=max(int(ocr_max_rois) * 2, 30),
                    readtext_kwargs=readtext_kwargs,
                    preproc_opts=preproc_opts,
                    min_confidence=max(float(ocr_min_conf) * 0.5, 0.15),
                )
            except Exception:
                pass
        # Log recognized OCR numbers for debugging
        logger.info(f"OCR raw nums: {ocr_nums}")
        # Post-filter and inject OCR IDs into pool if missing
        if ocr_nums:
            # length filter and optional whitelist
            filtered_nums = [n for n in ocr_nums if len(n) >= int(ocr_digits_min) and len(n) <= int(ocr_digits_max)]
            if ocr_filter_to_nodes:
                node_id_whitelist = set(str(int(k)) for k in node_ids)
                filtered_nums = [n for n in filtered_nums if n in node_id_whitelist]
            if filtered_nums:
                ocr_nums = filtered_nums
                # ensure pool contains these ids
                node_id_to_pos: Dict[int, int] = {int(nid): i for i, nid in enumerate(node_ids)}
                present = set(int(node_ids[i]) for i in idx0)
                inject_pos: List[int] = []
                for s in ocr_nums:
                    try:
                        nid = int(s)
                    except Exception:
                        continue
                    pos = node_id_to_pos.get(nid)
                    if pos is not None and (nid not in present):
                        inject_pos.append(pos)
                        present.add(nid)
                if inject_pos:
                    idx0 = np.concatenate([idx0, np.asarray(inject_pos, dtype=idx0.dtype)], axis=0)
                    vals0 = np.concatenate([vals0, sims[np.asarray(inject_pos)]], axis=0)
        # Now compute OCR adjustment over the (possibly expanded) pool
        ocr_adj = np.zeros_like(vals0)
        if ocr_nums:
            for j, ni in enumerate(idx0):
                nid = int(node_ids[ni])
                rec = node_records[nid]
                score = room_number_match_score(rec.extra, ocr_nums, rec.description)
                ocr_adj[j] = score
    # Additional: direct match between OCR numbers and node_id (fallback when extra metadata is sparse)
    id_adj = np.zeros_like(vals0)
    if use_ocr and ocr_nums:
        oset = set(ocr_nums)
        for j, ni in enumerate(idx0):
            nid = str(int(node_ids[ni]))
            best = 0.0
            # exact match
            if nid in oset:
                best = 1.0
            else:
                # prefix-3 match (e.g., 410x corridor)
                for s in ocr_nums:
                    if len(s) >= 3 and nid.startswith(s[:3]):
                        best = max(best, 0.7)
                    elif len(s) >= 2 and nid.startswith(s[:2]):
                        best = max(best, 0.4)
            id_adj[j] = best

    # optional geometric verification per candidate using node_images_dir
    geo_adj = np.zeros_like(vals0)
    if use_geo and node_images_dir:
        for j, ni in enumerate(idx0):
            nid = int(node_ids[ni])
            ref_paths = collect_reference_images(node_images_dir, nid, limit=int(globals().get("_GEO_REF_LIMIT", 8)))
            if ref_paths:
                geo_adj[j] = geometric_verification_score(image_path, ref_paths)

    # optional graph prior from prev_node
    prior_adj = np.zeros_like(vals0)
    if prev_node is not None:
        for j, ni in enumerate(idx0):
            nid = int(node_ids[ni])
            prior_adj[j] = graph_prior_score(graph, prev_node, nid, alpha=0.7)

    # Graph neighbor consistency bonus: candidates supported by neighbors also in the pool
    consistency_adj = np.zeros_like(vals0)
    # Build quick map from node_id -> index in pool
    pool_id_to_pos: Dict[int, int] = {int(node_ids[ni]): j for j, ni in enumerate(idx0)}
    for j, ni in enumerate(idx0):
        nid = int(node_ids[ni])
        if nid not in graph:
            continue
        neighs = list(graph.neighbors(nid))
        support_vals: List[float] = []
        for nb in neighs:
            pos = pool_id_to_pos.get(int(nb))
            if pos is not None:
                support_vals.append(float(vals0[pos]))
        if support_vals:
            # normalize by max of pool to keep in 0..1-ish (we will normalize later anyway)
            consistency_adj[j] = float(np.mean(support_vals))

    # Normalize auxiliary adjustments to 0..1 for more stable fusion
    vals0_n = normalize01(vals0)
    ocr_adj_n = normalize01(ocr_adj)
    id_adj_n = normalize01(id_adj)
    geo_adj_n = normalize01(geo_adj)
    prior_adj_n = normalize01(prior_adj)
    consistency_adj_n = normalize01(consistency_adj)

    # Dynamic weighting: if no OCR, zero out OCR-related terms
    effective_w_ocr = (w_ocr if (use_ocr and len(ocr_nums) > 0) else 0.0)
    effective_w_id = (w_id if (use_ocr and len(ocr_nums) > 0) else 0.0)

    # combine scores
    combined = (
        w_clip * vals0_n
        + effective_w_ocr * ocr_adj_n
        + effective_w_id * id_adj_n
        + w_geo * geo_adj_n
        + w_prior * prior_adj_n
        + w_consistency * consistency_adj_n
    )

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
            "weights": {
                "w_clip": w_clip,
                "w_ocr": effective_w_ocr,
                "w_id": effective_w_id,
                "w_geo": w_geo,
                "w_prior": w_prior,
                "w_consistency": w_consistency,
            },
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
    ap.add_argument("--node_images_dir", default="node_images/node_images", help="Directory of reference node images")
    ap.add_argument("--use_geo", action="store_true", help="Use geometric verification re-ranking")
    ap.add_argument("--prev_node", type=int, default=None, help="Previous node id for graph prior")
    ap.add_argument("--w_clip", type=float, default=1.0)
    ap.add_argument("--w_ocr", type=float, default=0.3)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--w_prior", type=float, default=0.2)
    ap.add_argument("--w_id", type=float, default=0.6, help="Weight for OCR-to-nodeID direct match")
    ap.add_argument("--w_consistency", type=float, default=0.4, help="Weight for graph neighbor consistency")
    ap.add_argument("--ocr_langs", default="en", help="Comma-separated OCR languages for easyocr (e.g., 'ko,en')")
    ap.add_argument("--ocr_use_roi", action="store_true", help="Enable ROI-based OCR (detect text regions then OCR crops)")
    ap.add_argument("--ocr_max_rois", type=int, default=8, help="Max number of OCR ROIs to try")
    # OCR preprocessing flags
    ap.add_argument("--ocr_contrast", action="store_true", help="Enable CLAHE contrast enhancement for OCR")
    ap.add_argument("--ocr_sharpen", action="store_true", help="Enable unsharp mask sharpening for OCR")
    ap.add_argument("--ocr_adaptive", action="store_true", help="Enable adaptive thresholding for OCR")
    ap.add_argument("--ocr_clahe_clip", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--ocr_clahe_grid", type=int, default=8, help="CLAHE tile grid size")
    ap.add_argument("--ocr_sharpen_amount", type=float, default=0.7, help="Unsharp mask amount")
    ap.add_argument("--ocr_adaptive_block", type=int, default=31, help="Adaptive threshold block size (odd)")
    ap.add_argument("--ocr_adaptive_C", type=int, default=5, help="Adaptive threshold C value (subtracted)")
    # EasyOCR readtext tuning
    ap.add_argument("--ocr_text_threshold", type=float, default=None, help="EasyOCR text_threshold")
    ap.add_argument("--ocr_low_text", type=float, default=None, help="EasyOCR low_text")
    ap.add_argument("--ocr_link_threshold", type=float, default=None, help="EasyOCR link_threshold")
    ap.add_argument("--ocr_decoder", type=str, default=None, help="EasyOCR decoder (greedy or beamsearch)")
    ap.add_argument("--ocr_beam_width", type=int, default=None, help="EasyOCR beamWidth for beamsearch")
    # OCR confidence and filtering
    ap.add_argument("--ocr_min_conf", type=float, default=0.4, help="Minimum OCR confidence to accept a text")
    ap.add_argument("--ocr_filter_to_nodes", action="store_true", help="Filter OCR digits to known node_ids only")
    ap.add_argument("--ocr_digits_min", type=int, default=3, help="Minimum digits length to keep (e.g., 3)")
    ap.add_argument("--ocr_digits_max", type=int, default=4, help="Maximum digits length to keep (e.g., 4)")
    # OCR TTA options
    ap.add_argument("--ocr_augment", action="store_true", help="Enable small-rotation/scale TTA on OCR ROIs")
    ap.add_argument("--ocr_tta_deg", type=float, default=6.0, help="Rotation degrees for OCR TTA (±deg)")
    ap.add_argument("--ocr_tta_scales", type=str, default="0.85,1.0,1.2", help="Comma-separated scales for OCR TTA")
    ap.add_argument("--ocr_tta_max_aug", type=int, default=6, help="Max augmentations per ROI")
    args = ap.parse_args()

    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device={device}")
    ocr_langs = [s.strip() for s in args.ocr_langs.split(",") if s.strip()] if hasattr(args, "ocr_langs") and args.ocr_langs else None
    # Parse OCR TTA scales list
    try:
        _tta_scales = [float(s.strip()) for s in str(getattr(args, "ocr_tta_scales", "0.85,1.0,1.2")).split(",") if s.strip()]
    except Exception:
        _tta_scales = [0.85, 1.0, 1.2]
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
        w_id=float(getattr(args, "w_id", 0.6)),
        w_consistency=float(getattr(args, "w_consistency", 0.4)),
        ocr_langs=ocr_langs,
        ocr_use_roi=bool(getattr(args, "ocr_use_roi", False)),
        ocr_max_rois=int(getattr(args, "ocr_max_rois", 8)),
        ocr_contrast=bool(getattr(args, "ocr_contrast", False)),
        ocr_sharpen=bool(getattr(args, "ocr_sharpen", False)),
        ocr_adaptive=bool(getattr(args, "ocr_adaptive", False)),
        ocr_clahe_clip=float(getattr(args, "ocr_clahe_clip", 2.0)),
        ocr_clahe_grid=int(getattr(args, "ocr_clahe_grid", 8)),
        ocr_sharpen_amount=float(getattr(args, "ocr_sharpen_amount", 0.7)),
        ocr_adaptive_block=int(getattr(args, "ocr_adaptive_block", 31)),
        ocr_adaptive_C=int(getattr(args, "ocr_adaptive_C", 5)),
        ocr_text_threshold=getattr(args, "ocr_text_threshold", None),
        ocr_low_text=getattr(args, "ocr_low_text", None),
        ocr_link_threshold=getattr(args, "ocr_link_threshold", None),
        ocr_decoder=getattr(args, "ocr_decoder", None),
        ocr_beam_width=getattr(args, "ocr_beam_width", None),
        ocr_min_conf=float(getattr(args, "ocr_min_conf", 0.4)),
        ocr_filter_to_nodes=bool(getattr(args, "ocr_filter_to_nodes", False)),
        ocr_digits_min=int(getattr(args, "ocr_digits_min", 3)),
        ocr_digits_max=int(getattr(args, "ocr_digits_max", 4)),
        ocr_augment=bool(getattr(args, "ocr_augment", False)),
        ocr_tta_deg=float(getattr(args, "ocr_tta_deg", 6.0)),
        ocr_tta_scales=_tta_scales,
        ocr_tta_max_aug=int(getattr(args, "ocr_tta_max_aug", 6)),
    )
    print(out)


if __name__ == "__main__":
    main()

