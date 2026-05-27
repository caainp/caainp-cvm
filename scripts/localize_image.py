import argparse
import glob
import os
import re
import warnings
from typing import Any, Dict, List, Tuple, Optional

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

_openclip_cache: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}
_easyocr_reader_cache: Dict[Tuple[Tuple[str, ...], bool], Any] = {}
_map_csv_cache: Dict[Tuple[str, int], Tuple[Any, Dict[int, Any], np.ndarray, List[int]]] = {}
_sift_detector_cache: Optional[Any] = None
_sift_feature_cache: Dict[Tuple[str, int], Tuple[List[Any], Optional[np.ndarray]]] = {}

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


def _get_easyocr_reader(langs: Tuple[str, ...], gpu: bool) -> Any:
    if easyocr is None:
        return None
    key = (tuple(langs), bool(gpu))
    if key not in _easyocr_reader_cache:
        _easyocr_reader_cache[key] = easyocr.Reader(list(langs), gpu=bool(gpu))
    return _easyocr_reader_cache[key]


def load_map_csv_cached(csv_path: str) -> Tuple[Any, Dict[int, Any], np.ndarray, List[int]]:
    path = os.path.abspath(str(csv_path))
    try:
        mtime_ns = int(os.stat(path).st_mtime_ns)
    except OSError:
        mtime_ns = 0
    key = (path, mtime_ns)
    if key not in _map_csv_cache:
        _map_csv_cache[key] = load_map_csv(csv_path)
    return _map_csv_cache[key]


def _path_cache_key(path: str) -> Tuple[str, int]:
    abs_path = os.path.abspath(str(path))
    try:
        mtime_ns = int(os.stat(abs_path).st_mtime_ns)
    except OSError:
        mtime_ns = 0
    return abs_path, mtime_ns


def _get_sift_detector() -> Optional[Any]:
    global _sift_detector_cache
    if _sift_detector_cache is not None:
        return _sift_detector_cache
    try:
        _sift_detector_cache = cv2.SIFT_create()
    except Exception:
        logger.warning("SIFT unavailable; skipping geometric verification")
        _sift_detector_cache = None
    return _sift_detector_cache


def _get_sift_features(image_path: str) -> Tuple[List[Any], Optional[np.ndarray]]:
    key = _path_cache_key(image_path)
    if key in _sift_feature_cache:
        return _sift_feature_cache[key]
    sift = _get_sift_detector()
    if sift is None:
        result: Tuple[List[Any], Optional[np.ndarray]] = ([], None)
        _sift_feature_cache[key] = result
        return result
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        result = ([], None)
        _sift_feature_cache[key] = result
        return result
    try:
        kp, des = sift.detectAndCompute(img, None)
    except Exception:
        kp, des = [], None
    result = (list(kp or []), des)
    _sift_feature_cache[key] = result
    return result


def load_openclip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
    key = (str(model_name), str(pretrained), str(device))
    if key not in _openclip_cache:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()
        _openclip_cache[key] = (model, preprocess)
    return _openclip_cache[key]


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


# =============================================================================
# [Patch 1] OCR 숫자 정규화 + 도메인 필터
#
# 목적:
# - 잡음 숫자를 필터링 하기 위한 도메인 필터를 넣는다.
# - O→0, I/l→1, S→5 같은 OCR 혼동 정규화를 같이 수행한다.
# - 앞뒤 한글/기호를 제거하고 숫자 후보만 추출한다.
# - 현재 노드맵 범위(4xx ~ 4xxx 중심)를 반영하되,
#   이후 node map 수정 가능성을 고려해 범위를 약간 넓게 둔다.
# =============================================================================

import re
from typing import List

OCR_DOMAIN_MIN = 350
OCR_DOMAIN_MAX = 5999
OCR_ALLOWED_LEADING_DIGITS = {"4", "5"}
OCR_VALID_LENGTHS = {3, 4}

OCR_CHAR_NORMALIZATION = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "S": "5", "s": "5",
    "B": "8", "Z": "2",
})


def _normalize_ocr_text(raw_text: str) -> str:
    """OCR 원문을 숫자 추출에 유리한 형태로 정규화한다."""
    s = str(raw_text).strip()
    s = s.translate(OCR_CHAR_NORMALIZATION)
    s = re.sub(r"[가-힣]+", " ", s)
    s = re.sub(r"[^0-9A-Za-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_numeric_candidates(normalized_text: str) -> List[str]:
    """정규화된 텍스트에서 숫자 후보를 추출한다."""
    candidates: List[str] = []
    candidates.extend(re.findall(r"\d{2,6}", normalized_text))
    for tok in normalized_text.split():
        digits_only = re.sub(r"\D", "", tok)
        if 2 <= len(digits_only) <= 6:
            candidates.append(digits_only)
    return candidates


def _canonicalize_number_string(num_str: str) -> str:
    """앞쪽 0 제거 후 canonical 문자열 반환."""
    s = str(num_str).strip().lstrip("0")
    return s if s else "0"


def _is_plausible_domain_number(num_str: str) -> bool:
    """프로젝트 도메인 범위 내 숫자인지 검사."""
    s = _canonicalize_number_string(num_str)
    if len(s) not in OCR_VALID_LENGTHS:
        return False
    if s[0] not in OCR_ALLOWED_LEADING_DIGITS:
        return False
    try:
        v = int(s)
    except:
        return False
    return OCR_DOMAIN_MIN <= v <= OCR_DOMAIN_MAX


def extract_room_numbers_from_text(texts: List[str]) -> List[str]:
    """OCR raw texts -> 정규화 + 도메인 필터 -> room/node 숫자 후보 리스트 반환"""
    filtered_candidates: List[str] = []

    for t in texts:
        norm = _normalize_ocr_text(t)
        cands = _extract_numeric_candidates(norm)
        for c in cands:
            c = _canonicalize_number_string(c)
            if _is_plausible_domain_number(c):
                filtered_candidates.append(c)

    # 중복 제거 (순서 유지)
    seen = set()
    out: List[str] = []
    for n in filtered_candidates:
        if n not in seen:
            seen.add(n)
            out.append(n)

    # logger 설정이 되어 있다면 디버깅 로그를 남깁니다.
    # logger.debug(f"filtered_room_numbers={out}")
    return out


def extract_room_number_counts_from_text(texts: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in texts:
        norm = _normalize_ocr_text(t)
        for c in _extract_numeric_candidates(norm):
            c = _canonicalize_number_string(c)
            if _is_plausible_domain_number(c):
                counts[c] = counts.get(c, 0) + 1
    return counts


def ocr_number_weights_from_counts(ocr_nums: List[str], counts: Dict[str, int]) -> Dict[str, float]:
    if not ocr_nums:
        return {}
    max_count = max([counts.get(str(n), 0) for n in ocr_nums] or [0])
    if max_count <= 0:
        return {str(n): 1.0 for n in ocr_nums}
    weights: Dict[str, float] = {}
    for n in ocr_nums:
        c = counts.get(str(n), 0)
        weights[str(n)] = 0.35 + 0.65 * (float(c) / float(max_count))
    return weights


def _ocr_item_parts(item: Any) -> Tuple[Optional[Any], str, Optional[float]]:
    if isinstance(item, str):
        return None, item, None
    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            try:
                return item[0], str(item[1]), float(item[2])
            except Exception:
                return item[0], str(item[1]), None
        if len(item) >= 2:
            return item[0], str(item[1]), None
    return None, str(item), None


def _ocr_box_xyxy(
    box: Any,
    *,
    origin_x: float,
    origin_y: float,
    coord_scale: float,
    image_w: int,
    image_h: int,
) -> Optional[List[int]]:
    if box is None:
        return None
    try:
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return None
        div = max(float(coord_scale), 1e-6)
        xs = (pts[:, 0] / div) + float(origin_x)
        ys = (pts[:, 1] / div) + float(origin_y)
        x1 = int(max(0, min(image_w - 1, np.floor(float(xs.min())))))
        y1 = int(max(0, min(image_h - 1, np.floor(float(ys.min())))))
        x2 = int(max(0, min(image_w - 1, np.ceil(float(xs.max())))))
        y2 = int(max(0, min(image_h - 1, np.ceil(float(ys.max())))))
        return [x1, y1, x2, y2]
    except Exception:
        return None


def _bbox_norm_xyxy(bbox: Optional[List[int]], image_w: int, image_h: int) -> Optional[List[float]]:
    if bbox is None or image_w <= 0 or image_h <= 0:
        return None
    x1, y1, x2, y2 = bbox
    return [
        round(float(x1) / float(image_w), 4),
        round(float(y1) / float(image_h), 4),
        round(float(x2) / float(image_w), 4),
        round(float(y2) / float(image_h), 4),
    ]


def _bbox_area_ratio(bbox: Optional[List[int]], image_w: int, image_h: int) -> Optional[float]:
    if bbox is None or image_w <= 0 or image_h <= 0:
        return None
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return round(float(area) / float(image_w * image_h), 6)


def _bbox_position_label(bbox: Optional[List[int]], image_w: int, image_h: int) -> Optional[str]:
    if bbox is None or image_w <= 0 or image_h <= 0:
        return None
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / float(image_w)
    cy = (y1 + y2) / 2.0 / float(image_h)
    horizontal = "left" if cx < 0.33 else "center" if cx < 0.67 else "right"
    vertical = "top" if cy < 0.33 else "middle" if cy < 0.67 else "bottom"
    return f"{horizontal}_{vertical}"


def _readtext_with_observations(
    reader: Any,
    image: np.ndarray,
    *,
    source: str,
    image_w: int,
    image_h: int,
    readtext_kwargs: Optional[Dict[str, Any]] = None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    coord_scale: float = 1.0,
    roi_index: Optional[int] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    crop_variant: Optional[str] = None,
    scale: float = 1.0,
    preprocessed: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    try:
        items = reader.readtext(image, detail=1, **(readtext_kwargs or {})) or []
    except Exception:
        return [], []

    texts: List[str] = []
    observations: List[Dict[str, Any]] = []
    for item in items:
        box, text, conf = _ocr_item_parts(item)
        texts.append(text)
        bbox = _ocr_box_xyxy(
            box,
            origin_x=origin_x,
            origin_y=origin_y,
            coord_scale=coord_scale,
            image_w=image_w,
            image_h=image_h,
        )
        obs: Dict[str, Any] = {
            "text": text,
            "numbers": extract_room_numbers_from_text([text]),
            "confidence": round(float(conf), 4) if conf is not None else None,
            "source": source,
            "bbox_xyxy": bbox,
            "bbox_norm_xyxy": _bbox_norm_xyxy(bbox, image_w, image_h),
            "bbox_area_ratio": _bbox_area_ratio(bbox, image_w, image_h),
            "position": _bbox_position_label(bbox, image_w, image_h),
            "roi_index": roi_index,
            "roi_bbox_xywh": list(roi_bbox) if roi_bbox is not None else None,
            "crop_variant": crop_variant,
            "scale": round(float(scale), 3),
            "preprocessed": bool(preprocessed),
        }
        observations.append(obs)
    return texts, observations


def summarize_ocr_observations(observations: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    source_summary: Dict[str, Any] = {
        "total_observations": len(observations),
        "number_observations": 0,
        "sources": {},
    }
    per_number: Dict[str, Dict[str, Any]] = {}

    for obs in observations:
        source = str(obs.get("source") or "unknown")
        sources = source_summary["sources"]
        if source not in sources:
            sources[source] = {"observations": 0, "number_observations": 0}
        sources[source]["observations"] += 1

        nums = obs.get("numbers") or []
        if nums:
            source_summary["number_observations"] += 1
            sources[source]["number_observations"] += 1

        for num in nums:
            key = str(num)
            entry = per_number.setdefault(
                key,
                {
                    "number": key,
                    "count": 0,
                    "sources": {},
                    "positions": [],
                    "sample_texts": [],
                    "max_confidence": None,
                    "min_area_ratio": None,
                    "max_area_ratio": None,
                },
            )
            entry["count"] += 1
            entry["sources"][source] = entry["sources"].get(source, 0) + 1
            pos = obs.get("position")
            if pos is not None and pos not in entry["positions"]:
                entry["positions"].append(pos)
            text = obs.get("text")
            if text and text not in entry["sample_texts"] and len(entry["sample_texts"]) < 5:
                entry["sample_texts"].append(text)
            conf = obs.get("confidence")
            if conf is not None:
                entry["max_confidence"] = conf if entry["max_confidence"] is None else max(entry["max_confidence"], conf)
            area = obs.get("bbox_area_ratio")
            if area is not None:
                entry["min_area_ratio"] = area if entry["min_area_ratio"] is None else min(entry["min_area_ratio"], area)
                entry["max_area_ratio"] = area if entry["max_area_ratio"] is None else max(entry["max_area_ratio"], area)

    number_sources = sorted(per_number.values(), key=lambda d: (-int(d["count"]), str(d["number"])))
    return source_summary, number_sources


def ocr_hints(image_path: str, languages: Optional[List[str]] = None) -> List[str]:
    if easyocr is None:
        logger.warning("easyocr not available; skipping OCR hints")
        return []
    # Default to Korean+English when not specified
    langs = languages if (languages and len(languages) > 0) else ["ko", "en"]
    # Use GPU if available to avoid CPU warning and speed up OCR
    reader = _get_easyocr_reader(tuple(langs), torch.cuda.is_available())
    result = reader.readtext(image_path, detail=0)
    return extract_room_numbers_from_text(result)


def preprocess_image_for_ocr(
    image_bgr: np.ndarray,
    use_grayscale: bool = False,
    upscale_factor: float = 1.0,
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
    OCR 전처리:
    - grayscale
    - upscale
    - CLAHE contrast
    - sharpen
    - adaptive threshold

    반환:
    - 일반 전처리면 BGR 또는 Gray image
    - adaptive threshold 사용 시 binary(gray) image
    """
    img = image_bgr.copy()

    # 1) grayscale
    if use_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) upscale
    if upscale_factor is not None and float(upscale_factor) > 1.0:
        scale = float(upscale_factor)
        h, w = img.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_CUBIC if scale >= 1.5 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 3) contrast (CLAHE)
    if use_contrast:
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(
                clipLimit=float(clahe_clip_limit),
                tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
            )
            img = clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=float(clahe_clip_limit),
                tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
            )
            L = clahe.apply(L)
            lab = cv2.merge([L, A, B])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4) sharpen
    if use_sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
        img = cv2.addWeighted(
            img,
            1.0 + float(sharpen_amount),
            blur,
            -float(sharpen_amount),
            0,
        )

    # 5) adaptive threshold
    if use_adaptive:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        block = int(adaptive_block_size)
        if block % 2 == 0:
            block += 1

        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            int(adaptive_C),
        )
        return th

    return img


def _expand_roi(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.25,
    min_pad_px: int = 12,
) -> Tuple[int, int, int, int]:
    pad_x = max(int(round(w * pad_ratio)), int(min_pad_px))
    pad_y = max(int(round(h * pad_ratio)), int(min_pad_px))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(img_h, y + h + pad_y)

    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def detect_text_rois(
    image_path: str,
    max_rois: int = 12,
    min_area: int = 180,
    max_area: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Morphology 기반 text ROI detector.
    반환: (x, y, w, h) 리스트
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    if max_area is None:
        max_area = int(0.9 * H * W)

    # 1) text-like 구조 강조
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = cv2.GaussianBlur(grad_x, (3, 3), 0)

    _, th = cv2.threshold(
        grad_x,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # 2) 글자들을 한 줄로 묶기
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # 3) 작은 점 노이즈 제거
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: List[Tuple[int, int, int, int]] = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < min_area or area > max_area:
            continue

        ar = w / float(h + 1e-6)

        # 번호판/텍스트 라인 후보 완화된 비율 조건
        if ar < 1.0 or ar > 30.0:
            continue

        # 너무 얇거나 너무 낮은 박스 제거
        if w < 18 or h < 8:
            continue

        # ROI 확장
        ex, ey, ew, eh = _expand_roi(
            x, y, w, h, W, H,
            pad_ratio=0.30,
            min_pad_px=14,
        )
        rois.append((ex, ey, ew, eh))

    # 정렬: 큰 영역 우선
    rois = sorted(rois, key=lambda r: r[2] * r[3], reverse=True)

    def iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter + 1e-6
        return inter / union

    # 중복 제거
    filtered: List[Tuple[int, int, int, int]] = []

    for r in rois:
        if all(iou(r, f) < 0.35 for f in filtered):
            filtered.append(r)
        if len(filtered) >= max_rois:
            break

    return filtered


def ocr_hints_with_roi(
    image_path: str,
    languages: Optional[List[str]] = None,
    max_rois: int = 12,
    readtext_kwargs: Optional[Dict[str, Any]] = None,
    preproc_opts: Optional[Dict[str, Any]] = None,
    return_texts: bool = False,
) -> Any:
    if easyocr is None:
        logger.warning("easyocr not available; skipping OCR hints")
        return ([], []) if return_texts else []

    langs = languages if (languages and len(languages) > 0) else ["ko", "en"]
    reader = _get_easyocr_reader(tuple(langs), torch.cuda.is_available())

    texts: List[str] = []

    img_full = cv2.imread(image_path)
    if img_full is None:
        return ([], []) if return_texts else []

    # -----------------------------------------------------------------
    # 1) 전체 이미지 OCR
    # -----------------------------------------------------------------
    try:
        if preproc_opts is not None:
            proc_full = preprocess_image_for_ocr(img_full, **preproc_opts)
            texts.extend(reader.readtext(proc_full, detail=0, **(readtext_kwargs or {})) or [])
        else:
            texts.extend(reader.readtext(img_full, detail=0, **(readtext_kwargs or {})) or [])
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 2) ROI 탐지
    # -----------------------------------------------------------------
    rois = detect_text_rois(image_path, max_rois=max_rois)

    # -----------------------------------------------------------------
    # 3) 각 ROI에 대해 multi-crop / multi-scale OCR
    # -----------------------------------------------------------------
    roi_scales = [1.0, 1.5, 2.0]

    for (x, y, w, h) in rois:
        crop = img_full[y:y + h, x:x + w]
        if crop is None or crop.size == 0:
            continue

        roi_variants: List[np.ndarray] = [crop]

        # 추가 확장 crop 한 번 더 생성
        H, W = img_full.shape[:2]
        x2, y2, w2, h2 = _expand_roi(
            x, y, w, h, W, H,
            pad_ratio=0.45,
            min_pad_px=20,
        )
        crop_loose = img_full[y2:y2 + h2, x2:x2 + w2]
        if crop_loose is not None and crop_loose.size > 0:
            roi_variants.append(crop_loose)

        for base_crop in roi_variants:
            for scale in roi_scales:
                try:
                    cur = base_crop
                    if scale > 1.0:
                        bh, bw = cur.shape[:2]
                        cur = cv2.resize(
                            cur,
                            (max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))),
                            interpolation=cv2.INTER_CUBIC,
                        )

                    # 원본 crop OCR
                    texts.extend(reader.readtext(cur, detail=0, **(readtext_kwargs or {})) or [])

                    # 전처리 crop OCR
                    if preproc_opts is not None:
                        proc = preprocess_image_for_ocr(cur, **preproc_opts)
                        texts.extend(reader.readtext(proc, detail=0, **(readtext_kwargs or {})) or [])
                except Exception:
                    continue

    nums = extract_room_numbers_from_text(texts)
    if return_texts:
        return nums, [str(t) for t in texts]
    return nums


def ocr_hints_with_roi_details(
    image_path: str,
    languages: Optional[List[str]] = None,
    max_rois: int = 12,
    readtext_kwargs: Optional[Dict[str, Any]] = None,
    preproc_opts: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    if easyocr is None:
        logger.warning("easyocr not available; skipping OCR hints")
        return [], [], []

    langs = languages if (languages and len(languages) > 0) else ["ko", "en"]
    reader = _get_easyocr_reader(tuple(langs), torch.cuda.is_available())

    img_full = cv2.imread(image_path)
    if img_full is None:
        return [], [], []
    H_full, W_full = img_full.shape[:2]
    preproc_upscale = float((preproc_opts or {}).get("upscale_factor", 1.0) or 1.0)

    texts: List[str] = []
    observations: List[Dict[str, Any]] = []

    if preproc_opts is not None:
        proc_full = preprocess_image_for_ocr(img_full, **preproc_opts)
        t, obs = _readtext_with_observations(
            reader,
            proc_full,
            source="full_preprocessed",
            image_w=W_full,
            image_h=H_full,
            readtext_kwargs=readtext_kwargs,
            coord_scale=preproc_upscale,
            preprocessed=True,
        )
    else:
        t, obs = _readtext_with_observations(
            reader,
            img_full,
            source="full_raw",
            image_w=W_full,
            image_h=H_full,
            readtext_kwargs=readtext_kwargs,
        )
    texts.extend(t)
    observations.extend(obs)

    rois = detect_text_rois(image_path, max_rois=max_rois)
    roi_scales = [1.0, 1.5, 2.0]

    for roi_index, (x, y, w, h) in enumerate(rois):
        crop = img_full[y:y + h, x:x + w]
        if crop is None or crop.size == 0:
            continue

        roi_variants: List[Tuple[np.ndarray, int, int, Tuple[int, int, int, int], str]] = [
            (crop, x, y, (x, y, w, h), "tight")
        ]
        x2, y2, w2, h2 = _expand_roi(
            x, y, w, h, W_full, H_full,
            pad_ratio=0.45,
            min_pad_px=20,
        )
        crop_loose = img_full[y2:y2 + h2, x2:x2 + w2]
        if crop_loose is not None and crop_loose.size > 0:
            roi_variants.append((crop_loose, x2, y2, (x2, y2, w2, h2), "loose"))

        for base_crop, origin_x, origin_y, roi_bbox, crop_variant in roi_variants:
            for scale in roi_scales:
                cur = base_crop
                if scale > 1.0:
                    bh, bw = cur.shape[:2]
                    cur = cv2.resize(
                        cur,
                        (max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))),
                        interpolation=cv2.INTER_CUBIC,
                    )

                t, obs = _readtext_with_observations(
                    reader,
                    cur,
                    source=f"roi_{crop_variant}_raw",
                    image_w=W_full,
                    image_h=H_full,
                    readtext_kwargs=readtext_kwargs,
                    origin_x=float(origin_x),
                    origin_y=float(origin_y),
                    coord_scale=float(scale),
                    roi_index=roi_index,
                    roi_bbox=roi_bbox,
                    crop_variant=crop_variant,
                    scale=float(scale),
                    preprocessed=False,
                )
                texts.extend(t)
                observations.extend(obs)

                if preproc_opts is not None:
                    proc = preprocess_image_for_ocr(cur, **preproc_opts)
                    t, obs = _readtext_with_observations(
                        reader,
                        proc,
                        source=f"roi_{crop_variant}_preprocessed",
                        image_w=W_full,
                        image_h=H_full,
                        readtext_kwargs=readtext_kwargs,
                        origin_x=float(origin_x),
                        origin_y=float(origin_y),
                        coord_scale=float(scale) * preproc_upscale,
                        roi_index=roi_index,
                        roi_bbox=roi_bbox,
                        crop_variant=crop_variant,
                        scale=float(scale),
                        preprocessed=True,
                    )
                    texts.extend(t)
                    observations.extend(obs)

    nums = extract_room_numbers_from_text(texts)
    return nums, [str(t) for t in texts], observations


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except Exception:
        pass
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def _canonical_int_tokens(value: Any) -> List[int]:
    if _is_missing_value(value):
        return []
    out: List[int] = []
    for tok in re.findall(r"\d{2,5}", str(value)):
        try:
            out.append(int(tok))
        except Exception:
            continue
    return out


def _description_has_number(description: Optional[str], number: int) -> bool:
    if _is_missing_value(description):
        return False
    return re.search(rf"(?<!\d){int(number)}(?!\d)", str(description)) is not None


def _is_room_type(node_type: Optional[str]) -> bool:
    return str(node_type or "").upper() == "ROOM"


def _has_strong_room_identifier(
    *,
    node_id: Optional[int],
    node_type: Optional[str],
    description: Optional[str],
    anchor_rooms: List[int],
    number: int,
) -> bool:
    if not _is_room_type(node_type):
        return False
    if number in anchor_rooms:
        return True
    if _description_has_number(description, number):
        return True
    # Do not treat an internal ROOM node id as visible signage unless metadata/description confirms it.
    return False


def parse_room_range(value: Optional[str]) -> List[Tuple[int, int]]:
    if _is_missing_value(value):
        return []
    ranges: List[Tuple[int, int]] = []
    parts = re.split(r"[;,]", str(value))
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
    node_meta: Dict[str, Any],
    ocr_nums: List[str],
    description: Optional[str],
    node_id: Optional[int] = None,
    node_type: Optional[str] = None,
    ocr_num_weights: Optional[Dict[str, float]] = None,
) -> float:
    if not ocr_nums:
        return 0.0
    anchor_room = node_meta.get("anchor_room")
    room_range = node_meta.get("room_range")

    ranges = parse_room_range(room_range if isinstance(room_range, str) else str(room_range) if room_range is not None else None)
    anchor_rooms = _canonical_int_tokens(anchor_room)

    score = 0.0
    for n in ocr_nums:
        try:
            vn = int(n)
        except Exception:
            continue
        weight = float((ocr_num_weights or {}).get(str(n), 1.0))
        raw_score = 0.0
        strong_room_identifier = _has_strong_room_identifier(
            node_id=node_id,
            node_type=node_type,
            description=description,
            anchor_rooms=anchor_rooms,
            number=vn,
        )
        # Exact OCR evidence should dominate only for visible ROOM identifiers.
        if strong_room_identifier and node_id is not None:
            try:
                if int(node_id) == vn:
                    raw_score = max(raw_score, 2.0)
            except Exception:
                pass
        if vn in anchor_rooms:
            if _is_room_type(node_type):
                raw_score = max(raw_score, 1.8)
            else:
                raw_score = max(raw_score, 0.25)
        if _description_has_number(description, vn):
            if _is_room_type(node_type):
                raw_score = max(raw_score, 1.6)
            else:
                raw_score = max(raw_score, 0.25)
        # Range evidence is useful but must stay weaker than an exact room hit.
        for a, b in ranges:
            if a <= vn <= b:
                raw_score = max(raw_score, 0.35)
                break
        score = max(score, raw_score * weight)
    return score


def _ocr_token_matches_node_id(node_id: int, ocr_token: str) -> bool:
    try:
        return int(str(ocr_token).strip()) == int(node_id)
    except Exception:
        return False


def _ocr_token_matches_visible_room_node(rec: Any, node_id: int, ocr_token: str) -> bool:
    try:
        vn = int(str(ocr_token).strip())
    except Exception:
        return False
    if not _ocr_token_matches_node_id(node_id, ocr_token):
        return False
    extra = getattr(rec, "extra", {}) or {}
    return _has_strong_room_identifier(
        node_id=node_id,
        node_type=getattr(rec, "node_type", None),
        description=getattr(rec, "description", None),
        anchor_rooms=_canonical_int_tokens(extra.get("anchor_room")),
        number=vn,
    )


def geometric_verification_score(
    query_image_path: str,
    reference_image_paths: List[str],
) -> float:
    if not reference_image_paths:
        return 0.0
    if _get_sift_detector() is None:
        return 0.0

    kp_q, des_q = _get_sift_features(query_image_path)
    if des_q is None or len(kp_q) < 8:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    best_norm_inliers = 0.0
    tested = 0

    for ref_path in reference_image_paths:
        try:
            kp_r, des_r = _get_sift_features(ref_path)
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


def _clean_debug_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def _build_fusion_debug_rows(
    idx0: np.ndarray,
    node_ids: np.ndarray,
    node_records: Dict[int, Any],
    vals0: np.ndarray,
    ocr_adj: np.ndarray,
    geo_adj: np.ndarray,
    prior_adj: np.ndarray,
    combined: np.ndarray,
    *,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    order = np.argsort(-combined)[: max(0, int(limit))]
    rows: List[Dict[str, Any]] = []
    for rank, j in enumerate(order, start=1):
        ni = int(idx0[j])
        nid = int(node_ids[ni])
        rec = node_records.get(nid)
        extra = getattr(rec, "extra", {}) or {}
        rows.append({
            "rank": rank,
            "node_id": nid,
            "node_type": getattr(rec, "node_type", None),
            "description": getattr(rec, "description", None),
            "anchor_room": _clean_debug_value(extra.get("anchor_room")),
            "room_range": _clean_debug_value(extra.get("room_range")),
            "clip_score": round(float(vals0[j]), 6),
            "ocr_score": round(float(ocr_adj[j]), 6),
            "geo_score": round(float(geo_adj[j]), 6),
            "prior_score": round(float(prior_adj[j]), 6),
            "combined_score": round(float(combined[j]), 6),
        })
    return rows


def localize_image(
    image_path: str,
    csv_path: str,
    device: str = "cpu",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    topk: int = 5,
    clip_pool_size: int = 50,
    ocr_merge_min_score: float = 0.4,
    use_ocr: bool = False,
    node_images_dir: Optional[str] = None,
    use_geo: bool = False,
    geo_candidate_limit: int = 10,
    geo_ref_limit: int = 4,
    prev_node: Optional[int] = None,
    w_clip: float = 1.0,
    w_ocr: float = 0.3,
    w_geo: float = 0.4,
    w_prior: float = 0.2,
    auto_match_model: bool = True,
    ocr_langs: Optional[List[str]] = None,
    ocr_use_roi: bool = False,
    ocr_max_rois: int = 8,
    # OCR preprocessing flags
    ocr_grayscale: bool = False,
    ocr_upscale: float = 1.0,
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
    ocr_debug_max_observations: int = 200,
) -> dict:
    # load map
    graph, node_records, emb_matrix, node_ids = load_map_csv_cached(csv_path)
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

    # Wide CLIP shortlist for re-ranking; OCR can then merge exact/range matches outside that pool.
    k_pool = max(int(topk), int(clip_pool_size), 10)
    k_pool = min(k_pool, int(sims.shape[0]))
    idx0, vals0 = topk_indices(sims, k_pool)
    vals0 = vals0.astype(np.float32)
    initial_clip_pool_size = int(len(idx0))

    # optional OCR hints
    ocr_nums: List[str] = []
    ocr_raw_texts: List[str] = []
    ocr_observations: List[Dict[str, Any]] = []
    ocr_source_summary: Dict[str, Any] = {}
    ocr_number_sources: List[Dict[str, Any]] = []
    ocr_num_counts: Dict[str, int] = {}
    ocr_num_weights: Dict[str, float] = {}
    ocr_pool_merge_added = 0
    ocr_merged_node_ids: List[int] = []
    langs_eff: List[str] = list(ocr_langs) if ocr_langs else ["ko", "en"]
    if use_ocr:
        preproc_opts = {
            "use_grayscale": bool(ocr_grayscale),
            "upscale_factor": float(ocr_upscale),
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
            ocr_nums, ocr_raw_texts, ocr_observations = ocr_hints_with_roi_details(
                image_path,
                languages=langs_eff,
                max_rois=ocr_max_rois,
                readtext_kwargs=readtext_kwargs,
                preproc_opts=preproc_opts,
            )
        else:
            if easyocr is None:
                logger.warning("easyocr not available; skipping OCR hints")
            else:
                reader = _get_easyocr_reader(tuple(langs_eff), torch.cuda.is_available())
                img_full = cv2.imread(image_path)
                if img_full is not None:
                    img_full = preprocess_image_for_ocr(img_full, **preproc_opts)
                    try:
                        raw_img = cv2.imread(image_path)
                        H_full, W_full = raw_img.shape[:2] if raw_img is not None else img_full.shape[:2]
                        preproc_upscale = float(preproc_opts.get("upscale_factor", 1.0) or 1.0)
                        ocr_raw_texts, ocr_observations = _readtext_with_observations(
                            reader,
                            img_full,
                            source="full_preprocessed",
                            image_w=W_full,
                            image_h=H_full,
                            readtext_kwargs=readtext_kwargs,
                            coord_scale=preproc_upscale,
                            preprocessed=True,
                        )
                        ocr_nums = extract_room_numbers_from_text(ocr_raw_texts)
                    except Exception:
                        ocr_nums = []
        # Log recognized OCR numbers for debugging
        logger.info(f"OCR raw nums: {ocr_nums} (langs={langs_eff})")
        ocr_source_summary, ocr_number_sources = summarize_ocr_observations(ocr_observations)
        ocr_num_counts = extract_room_number_counts_from_text(ocr_raw_texts)
        ocr_num_weights = ocr_number_weights_from_counts(ocr_nums, ocr_num_counts)
        if ocr_nums:
            node_id_to_row = {int(node_ids[i]): i for i in range(len(node_ids))}
            pool_set = set(int(x) for x in idx0.tolist())
            n_before = len(pool_set)
            thr = float(ocr_merge_min_score)
            for nid, rec in node_records.items():
                if nid not in node_id_to_row:
                    continue
                sc = room_number_match_score(
                    rec.extra,
                    ocr_nums,
                    rec.description,
                    node_id=nid,
                    node_type=rec.node_type,
                    ocr_num_weights=ocr_num_weights,
                )
                id_hit = any(_ocr_token_matches_visible_room_node(rec, nid, on) for on in ocr_nums)
                if id_hit or sc >= thr:
                    row_idx = int(node_id_to_row[nid])
                    if row_idx not in pool_set:
                        ocr_merged_node_ids.append(int(nid))
                    pool_set.add(row_idx)
            if len(pool_set) > n_before:
                ocr_pool_merge_added = len(pool_set) - n_before
                logger.info(
                    f"OCR pool merge: added {ocr_pool_merge_added} nodes "
                    f"(min_match_score>={thr} or OCR digit == node_id)"
                )
            idx0 = np.array(sorted(pool_set), dtype=np.int64)
            vals0 = sims[idx0].astype(np.float32)

    ocr_adj = np.zeros(len(idx0), dtype=np.float32)
    if use_ocr and ocr_nums:
        for j, ni in enumerate(idx0):
            nid = int(node_ids[ni])
            rec = node_records[nid]
            ocr_adj[j] = room_number_match_score(
                rec.extra,
                ocr_nums,
                rec.description,
                node_id=nid,
                node_type=rec.node_type,
                ocr_num_weights=ocr_num_weights,
            )

    # optional graph prior from prev_node
    prior_adj = np.zeros_like(vals0)
    if prev_node is not None:
        for j, ni in enumerate(idx0):
            nid = int(node_ids[ni])
            prior_adj[j] = graph_prior_score(graph, prev_node, nid, alpha=0.7)

    # optional geometric verification per candidate using node_images_dir
    geo_adj = np.zeros_like(vals0)
    geo_evaluated_count = 0
    if use_geo and node_images_dir:
        geo_order = np.arange(len(idx0))
        if geo_candidate_limit is not None and int(geo_candidate_limit) > 0 and len(geo_order) > int(geo_candidate_limit):
            pre_geo = w_clip * vals0 + w_ocr * ocr_adj + w_prior * prior_adj
            geo_order = np.argsort(-pre_geo)[: int(geo_candidate_limit)]
        for j in geo_order:
            ni = int(idx0[int(j)])
            nid = int(node_ids[ni])
            ref_paths = collect_reference_images(node_images_dir, nid, limit=int(geo_ref_limit))
            if ref_paths:
                geo_adj[int(j)] = geometric_verification_score(image_path, ref_paths)
            geo_evaluated_count += 1

    # combine scores
    combined = w_clip * vals0 + w_ocr * ocr_adj + w_geo * geo_adj + w_prior * prior_adj
    fusion_debug = _build_fusion_debug_rows(
        idx0=idx0,
        node_ids=node_ids,
        node_records=node_records,
        vals0=vals0,
        ocr_adj=ocr_adj,
        geo_adj=geo_adj,
        prior_adj=prior_adj,
        combined=combined,
        limit=max(int(topk), 20),
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
            "ocr_raw_texts": ocr_raw_texts,
            "ocr_observation_count": int(len(ocr_observations)),
            "ocr_observations": ocr_observations[: max(0, int(ocr_debug_max_observations))],
            "ocr_source_summary": ocr_source_summary,
            "ocr_number_sources": ocr_number_sources,
            "ocr_num_counts": ocr_num_counts,
            "ocr_num_weights": {k: round(float(v), 4) for k, v in ocr_num_weights.items()},
            "ocr_langs": list(langs_eff) if use_ocr else [],
            "clip_pool_size": int(k_pool),
            "initial_clip_pool_size": initial_clip_pool_size,
            "final_pool_size": int(len(idx0)),
            "clip_pool_after_merge": int(len(idx0)),
            "ocr_pool_merge_added": int(ocr_pool_merge_added),
            "ocr_merged_node_ids": ocr_merged_node_ids,
            "ocr_merge_min_score": float(ocr_merge_min_score),
            "geo_candidate_limit": int(geo_candidate_limit) if geo_candidate_limit is not None else None,
            "geo_evaluated_count": int(geo_evaluated_count),
            "geo_ref_limit": int(geo_ref_limit),
            "fusion_candidates": fusion_debug,
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
    ap.add_argument("--node_images_dir", default="node_images/node_images", help="Directory of reference node images")
    ap.add_argument("--use_geo", action="store_true", help="Use geometric verification re-ranking")
    ap.add_argument("--clip_pool_size", "--clip-pool-size", type=int, default=50, help="CLIP shortlist size before OCR/geo fusion")
    ap.add_argument("--ocr_merge_min_score", "--ocr-merge-min-score", type=float, default=0.4, help="Merge nodes into the pool when OCR match score is at least this value")
    ap.add_argument("--geo_candidate_limit", "--geo-candidate-limit", type=int, default=10, help="Run SIFT geometry only on the top N pre-geo fusion candidates; <=0 means all")
    ap.add_argument("--geo_ref_limit", "--geo-ref-limit", type=int, default=4, help="Reference images per node for geometric verification")
    ap.add_argument("--prev_node", type=int, default=None, help="Previous node id for graph prior")
    ap.add_argument("--w_clip", type=float, default=1.0)
    ap.add_argument("--w_ocr", type=float, default=0.3)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--w_prior", type=float, default=0.2)
    ap.add_argument("--ocr_langs", default="en", help="Comma-separated OCR languages for easyocr (e.g., 'ko,en')")
    ap.add_argument("--ocr_use_roi", action="store_true", help="Enable ROI-based OCR (detect text regions then OCR crops)")
    ap.add_argument("--ocr_max_rois", type=int, default=8, help="Max number of OCR ROIs to try")
    # OCR preprocessing flags
    ap.add_argument("--ocr_grayscale", action="store_true", help="Enable grayscale preprocessing for OCR")
    ap.add_argument("--ocr_upscale", type=float, default=1.0, help="Upscale factor for OCR preprocessing (e.g. 2.0)")
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
    ap.add_argument("--ocr_debug_max_observations", type=int, default=200, help="Max OCR observation rows to include in debug output")
    args = ap.parse_args()

    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device={device}")
    ocr_langs = [s.strip() for s in args.ocr_langs.split(",") if s.strip()] if hasattr(args, "ocr_langs") and args.ocr_langs else None
    out = localize_image(
        image_path=args.image,
        csv_path=args.csv,
        device=device,
        model_name=args.model,
        pretrained=args.pretrained,
        topk=args.topk,
        clip_pool_size=int(getattr(args, "clip_pool_size", 50)),
        ocr_merge_min_score=float(getattr(args, "ocr_merge_min_score", 0.4)),
        use_ocr=args.use_ocr,
        node_images_dir=args.node_images_dir,
        use_geo=args.use_geo,
        geo_candidate_limit=int(getattr(args, "geo_candidate_limit", 10)),
        geo_ref_limit=int(getattr(args, "geo_ref_limit", 4)),
        prev_node=args.prev_node,
        w_clip=args.w_clip,
        w_ocr=args.w_ocr,
        w_geo=args.w_geo,
        w_prior=args.w_prior,
        ocr_langs=ocr_langs,
        ocr_use_roi=bool(getattr(args, "ocr_use_roi", False)),
        ocr_max_rois=int(getattr(args, "ocr_max_rois", 8)),
        ocr_grayscale=bool(getattr(args, "ocr_grayscale", False)),
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
        ocr_debug_max_observations=int(getattr(args, "ocr_debug_max_observations", 200)),
    )
    print(out)


if __name__ == "__main__":
    main()

