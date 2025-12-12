import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from loguru import logger

# Support running as a module and as a script
try:
    from scripts.localize_image import localize_image, load_openclip, guess_model_for_embed_dim  # type: ignore
    from scripts.map_loader import load_map_csv  # type: ignore
except Exception:
    from localize_image import localize_image, load_openclip, guess_model_for_embed_dim  # type: ignore
    from map_loader import load_map_csv  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


def parse_node_id_from_filename(path: Path) -> int | None:
    """
    Extract leading integer as node id.
    Examples:
      '4101.jpg' -> 4101
      '4101(2).jpg' -> 4101
      '403_a.png' -> 403
    """
    m = re.match(r"^(\d+)", path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def collect_images(images_dir: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(images_dir.glob(pat))
    # Remove duplicates and keep only files
    uniq = []
    seen = set()
    for p in sorted(out):
        if p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(p)
    return uniq


def evaluate(
    images: List[Path],
    csv_path: Path,
    device: str = "cpu",
    topk: int = 5,
    use_ocr: bool = False,
    use_geo: bool = False,
    ocr_use_roi: bool = False,
    ocr_langs: List[str] | None = None,
    node_images_dir: str | None = None,
    w_clip: float = 1.0,
    w_ocr: float = 0.6,
    w_geo: float = 0.5,
    w_prior: float = 0.3,
    w_id: float = 0.6,
    w_consistency: float = 0.4,
    tta: int = 2,
) -> Dict[str, Any]:
    total = 0
    top1_correct = 0
    topk_correct = 0
    per_node: Dict[int, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []

    # Preload map and model once for speed/consistency
    graph, node_records, emb_matrix, node_ids = load_map_csv(str(csv_path))
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    if emb_matrix.size > 0:
        guessed = guess_model_for_embed_dim(int(emb_matrix.shape[1]))
        if guessed is not None:
            model_name, pretrained = guessed
    model, preprocess = load_openclip(model_name=model_name, pretrained=pretrained, device=device)

    for img_path in images:
        gt = parse_node_id_from_filename(img_path)
        if gt is None:
            continue
        total += 1
        try:
            res = localize_image(
                image_path=str(img_path),
                csv_path=str(csv_path),
                device=device,
                tta=int(tta),
                topk=topk,
                use_ocr=use_ocr,
                use_geo=use_geo,
                node_images_dir=str(node_images_dir or images[0].parent),
                prev_node=None,
                w_clip=float(w_clip),
                w_ocr=float(w_ocr),
                w_geo=float(w_geo),
                w_prior=float(w_prior),
                w_id=float(w_id),
                w_consistency=float(w_consistency),
                ocr_langs=ocr_langs,
                ocr_use_roi=bool(ocr_use_roi),
                preloaded_graph=graph,
                preloaded_node_records=node_records,
                preloaded_emb_matrix=emb_matrix,
                preloaded_node_ids=node_ids,
                preloaded_model=model,
                preloaded_preprocess=preprocess,
            )
        except Exception as e:
            logger.warning(f"Failed on {img_path.name}: {e}")
            failures.append({"image": img_path.name, "error": str(e)})
            continue

        pred = int(res.get("current_node", -1))
        cands = [int(c.get("node_id", -1)) for c in (res.get("candidates") or [])]

        correct1 = int(pred == gt)
        correctk = int(gt in cands[:topk])

        top1_correct += correct1
        topk_correct += correctk

        if gt not in per_node:
            per_node[gt] = {"n": 0, "top1": 0, "topk": 0}
        per_node[gt]["n"] += 1
        per_node[gt]["top1"] += correct1
        per_node[gt]["topk"] += correctk

        if not correct1:
            failures.append(
                {
                    "image": img_path.name,
                    "gt": gt,
                    "pred": pred,
                    "topk": cands[:topk],
                    "confidence": float(res.get("confidence", 0.0)),
                }
            )

    top1_acc = (top1_correct / total) if total > 0 else 0.0
    topk_acc = (topk_correct / total) if total > 0 else 0.0
    micro = {
        "count": total,
        "top1_acc": top1_acc,
        "top{}_acc".format(topk): topk_acc,
    }
    # Macro average per node (balanced over nodes even with uneven samples)
    if per_node:
        top1_macro = sum((v["top1"] / v["n"]) for v in per_node.values()) / len(per_node)
        topk_macro = sum((v["topk"] / v["n"]) for v in per_node.values()) / len(per_node)
    else:
        top1_macro = 0.0
    macro = {
        "nodes": len(per_node),
        "top1_macro": top1_macro,
        "top{}_macro".format(topk): topk_macro if per_node else 0.0,
    }

    return {
        "micro": micro,
        "macro": macro,
        "failures": failures[:50],  # truncate
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="node_images/node_images", help="Directory with node images")
    ap.add_argument("--csv", default="ai_4f_node_map_fixed_embeded.csv", help="CSV with embeddings")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda[:idx]")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--use_ocr", action="store_true")
    ap.add_argument("--use_geo", action="store_true")
    ap.add_argument("--ocr_use_roi", action="store_true")
    ap.add_argument("--ocr_langs", default="ko,en", help="Comma-separated OCR languages")
    ap.add_argument("--node_images_dir", default="node_images/node_images", help="Directory of reference node images")
    ap.add_argument("--w_clip", type=float, default=1.0)
    ap.add_argument("--w_ocr", type=float, default=0.6)
    ap.add_argument("--w_geo", type=float, default=0.5)
    ap.add_argument("--w_prior", type=float, default=0.3)
    ap.add_argument("--w_id", type=float, default=0.6)
    ap.add_argument("--w_consistency", type=float, default=0.4)
    ap.add_argument("--tta", type=int, default=2, help="TTA views (1 or 2)")
    ap.add_argument("--limit", type=int, default=100, help="Max images to evaluate (0 for all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_failures", action="store_true", help="Do not include per-sample failures in output")
    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")

    all_images = collect_images(images_dir, ["*.jpg", "*.jpeg", "*.png"])
    # Keep only files that start with digits (have a node id)
    all_images = [p for p in all_images if re.match(r"^\d+", p.stem)]
    if args.limit and args.limit > 0 and len(all_images) > args.limit:
        random.Random(args.seed).shuffle(all_images)
        images = all_images[: args.limit]
    else:
        images = all_images

    logger.info(f"Evaluating on {len(images)} images (from {len(all_images)} total) on device={device}")
    ocr_langs = [s.strip() for s in (args.ocr_langs or "").split(",") if s.strip()] or None
    result = evaluate(
        images=images,
        csv_path=csv_path,
        device=device,
        topk=int(args.topk),
        use_ocr=bool(args.use_ocr),
        use_geo=bool(args.use_geo),
        ocr_use_roi=bool(args.ocr_use_roi),
        ocr_langs=ocr_langs,
        node_images_dir=str(getattr(args, "node_images_dir", images_dir)),
        w_clip=float(args.w_clip),
        w_ocr=float(args.w_ocr),
        w_geo=float(args.w_geo),
        w_prior=float(args.w_prior),
        w_id=float(args.w_id),
        w_consistency=float(args.w_consistency),
        tta=int(args.tta),
    )
    if bool(getattr(args, "no_failures", False)):
        # Keep only summary
        result = {"micro": result["micro"], "macro": result["macro"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




