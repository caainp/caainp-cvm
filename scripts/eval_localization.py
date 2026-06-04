from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from loguru import logger

# 레포 루트에서 python -m scripts.eval_localization 로 실행할 때 / 직접 실행할 때 둘 다 지원
try:
    from scripts.localize_image import localize_image  # type: ignore
    from scripts.map_loader import load_map_csv  # type: ignore
except ImportError:
    from localize_image import localize_image  # type: ignore
    from map_loader import load_map_csv  # type: ignore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 지원 파일명 규칙
# 401_L.jpg
# 401_R.jpg
# 4150_R_1.jpg
# 4150_L_3.jpg
FILENAME_RE = re.compile(
    r"^(?P<node>\d+)_(?P<angle>L|R)(?:_(?P<variant>\d+))?$",
    re.IGNORECASE,
)


def parse_gt_from_filename(path: Path) -> Tuple[int, str, Optional[int]]:
    """
    파일명에서 GT node / angle / variant 파싱
    예:
      401_L.jpg    -> (401, "L", None)
      4150_R_2.jpg -> (4150, "R", 2)
    """
    stem = path.stem
    m = FILENAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"파일명 규칙이 맞지 않습니다: {path.name} "
            f"(예: 401_L.jpg, 402_R.jpg, 4150_R_2.jpg)"
        )

    gt_node = int(m.group("node"))
    angle = m.group("angle").upper()
    variant = int(m.group("variant")) if m.group("variant") else None
    return gt_node, angle, variant


def collect_images(test_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in test_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def shortest_path_distance_or_none(
    graph: nx.Graph,
    src: int,
    dst: int,
) -> Optional[int]:
    if src not in graph or dst not in graph:
        return None
    try:
        return int(nx.shortest_path_length(graph, src, dst))
    except Exception:
        return None


def reciprocal_rank(candidate_nodes: List[int], gt_node: int) -> float:
    for i, nid in enumerate(candidate_nodes, start=1):
        if int(nid) == int(gt_node):
            return 1.0 / float(i)
    return 0.0


def rank_of_gt(candidate_nodes: List[int], gt_node: int) -> Optional[int]:
    for i, nid in enumerate(candidate_nodes, start=1):
        if int(nid) == int(gt_node):
            return i
    return None


def summarize_by_group(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get(key, "UNK"))].append(r)

    out: Dict[str, Any] = {}
    for g, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        n = len(items)
        if n == 0:
            continue

        top1 = sum(int(x["top1_correct"]) for x in items) / n
        top3 = sum(int(x["top3_hit"]) for x in items) / n
        top5 = sum(int(x["top5_hit"]) for x in items) / n
        dist1 = sum(int(x["dist_le_1"]) for x in items) / n
        dist2 = sum(int(x["dist_le_2"]) for x in items) / n

        confs = [
            float(x["confidence"])
            for x in items
            if x.get("confidence") not in ("", None)
        ]
        mrrs = [float(x["mrr"]) for x in items]

        out[g] = {
            "n": n,
            "top1_acc": round(top1, 4),
            "top3_hit": round(top3, 4),
            "top5_hit": round(top5, 4),
            "mrr": round(sum(mrrs) / n, 4),
            "dist_le_1_acc": round(dist1, 4),
            "dist_le_2_acc": round(dist2, 4),
            "mean_confidence": round(mean(confs), 4) if confs else None,
        }
    return out


def ensure_no_geo_leakage(
    test_dir: Path,
    node_images_dir: Optional[Path],
    use_geo: bool,
    allow_overlap: bool,
) -> None:
    """
    use_geo=True일 때 테스트셋 폴더와 레퍼런스 이미지 폴더가 겹치면
    자기 자신과 매칭되어 성능이 부풀려질 수 있으므로 차단.
    """
    if not use_geo or node_images_dir is None:
        return

    test_dir = test_dir.resolve()
    node_images_dir = node_images_dir.resolve()

    overlap = (
        test_dir == node_images_dir
        or str(test_dir).startswith(str(node_images_dir))
        or str(node_images_dir).startswith(str(test_dir))
    )

    if overlap and not allow_overlap:
        raise ValueError(
            "test_dir 와 node_images_dir 가 겹칩니다. "
            "현재 use_geo는 node_images_dir 에서 node_id*.jpg/png/jpeg 를 레퍼런스로 모으므로, "
            "테스트 이미지를 같은 폴더에 두면 자기 자신과 매칭되어 정확도가 부풀려질 수 있습니다. "
            "--allow_geo_overlap 으로 강행할 수는 있지만 권장하지 않습니다."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="테스트 이미지 폴더")
    ap.add_argument("--csv", required=True, help="지도 CSV 경로")
    ap.add_argument(
        "--out_dir",
        default="benchmark_results/localization",
        help="결과 저장 폴더",
    )
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument(
        "--topk",
        type=int,
        default=5,
        help="localize_image 내부 top-k 후보 수",
    )

    # localize_image 옵션
    ap.add_argument("--use_ocr", action="store_true", help="OCR 재랭킹 사용")
    ap.add_argument("--use_geo", action="store_true", help="기하 검증 사용")
    ap.add_argument(
        "--node_images_dir",
        default=None,
        help="레퍼런스 노드 이미지 폴더 (기본: csv 부모/node_images/node_images)",
    )
    ap.add_argument(
        "--allow_geo_overlap",
        action="store_true",
        help="test_dir 와 node_images_dir 중복 허용",
    )

    # 가중치
    ap.add_argument("--w_clip", type=float, default=1.0)
    ap.add_argument("--w_ocr", type=float, default=0.3)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--clip_pool_size", type=int, default=50)
    ap.add_argument("--ocr_merge_min_score", type=float, default=0.4)
    ap.add_argument("--geo_candidate_limit", type=int, default=10)
    ap.add_argument("--geo_ref_limit", type=int, default=4)
    ap.add_argument(
        "--w_prior",
        type=float,
        default=0.0,
        help="독립 이미지셋 benchmark 는 보통 0 권장",
    )

    # OCR 세부 옵션
    ap.add_argument("--ocr_langs", default="ko,en", help="예: ko,en")
    ap.add_argument("--ocr_use_roi", action="store_true")
    ap.add_argument("--ocr_max_rois", type=int, default=8)

    ap.add_argument("--ocr_grayscale", action="store_true")
    ap.add_argument("--ocr_upscale", type=float, default=1.0)
    ap.add_argument("--ocr_contrast", action="store_true")
    ap.add_argument("--ocr_sharpen", action="store_true")
    ap.add_argument("--ocr_adaptive", action="store_true")
    ap.add_argument("--ocr_clahe_clip", type=float, default=2.0)
    ap.add_argument("--ocr_clahe_grid", type=int, default=8)
    ap.add_argument("--ocr_sharpen_amount", type=float, default=0.7)
    ap.add_argument("--ocr_adaptive_block", type=int, default=31)
    ap.add_argument("--ocr_adaptive_C", type=int, default=5)

    ap.add_argument("--ocr_text_threshold", type=float, default=None)
    ap.add_argument("--ocr_low_text", type=float, default=None)
    ap.add_argument("--ocr_link_threshold", type=float, default=None)
    ap.add_argument("--ocr_decoder", type=str, default=None)
    ap.add_argument("--ocr_beam_width", type=int, default=None)
    ap.add_argument("--ocr_debug_max_observations", type=int, default=200)

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서부터 N장만 평가",
    )

    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.node_images_dir is None:
        # run_cvm_step.py 와 동일한 관례
        node_images_dir = csv_path.resolve().parent / "node_images" / "node_images"
    else:
        node_images_dir = Path(args.node_images_dir)

    ensure_no_geo_leakage(
        test_dir=test_dir,
        node_images_dir=node_images_dir,
        use_geo=bool(args.use_geo),
        allow_overlap=bool(args.allow_geo_overlap),
    )

    graph, node_records, emb_matrix, node_ids = load_map_csv(str(csv_path))
    image_paths = collect_images(test_dir)

    print("[DEBUG] eval_localization started")
    print(f"[DEBUG] test_dir={test_dir}")
    print(f"[DEBUG] csv_path={csv_path}")
    print(f"[DEBUG] found_images={len(image_paths)}")

    if args.limit is not None:
        image_paths = image_paths[: int(args.limit)]

    if not image_paths:
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {test_dir}")

    logger.info(f"test images: {len(image_paths)}")
    logger.info(f"csv: {csv_path}")
    logger.info(f"use_ocr={args.use_ocr}, use_geo={args.use_geo}, node_images_dir={node_images_dir}")

    ocr_langs = [s.strip() for s in args.ocr_langs.split(",") if s.strip()]

    rows: List[Dict[str, Any]] = []
    confusion_counter: Counter[Tuple[int, int]] = Counter()

    for idx, image_path in enumerate(image_paths, start=1):
        gt_node, angle, variant = parse_gt_from_filename(image_path)

        try:
            out = localize_image(
                image_path=str(image_path),
                csv_path=str(csv_path),
                device=args.device,
                topk=max(int(args.topk), 5),
                clip_pool_size=int(args.clip_pool_size),
                ocr_merge_min_score=float(args.ocr_merge_min_score),
                use_ocr=bool(args.use_ocr),
                node_images_dir=str(node_images_dir),
                use_geo=bool(args.use_geo),
                geo_candidate_limit=int(args.geo_candidate_limit),
                geo_ref_limit=int(args.geo_ref_limit),
                prev_node=None,  # 독립 샘플 benchmark 는 None 권장
                w_clip=float(args.w_clip),
                w_ocr=float(args.w_ocr),
                w_geo=float(args.w_geo),
                w_prior=float(args.w_prior),
                ocr_langs=ocr_langs,
                ocr_use_roi=bool(args.ocr_use_roi),
                ocr_max_rois=int(args.ocr_max_rois),
                ocr_grayscale=bool(args.ocr_grayscale),
                ocr_upscale=float(args.ocr_upscale),
                ocr_contrast=bool(args.ocr_contrast),
                ocr_sharpen=bool(args.ocr_sharpen),
                ocr_adaptive=bool(args.ocr_adaptive),
                ocr_clahe_clip=float(args.ocr_clahe_clip),
                ocr_clahe_grid=int(args.ocr_clahe_grid),
                ocr_sharpen_amount=float(args.ocr_sharpen_amount),
                ocr_adaptive_block=int(args.ocr_adaptive_block),
                ocr_adaptive_C=int(args.ocr_adaptive_C),
                ocr_text_threshold=args.ocr_text_threshold,
                ocr_low_text=args.ocr_low_text,
                ocr_link_threshold=args.ocr_link_threshold,
                ocr_decoder=args.ocr_decoder,
                ocr_beam_width=args.ocr_beam_width,
                ocr_debug_max_observations=int(args.ocr_debug_max_observations),
            )

            pred_node = int(out["current_node"])
            confidence = float(out.get("confidence", 0.0))
            candidates = out.get("candidates", []) or []
            candidate_nodes = [int(c["node_id"]) for c in candidates]
            debug = out.get("debug", {}) or {}

            rr = reciprocal_rank(candidate_nodes, gt_node)
            gt_rank = rank_of_gt(candidate_nodes, gt_node)
            gdist = shortest_path_distance_or_none(graph, pred_node, gt_node)

            row: Dict[str, Any] = {
                "image": str(image_path),
                "filename": image_path.name,
                "gt_node": gt_node,
                "angle": angle,
                "variant": variant if variant is not None else "",
                "pred_node": pred_node,
                "confidence": round(confidence, 6),
                "top1_correct": int(pred_node == gt_node),
                "top3_hit": int(gt_rank is not None and gt_rank <= 3),
                "top5_hit": int(gt_rank is not None and gt_rank <= 5),
                "gt_rank": gt_rank if gt_rank is not None else "",
                "mrr": round(rr, 6),
                "graph_distance": gdist if gdist is not None else "",
                "dist_le_1": int(gdist is not None and gdist <= 1),
                "dist_le_2": int(gdist is not None and gdist <= 2),
                "ocr_numbers": ",".join(debug.get("ocr_numbers", []) or []),
                "ocr_raw_texts": json.dumps(debug.get("ocr_raw_texts", []) or [], ensure_ascii=False),
                "ocr_observation_count": debug.get("ocr_observation_count", ""),
                "ocr_source_summary": json.dumps(debug.get("ocr_source_summary", {}) or {}, ensure_ascii=False),
                "ocr_number_sources": json.dumps(debug.get("ocr_number_sources", []) or [], ensure_ascii=False),
                "ocr_observations": json.dumps(debug.get("ocr_observations", []) or [], ensure_ascii=False),
                "ocr_num_counts": json.dumps(debug.get("ocr_num_counts", {}) or {}, ensure_ascii=False),
                "ocr_num_weights": json.dumps(debug.get("ocr_num_weights", {}) or {}, ensure_ascii=False),
                "candidate_nodes": json.dumps(candidate_nodes, ensure_ascii=False),
                "initial_clip_pool_size": debug.get("initial_clip_pool_size", ""),
                "final_pool_size": debug.get("final_pool_size", ""),
                "ocr_pool_merge_added": debug.get("ocr_pool_merge_added", ""),
                "ocr_merged_node_ids": json.dumps(debug.get("ocr_merged_node_ids", []) or [], ensure_ascii=False),
                "geo_evaluated_count": debug.get("geo_evaluated_count", ""),
                "fusion_candidates": json.dumps(debug.get("fusion_candidates", []) or [], ensure_ascii=False),
                "error": "",
            }
            rows.append(row)

            if pred_node != gt_node:
                confusion_counter[(gt_node, pred_node)] += 1

            logger.info(
                f"[{idx}/{len(image_paths)}] {image_path.name} | "
                f"gt={gt_node} pred={pred_node} conf={confidence:.3f} "
                f"dist={gdist} rank={gt_rank}"
            )

        except Exception as e:
            logger.exception(f"Failed on {image_path.name}: {e}")
            rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "gt_node": gt_node,
                "angle": angle,
                "variant": variant if variant is not None else "",
                "pred_node": "",
                "confidence": "",
                "top1_correct": 0,
                "top3_hit": 0,
                "top5_hit": 0,
                "gt_rank": "",
                "mrr": 0.0,
                "graph_distance": "",
                "dist_le_1": 0,
                "dist_le_2": 0,
                "ocr_numbers": "",
                "ocr_raw_texts": "[]",
                "ocr_observation_count": "",
                "ocr_source_summary": "{}",
                "ocr_number_sources": "[]",
                "ocr_observations": "[]",
                "ocr_num_counts": "{}",
                "ocr_num_weights": "{}",
                "candidate_nodes": "[]",
                "initial_clip_pool_size": "",
                "final_pool_size": "",
                "ocr_pool_merge_added": "",
                "ocr_merged_node_ids": "[]",
                "geo_evaluated_count": "",
                "fusion_candidates": "[]",
                "error": str(e),
            })

    valid_rows = [r for r in rows if str(r.get("pred_node", "")).strip() != ""]
    n = len(valid_rows)

    if n == 0:
        raise RuntimeError("유효한 평가 결과가 없습니다.")

    top1_acc = sum(int(r["top1_correct"]) for r in valid_rows) / n
    top3_hit = sum(int(r["top3_hit"]) for r in valid_rows) / n
    top5_hit = sum(int(r["top5_hit"]) for r in valid_rows) / n
    dist1_acc = sum(int(r["dist_le_1"]) for r in valid_rows) / n
    dist2_acc = sum(int(r["dist_le_2"]) for r in valid_rows) / n
    mrr_score = sum(float(r["mrr"]) for r in valid_rows) / n

    valid_dists = [
        int(r["graph_distance"])
        for r in valid_rows
        if str(r["graph_distance"]) != ""
    ]
    mean_gdist = mean(valid_dists) if valid_dists else None

    correct_confs = [
        float(r["confidence"])
        for r in valid_rows
        if int(r["top1_correct"]) == 1 and r.get("confidence") not in ("", None)
    ]
    wrong_confs = [
        float(r["confidence"])
        for r in valid_rows
        if int(r["top1_correct"]) == 0 and r.get("confidence") not in ("", None)
    ]

    summary = {
        "n_total": len(rows),
        "n_valid": n,
        "top1_accuracy": round(top1_acc, 4),
        "top3_hit_rate": round(top3_hit, 4),
        "top5_hit_rate": round(top5_hit, 4),
        "mrr": round(mrr_score, 4),
        "graph_distance_mean": round(mean_gdist, 4) if mean_gdist is not None else None,
        "graph_distance_le_1_accuracy": round(dist1_acc, 4),
        "graph_distance_le_2_accuracy": round(dist2_acc, 4),
        "confidence_mean_correct": round(mean(correct_confs), 4) if correct_confs else None,
        "confidence_mean_wrong": round(mean(wrong_confs), 4) if wrong_confs else None,
        "per_angle": summarize_by_group(valid_rows, "angle"),
        "per_node": summarize_by_group(valid_rows, "gt_node"),
        "top_confusions": [
            {"gt_node": gt, "pred_node": pred, "count": cnt}
            for (gt, pred), cnt in confusion_counter.most_common(20)
        ],
        "settings": {
            "test_dir": str(test_dir),
            "csv": str(csv_path),
            "device": args.device,
            "topk": int(args.topk),
            "use_ocr": bool(args.use_ocr),
            "use_geo": bool(args.use_geo),
            "node_images_dir": str(node_images_dir),
            "w_clip": float(args.w_clip),
            "w_ocr": float(args.w_ocr),
            "w_geo": float(args.w_geo),
            "w_prior": float(args.w_prior),
            "clip_pool_size": int(args.clip_pool_size),
            "ocr_merge_min_score": float(args.ocr_merge_min_score),
            "geo_candidate_limit": int(args.geo_candidate_limit),
            "geo_ref_limit": int(args.geo_ref_limit),
            "ocr_langs": ocr_langs,
            "ocr_use_roi": bool(args.ocr_use_roi),
            "ocr_max_rois": int(args.ocr_max_rois),
            "ocr_grayscale": bool(args.ocr_grayscale),
            "ocr_upscale": float(args.ocr_upscale),
            "ocr_contrast": bool(args.ocr_contrast),
            "ocr_sharpen": bool(args.ocr_sharpen),
            "ocr_adaptive": bool(args.ocr_adaptive),
            "ocr_clahe_clip": float(args.ocr_clahe_clip),
            "ocr_clahe_grid": int(args.ocr_clahe_grid),
            "ocr_sharpen_amount": float(args.ocr_sharpen_amount),
            "ocr_adaptive_block": int(args.ocr_adaptive_block),
            "ocr_adaptive_C": int(args.ocr_adaptive_C),
            "ocr_text_threshold": args.ocr_text_threshold,
            "ocr_low_text": args.ocr_low_text,
            "ocr_link_threshold": args.ocr_link_threshold,
            "ocr_decoder": args.ocr_decoder,
            "ocr_beam_width": args.ocr_beam_width,
            "ocr_debug_max_observations": int(args.ocr_debug_max_observations),
        },
    }

    csv_out = out_dir / "per_image_results.csv"
    json_out = out_dir / "summary.json"

    fieldnames = [
        "image",
        "filename",
        "gt_node",
        "angle",
        "variant",
        "pred_node",
        "confidence",
        "top1_correct",
        "top3_hit",
        "top5_hit",
        "gt_rank",
        "mrr",
        "graph_distance",
        "dist_le_1",
        "dist_le_2",
        "ocr_numbers",
        "ocr_raw_texts",
        "ocr_observation_count",
        "ocr_source_summary",
        "ocr_number_sources",
        "ocr_observations",
        "ocr_num_counts",
        "ocr_num_weights",
        "candidate_nodes",
        "initial_clip_pool_size",
        "final_pool_size",
        "ocr_pool_merge_added",
        "ocr_merged_node_ids",
        "geo_evaluated_count",
        "fusion_candidates",
        "error",
    ]

    with csv_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Localization Benchmark Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[Saved] {csv_out}")
    print(f"[Saved] {json_out}")


if __name__ == "__main__":
    main()
