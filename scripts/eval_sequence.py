# scripts/eval_sequence.py
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from loguru import logger

try:
    from scripts.localize_image import localize_image  # type: ignore
    from scripts.map_loader import load_map_csv  # type: ignore
    from scripts.value_map import build_value_map_v2  # type: ignore
except ImportError:
    from localize_image import localize_image  # type: ignore
    from map_loader import load_map_csv  # type: ignore
    from value_map import build_value_map_v2  # type: ignore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 파일명 규칙:
#   start_end_gtcurrent_index.jpg
# 예:
#   401_4150_401_001.jpg
#   401_4150_4102_002.jpg
FILENAME_RE = re.compile(
    r"^(?P<start>\d+)_(?P<end>\d+)_(?P<gt>\d+)_(?P<index>\d+)$",
    re.IGNORECASE,
)


def parse_sequence_filename(path: Path) -> Tuple[int, int, int, int]:
    stem = path.stem
    m = FILENAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"파일명 규칙이 맞지 않습니다: {path.name} "
            f"(예: 401_4150_401_001.jpg)"
        )
    start_node = int(m.group("start"))
    end_node = int(m.group("end"))
    gt_current_node = int(m.group("gt"))
    frame_idx = int(m.group("index"))
    return start_node, end_node, gt_current_node, frame_idx


def collect_sequence_images(
    test_dir: Path,
    set_key: str,
) -> Tuple[int, int, List[Tuple[int, Path, int]]]:
    """
    지정한 세트(예: 401_4150)에 해당하는 이미지만 모아서
    (frame_idx, path, gt_current_node) 리스트 반환
    """
    try:
        set_start, set_end = [int(x) for x in set_key.split("_")]
    except Exception:
        raise ValueError(
            f"--set 형식이 잘못되었습니다: {set_key} (예: 401_4150)"
        )

    matched: List[Tuple[int, Path, int]] = []

    for p in test_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        start_node, end_node, gt_current_node, frame_idx = parse_sequence_filename(p)
        if start_node == set_start and end_node == set_end:
            matched.append((frame_idx, p, gt_current_node))

    matched.sort(key=lambda x: x[0])

    if not matched:
        raise FileNotFoundError(
            f"세트 {set_key} 에 해당하는 이미지가 없습니다. "
            f"test_dir={test_dir}"
        )

    return set_start, set_end, matched


def shortest_path_distance_or_none(
    graph: nx.Graph,
    src: Optional[int],
    dst: Optional[int],
) -> Optional[int]:
    if src is None or dst is None:
        return None
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


def ensure_no_geo_leakage(
    test_dir: Path,
    node_images_dir: Optional[Path],
    use_geo: bool,
    allow_overlap: bool,
) -> None:
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
            "use_geo=True 상태에서 테스트 이미지가 레퍼런스 폴더와 겹치면 "
            "자기 자신과 매칭되어 성능이 부풀려질 수 있습니다. "
            "--allow_geo_overlap 으로 강행할 수는 있지만 권장하지 않습니다."
        )


def choose_next_node(value_map: Dict[str, Any]) -> Optional[int]:
    neighbors = value_map.get("neighbor_values", {}) or {}
    if not neighbors:
        return None
    return int(max(neighbors.items(), key=lambda kv: kv[1])[0])


def get_shortest_path_next_candidates(
    graph: nx.Graph,
    gt_current_node: int,
    end_node: int,
) -> List[int]:
    """
    gt_current_node에서 end_node로 가는 shortest path 상의
    '올바른 다음 노드 후보들' 반환.
    shortest path가 여러 개면 모두 허용.
    """
    if gt_current_node not in graph or end_node not in graph:
        return []

    if gt_current_node == end_node:
        return []

    try:
        shortest_len = nx.shortest_path_length(graph, gt_current_node, end_node)
    except Exception:
        return []

    candidates: List[int] = []
    for nb in graph.neighbors(gt_current_node):
        try:
            d = nx.shortest_path_length(graph, nb, end_node)
            if d == shortest_len - 1:
                candidates.append(int(nb))
        except Exception:
            continue
    return sorted(set(candidates))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="시퀀스 이미지 폴더")
    ap.add_argument("--set", required=True, help="평가할 세트 키 (예: 401_4150)")
    ap.add_argument("--csv", required=True, help="지도 CSV 경로")
    ap.add_argument(
        "--out_dir",
        default="benchmark_results/sequence",
        help="결과 저장 폴더",
    )
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--topk", type=int, default=5)

    # 현재 시스템 기준 기본값: OCR on / GEO on / prior on(시퀀스에서 prev_node 전달)
    ap.add_argument("--disable_ocr", action="store_true", help="OCR 끄기")
    ap.add_argument("--disable_geo", action="store_true", help="Geometry 끄기")

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

    # 현재 시스템 기본 가중치
    ap.add_argument("--w_clip", type=float, default=1.0)
    ap.add_argument("--w_ocr", type=float, default=0.8)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--w_prior", type=float, default=0.2)
    ap.add_argument("--clip_pool_size", type=int, default=50)
    ap.add_argument("--ocr_merge_min_score", type=float, default=0.4)
    ap.add_argument("--geo_candidate_limit", type=int, default=10)
    ap.add_argument("--geo_ref_limit", type=int, default=4)

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

    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.node_images_dir is None:
        node_images_dir = csv_path.resolve().parent / "node_images" / "node_images"
    else:
        node_images_dir = Path(args.node_images_dir)

    use_ocr = not bool(args.disable_ocr)
    use_geo = not bool(args.disable_geo)

    ensure_no_geo_leakage(
        test_dir=test_dir,
        node_images_dir=node_images_dir,
        use_geo=use_geo,
        allow_overlap=bool(args.allow_geo_overlap),
    )

    graph, node_records, emb_matrix, node_ids = load_map_csv(str(csv_path))

    start_node, end_node, matched = collect_sequence_images(test_dir, args.set)

    if start_node not in graph:
        raise ValueError(f"시작 노드 {start_node} 가 graph 에 없습니다.")
    if end_node not in graph:
        raise ValueError(f"종료 노드 {end_node} 가 graph 에 없습니다.")

    logger.info(f"sequence set: {args.set}")
    logger.info(f"frames: {len(matched)}")
    logger.info(f"use_ocr={use_ocr}, use_geo={use_geo}, node_images_dir={node_images_dir}")

    ocr_langs = [s.strip() for s in args.ocr_langs.split(",") if s.strip()]

    prev_node: Optional[int] = None
    rows: List[Dict[str, Any]] = []

    graph_errors: List[int] = []
    confs_correct: List[float] = []
    confs_wrong: List[float] = []

    for order, (frame_idx, image_path, gt_current_node) in enumerate(matched, start=1):
        # 이 프레임 기준 route는 gt_current -> end_node shortest path 로 재계산
        try:
            route_nodes = [int(n) for n in nx.shortest_path(graph, gt_current_node, end_node)]
        except Exception as e:
            raise RuntimeError(
                f"그래프에서 {gt_current_node} -> {end_node} shortest path를 찾을 수 없습니다: {e}"
            )

        cvm_result = localize_image(
            image_path=str(image_path),
            csv_path=str(csv_path),
            device=args.device,
            topk=max(int(args.topk), 5),
            clip_pool_size=int(args.clip_pool_size),
            ocr_merge_min_score=float(args.ocr_merge_min_score),
            use_ocr=use_ocr,
            node_images_dir=str(node_images_dir),
            use_geo=use_geo,
            geo_candidate_limit=int(args.geo_candidate_limit),
            geo_ref_limit=int(args.geo_ref_limit),
            prev_node=prev_node,
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
        )

        pred_current_node = int(cvm_result["current_node"])
        confidence = float(cvm_result.get("confidence", 0.0))

        candidates = cvm_result.get("candidates", []) or []
        candidate_nodes = [int(c["node_id"]) for c in candidates]
        debug = cvm_result.get("debug", {}) or {}

        value_map = build_value_map_v2(
            current_node=pred_current_node,
            route_nodes=route_nodes,
            target_nodes=[end_node],
            graph=graph,
            cvm_candidates=cvm_result.get("candidates"),
            blocked_edges=set(),
        )
        pred_next_node = choose_next_node(value_map)

        gt_rank = rank_of_gt(candidate_nodes, gt_current_node)
        rr = reciprocal_rank(candidate_nodes, gt_current_node)

        graph_error = shortest_path_distance_or_none(graph, pred_current_node, gt_current_node)
        if graph_error is not None:
            graph_errors.append(graph_error)

        gt_next_candidates = get_shortest_path_next_candidates(graph, gt_current_node, end_node)
        next_step_correct = (
            pred_next_node is not None and pred_next_node in gt_next_candidates
        )

        dist_pred_to_goal = shortest_path_distance_or_none(graph, pred_current_node, end_node)
        dist_gt_to_goal = shortest_path_distance_or_none(graph, gt_current_node, end_node)

        top1_correct = int(pred_current_node == gt_current_node)
        top3_hit = int(gt_rank is not None and gt_rank <= 3)
        top5_hit = int(gt_rank is not None and gt_rank <= 5)

        if top1_correct:
            confs_correct.append(confidence)
        else:
            confs_wrong.append(confidence)

        row = {
            "frame_order": order,
            "frame_idx": frame_idx,
            "image": str(image_path),
            "filename": image_path.name,
            "set_start_node": start_node,
            "set_end_node": end_node,
            "gt_current_node": gt_current_node,
            "pred_current_node": pred_current_node,
            "pred_next_node": pred_next_node if pred_next_node is not None else "",
            "top1_correct": top1_correct,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "gt_rank": gt_rank if gt_rank is not None else "",
            "mrr": round(rr, 6),
            "graph_error_to_gt": graph_error if graph_error is not None else "",
            "gt_next_candidates": json.dumps(gt_next_candidates, ensure_ascii=False),
            "next_step_correct": int(next_step_correct),
            "distance_pred_to_goal": dist_pred_to_goal if dist_pred_to_goal is not None else "",
            "distance_gt_to_goal": dist_gt_to_goal if dist_gt_to_goal is not None else "",
            "confidence": round(confidence, 6),
            "ocr_numbers": ",".join(debug.get("ocr_numbers", []) or []),
            "ocr_raw_texts": json.dumps(debug.get("ocr_raw_texts", []) or [], ensure_ascii=False),
            "ocr_num_counts": json.dumps(debug.get("ocr_num_counts", {}) or {}, ensure_ascii=False),
            "ocr_num_weights": json.dumps(debug.get("ocr_num_weights", {}) or {}, ensure_ascii=False),
            "candidate_nodes": json.dumps(candidate_nodes, ensure_ascii=False),
            "initial_clip_pool_size": debug.get("initial_clip_pool_size", ""),
            "final_pool_size": debug.get("final_pool_size", ""),
            "ocr_pool_merge_added": debug.get("ocr_pool_merge_added", ""),
            "ocr_merged_node_ids": json.dumps(debug.get("ocr_merged_node_ids", []) or [], ensure_ascii=False),
            "geo_evaluated_count": debug.get("geo_evaluated_count", ""),
            "fusion_candidates": json.dumps(debug.get("fusion_candidates", []) or [], ensure_ascii=False),
        }
        rows.append(row)

        logger.info(
            f"[{order}/{len(matched)}] {image_path.name} | "
            f"gt_cur={gt_current_node} pred_cur={pred_current_node} "
            f"pred_next={pred_next_node} graph_err={graph_error} rank={gt_rank}"
        )

        # 실제 시스템처럼 이전 예측 노드를 다음 프레임 prior로 사용
        prev_node = pred_current_node

    n = len(rows)
    top1_acc = sum(int(r["top1_correct"]) for r in rows) / n
    top3_hit_rate = sum(int(r["top3_hit"]) for r in rows) / n
    top5_hit_rate = sum(int(r["top5_hit"]) for r in rows) / n
    next_step_acc = sum(int(r["next_step_correct"]) for r in rows) / n
    mrr_score = sum(float(r["mrr"]) for r in rows) / n

    summary = {
        "set": args.set,
        "n_frames": n,
        "start_node": start_node,
        "end_node": end_node,
        "frame_top1_accuracy": round(top1_acc, 4),
        "frame_top3_hit_rate": round(top3_hit_rate, 4),
        "frame_top5_hit_rate": round(top5_hit_rate, 4),
        "frame_mrr": round(mrr_score, 4),
        "mean_graph_error_to_gt": round(mean(graph_errors), 4) if graph_errors else None,
        "next_step_accuracy": round(next_step_acc, 4),
        "start_frame_exact": int(int(rows[0]["pred_current_node"]) == int(rows[0]["gt_current_node"])),
        "final_frame_exact": int(int(rows[-1]["pred_current_node"]) == int(rows[-1]["gt_current_node"])),
        "final_goal_hit": int(int(rows[-1]["pred_current_node"]) == end_node),
        "best_goal_hit_any_frame": int(any(int(r["pred_current_node"]) == end_node for r in rows)),
        "confidence_mean_correct": round(mean(confs_correct), 4) if confs_correct else None,
        "confidence_mean_wrong": round(mean(confs_wrong), 4) if confs_wrong else None,
        "settings": {
            "test_dir": str(test_dir),
            "csv": str(csv_path),
            "device": args.device,
            "topk": int(args.topk),
            "use_ocr": use_ocr,
            "use_geo": use_geo,
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
        },
    }

    csv_out = out_dir / f"{args.set}_per_frame.csv"
    json_out = out_dir / f"{args.set}_summary.json"

    fieldnames = [
        "frame_order",
        "frame_idx",
        "image",
        "filename",
        "set_start_node",
        "set_end_node",
        "gt_current_node",
        "pred_current_node",
        "pred_next_node",
        "top1_correct",
        "top3_hit",
        "top5_hit",
        "gt_rank",
        "mrr",
        "graph_error_to_gt",
        "gt_next_candidates",
        "next_step_correct",
        "distance_pred_to_goal",
        "distance_gt_to_goal",
        "confidence",
        "ocr_numbers",
        "ocr_raw_texts",
        "ocr_num_counts",
        "ocr_num_weights",
        "candidate_nodes",
        "initial_clip_pool_size",
        "final_pool_size",
        "ocr_pool_merge_added",
        "ocr_merged_node_ids",
        "geo_evaluated_count",
        "fusion_candidates",
    ]

    with csv_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Sequence Benchmark Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[Saved] {csv_out}")
    print(f"[Saved] {json_out}")


if __name__ == "__main__":
    main()
