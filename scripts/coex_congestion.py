"""
COEX 3F corridor congestion CLI.

Person detection (YOLO) + walkable ROI occupancy → congestion level.
Optionally writes viz images and blocked-edge hints for Value Map v2.

Examples:
  python -m scripts.coex_congestion --image view/3F/3014/3F_3014_junction_axis_01.jpg --save-viz
  python -m scripts.coex_congestion --batch-passage --root_dir . --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from caainp_cvm.congestion import (
    CongestionLevel,
    PersonDetector,
    analyze_congestion,
    is_passage_view,
    suggest_blocked_edges,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def _load_views(views_csv: Path) -> pd.DataFrame:
    return pd.read_csv(views_csv, encoding="utf-8-sig")


def _lookup_view(df: pd.DataFrame, image_path: Path, root: Path) -> Dict[str, Any]:
    rel = image_path.resolve().relative_to(root).as_posix()
    alt = rel.replace("/", "\\")
    row = df[df["image_path"].isin([rel, alt])]
    if row.empty:
        row = df[df["image_path"].str.replace("\\", "/", regex=False) == rel]
    if row.empty:
        return {}
    r = row.iloc[0]
    direction = r.get("direction_to", "")
    direction_to = None
    if pd.notna(direction) and str(direction).strip():
        direction_to = str(direction).strip()
    return {
        "view_id": str(r.get("view_id", "")),
        "node_id": int(r["node_id"]) if pd.notna(r.get("node_id")) else None,
        "view_role": str(r.get("view_role", "")),
        "direction_to": direction_to,
    }


def _run_single(args: argparse.Namespace) -> Dict[str, Any]:
    root = _resolve(Path(args.root_dir), ".")
    image = _resolve(root, args.image)
    views_df = _load_views(_resolve(root, args.views_csv)) if args.views_csv else None
    meta: Dict[str, Any] = {}
    if views_df is not None:
        meta = _lookup_view(views_df, image, root)

    viz_path = None
    if args.save_viz:
        out_dir = _resolve(root, args.viz_dir)
        viz_path = out_dir / f"{image.stem}_congestion.jpg"

    detector = PersonDetector(
        model_name=args.model,
        device=args.device,
    )
    result = analyze_congestion(
        image,
        detector=detector,
        node_id=meta.get("node_id"),
        view_id=meta.get("view_id"),
        view_role=meta.get("view_role"),
        direction_to=meta.get("direction_to"),
        save_viz_path=viz_path,
    )

    payload = result.to_dict()
    blocked: List[List[int]] = []
    if meta.get("node_id") is not None:
        blocked = [
            list(e)
            for e in suggest_blocked_edges(
                int(meta["node_id"]),
                result,
                block_from_level=CongestionLevel[args.block_from_level],
            )
        ]
    payload["suggested_blocked_edges"] = blocked
    payload["viz_path"] = str(viz_path) if viz_path else None
    return payload


def _run_batch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = _resolve(Path(args.root_dir), ".")
    views_df = _load_views(_resolve(root, args.views_csv))
    if args.passage_only:
        views_df = views_df[views_df["view_role"].map(is_passage_view)]

    rows = views_df
    if args.limit and args.limit > 0:
        rows = rows.head(int(args.limit))

    detector = PersonDetector(model_name=args.model, device=args.device)
    out_dir = _resolve(root, args.viz_dir) if args.save_viz else None
    results: List[Dict[str, Any]] = []

    for _, row in rows.iterrows():
        image = _resolve(root, str(row["image_path"]))
        viz_path = (out_dir / f"{row['view_id']}_congestion.jpg") if out_dir else None
        result = analyze_congestion(
            image,
            detector=detector,
            node_id=int(row["node_id"]),
            view_id=str(row["view_id"]),
            view_role=str(row.get("view_role", "")),
            direction_to=str(row.get("direction_to", "")).strip() or None,
            save_viz_path=viz_path,
        )
        item = result.to_dict()
        item["suggested_blocked_edges"] = [
            list(e)
            for e in suggest_blocked_edges(
                int(row["node_id"]),
                result,
                block_from_level=CongestionLevel[args.block_from_level],
            )
        ]
        results.append(item)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="COEX corridor congestion detection")
    parser.add_argument("--root_dir", default=".", help="Repository root")
    parser.add_argument("--image", help="Single image path (relative to root_dir)")
    parser.add_argument(
        "--views_csv",
        default="data/coex/localization/views.csv",
        help="views.csv for node/view metadata",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO weights")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--save-viz", action="store_true", help="Write annotated images")
    parser.add_argument(
        "--viz_dir",
        default="data/coex/reports/congestion_viz",
        help="Output folder for visualizations",
    )
    parser.add_argument(
        "--batch-passage",
        action="store_true",
        help="Run on all passage-like views (corridor/connector/context/foyer)",
    )
    parser.add_argument("--passage-only", action="store_true", help="Alias for batch filter")
    parser.add_argument("--limit", type=int, default=0, help="Max images in batch mode (0=all)")
    parser.add_argument(
        "--block-from-level",
        default="HIGH",
        choices=[m.name for m in CongestionLevel],
        help="Suggest blocked_edges from this level upward",
    )
    parser.add_argument(
        "--output",
        help="Write JSON report to this path",
    )
    args = parser.parse_args()

    if args.batch_passage or args.passage_only:
        if not args.batch_passage:
            args.batch_passage = True
        payload: Any = _run_batch(args)
    elif args.image:
        payload = _run_single(args)
    else:
        parser.error("Provide --image or --batch-passage")

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        out = _resolve(Path(args.root_dir), args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
