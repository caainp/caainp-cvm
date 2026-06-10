from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from scripts.localize_image import encode_image_openclip, load_openclip  # type: ignore
except ImportError:
    from localize_image import encode_image_openclip, load_openclip  # type: ignore


REQUIRED_VIEW_COLUMNS = [
    "view_id",
    "node_id",
    "view_label",
    "image_path",
    "use_for_localization",
]


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_views(views_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(views_csv, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_VIEW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"views.csv missing required columns: {missing}")
    return df


def _validate_views(df: pd.DataFrame, root_dir: Path) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []

    duplicated_view_ids = df[df["view_id"].duplicated(keep=False)]
    for _, row in duplicated_view_ids.iterrows():
        failures.append({
            "category": "duplicate_view_id",
            "view_id": str(row.get("view_id", "")),
            "image_path": str(row.get("image_path", "")),
            "message": "view_id is duplicated",
        })

    duplicated_image_paths = df[df["image_path"].duplicated(keep=False)]
    for _, row in duplicated_image_paths.iterrows():
        failures.append({
            "category": "duplicate_image_path",
            "view_id": str(row.get("view_id", "")),
            "image_path": str(row.get("image_path", "")),
            "message": "image_path is duplicated",
        })

    for _, row in df.iterrows():
        image_path = root_dir / str(row["image_path"])
        if not image_path.exists():
            failures.append({
                "category": "missing_image_path",
                "view_id": str(row.get("view_id", "")),
                "image_path": str(row.get("image_path", "")),
                "message": "image_path does not exist",
            })

    return failures


def _write_report(
    report_path: Path,
    *,
    views_csv: Path,
    embeddings_path: Path,
    index_path: Path,
    model_name: str,
    pretrained: str,
    device: str,
    aug_times: int,
    views_count: int,
    image_load_success_count: int,
    embedding_rows: int,
    embedding_dim: int,
    has_nan: bool,
    has_inf: bool,
    true_count: int,
    false_count: int,
    false_included_count: int,
    duplicate_view_id_count: int,
    duplicate_image_path_count: int,
    failures: List[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# COEX View Embedding Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- views_csv: {views_csv.as_posix()}")
    lines.append(f"- embeddings: {embeddings_path.as_posix()}")
    lines.append(f"- index: {index_path.as_posix()}")
    lines.append(f"- model: {model_name}")
    lines.append(f"- pretrained: {pretrained}")
    lines.append(f"- device: {device}")
    lines.append(f"- aug_times: {aug_times}")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append(f"- views.csv row count: {views_count}")
    lines.append(f"- image load success count: {image_load_success_count}")
    lines.append(f"- embedding row count: {embedding_rows}")
    lines.append(f"- embedding dimension: {embedding_dim}")
    lines.append(f"- has NaN: {has_nan}")
    lines.append(f"- has inf: {has_inf}")
    lines.append(f"- use_for_localization=true count: {true_count}")
    lines.append(f"- use_for_localization=false count: {false_count}")
    lines.append(f"- false candidates included in embeddings: {false_included_count}")
    lines.append(f"- duplicate view_id count: {duplicate_view_id_count}")
    lines.append(f"- duplicate image_path count: {duplicate_image_path_count}")
    lines.append("")
    lines.append("## Failed Images")
    lines.append("")
    if failures:
        for failure in failures:
            lines.append(
                f"- [{failure['category']}] {failure['view_id']} | "
                f"{failure['image_path']} | {failure['message']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_embeddings(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir).resolve()
    views_csv = Path(args.views_csv)
    if not views_csv.is_absolute():
        views_csv = root_dir / views_csv
    views_csv = views_csv.resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = root_dir / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = (out_dir / args.embeddings_name).resolve()
    index_path = (out_dir / args.index_name).resolve()
    report_path = (report_dir / args.report_name).resolve()
    failure_csv_path = (report_dir / args.failure_name).resolve()

    df = _load_views(views_csv)
    failures = _validate_views(df, root_dir)
    duplicate_view_id_count = int(df["view_id"].duplicated().sum())
    duplicate_image_path_count = int(df["image_path"].duplicated().sum())
    true_count = int(df["use_for_localization"].map(_boolish).sum())
    false_count = int(len(df) - true_count)

    if failures:
        with failure_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "view_id", "image_path", "message"])
            writer.writeheader()
            writer.writerows(failures)
        _write_report(
            report_path,
            views_csv=views_csv,
            embeddings_path=embeddings_path,
            index_path=index_path,
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            aug_times=int(args.aug_times),
            views_count=len(df),
            image_load_success_count=0,
            embedding_rows=0,
            embedding_dim=0,
            has_nan=False,
            has_inf=False,
            true_count=true_count,
            false_count=false_count,
            false_included_count=0,
            duplicate_view_id_count=duplicate_view_id_count,
            duplicate_image_path_count=duplicate_image_path_count,
            failures=failures,
        )
        raise RuntimeError(f"View validation failed. See {report_path}")

    device = _resolve_device(str(args.device))
    model, preprocess = load_openclip(
        model_name=str(args.model),
        pretrained=str(args.pretrained),
        device=device,
    )

    embeddings: List[np.ndarray] = []
    index_rows: List[Dict[str, Any]] = []
    encode_failures: List[Dict[str, str]] = []

    for row_index, row in df.reset_index(drop=True).iterrows():
        image_rel = str(row["image_path"])
        image_path = (root_dir / image_rel).resolve()
        try:
            emb = encode_image_openclip(
                model,
                preprocess,
                str(image_path),
                device=device,
                aug_times=int(args.aug_times),
            )
        except Exception as exc:
            encode_failures.append({
                "category": "image_encode_failed",
                "view_id": str(row["view_id"]),
                "image_path": image_rel,
                "message": repr(exc),
            })
            continue

        embeddings.append(emb.astype(np.float32))
        index_rows.append({
            "row_index": int(row_index),
            "view_id": str(row["view_id"]),
            "node_id": int(row["node_id"]),
            "view_label": str(row["view_label"]),
            "view_role": "" if pd.isna(row.get("view_role", "")) else str(row.get("view_role", "")),
            "direction_to": "" if pd.isna(row.get("direction_to", "")) else str(row.get("direction_to", "")),
            "image_path": image_rel,
            "use_for_localization": str(row["use_for_localization"]).strip().lower(),
        })

    if encode_failures:
        with failure_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "view_id", "image_path", "message"])
            writer.writeheader()
            writer.writerows(encode_failures)
        _write_report(
            report_path,
            views_csv=views_csv,
            embeddings_path=embeddings_path,
            index_path=index_path,
            model_name=args.model,
            pretrained=args.pretrained,
            device=device,
            aug_times=int(args.aug_times),
            views_count=len(df),
            image_load_success_count=len(embeddings),
            embedding_rows=len(embeddings),
            embedding_dim=int(embeddings[0].shape[0]) if embeddings else 0,
            has_nan=False,
            has_inf=False,
            true_count=true_count,
            false_count=false_count,
            false_included_count=sum(1 for r in index_rows if r["use_for_localization"] == "false"),
            duplicate_view_id_count=duplicate_view_id_count,
            duplicate_image_path_count=duplicate_image_path_count,
            failures=encode_failures,
        )
        raise RuntimeError(f"Image encoding failed. See {report_path}")

    emb_matrix = np.vstack(embeddings).astype(np.float32)
    has_nan = bool(np.isnan(emb_matrix).any())
    has_inf = bool(np.isinf(emb_matrix).any())
    if has_nan or has_inf:
        encode_failures.append({
            "category": "embedding_invalid_value",
            "view_id": "",
            "image_path": "",
            "message": f"has_nan={has_nan}, has_inf={has_inf}",
        })
        raise RuntimeError("Embedding matrix contains NaN or inf")

    if emb_matrix.shape[0] != len(df):
        raise RuntimeError(f"Embedding row mismatch: {emb_matrix.shape[0]} != {len(df)}")
    if [int(r["row_index"]) for r in index_rows] != list(range(len(index_rows))):
        raise RuntimeError("row_index is not aligned with embedding row order")

    np.save(embeddings_path, emb_matrix)
    with index_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "view_id",
                "node_id",
                "view_label",
                "view_role",
                "direction_to",
                "image_path",
                "use_for_localization",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    if failure_csv_path.exists():
        failure_csv_path.unlink()

    _write_report(
        report_path,
        views_csv=views_csv,
        embeddings_path=embeddings_path,
        index_path=index_path,
        model_name=args.model,
        pretrained=args.pretrained,
        device=device,
        aug_times=int(args.aug_times),
        views_count=len(df),
        image_load_success_count=len(embeddings),
        embedding_rows=int(emb_matrix.shape[0]),
        embedding_dim=int(emb_matrix.shape[1]),
        has_nan=has_nan,
        has_inf=has_inf,
        true_count=true_count,
        false_count=false_count,
        false_included_count=sum(1 for r in index_rows if r["use_for_localization"] == "false"),
        duplicate_view_id_count=duplicate_view_id_count,
        duplicate_image_path_count=duplicate_image_path_count,
        failures=[],
    )

    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved index: {index_path}")
    print(f"Saved report: {report_path}")
    print(f"shape={emb_matrix.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--views_csv", default="data/coex/localization/views.csv")
    parser.add_argument("--out_dir", default="data/coex/localization")
    parser.add_argument("--report_dir", default="data/coex/reports")
    parser.add_argument("--embeddings_name", default="view_embeddings.npy")
    parser.add_argument("--index_name", default="view_embedding_index.csv")
    parser.add_argument("--report_name", default="view_embedding_report.md")
    parser.add_argument("--failure_name", default="view_embedding_failures.csv")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aug_times", type=int, default=1)
    args = parser.parse_args()
    if int(args.aug_times) < 1:
        raise ValueError("--aug_times must be >= 1")
    build_embeddings(args)


if __name__ == "__main__":
    main()
