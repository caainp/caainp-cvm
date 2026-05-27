from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


REQUIRED_COLUMNS = ["node_id", "floor", "description", "neighbors", "type"]


def _resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _project_relative(root_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(root_dir).as_posix()


def _clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_neighbors(value: Any) -> List[int]:
    text = _clean_cell(value)
    if not text:
        return []
    out: List[int] = []
    for part in re.split(r"[^0-9]+", text):
        if part:
            out.append(int(part))
    return out


def _write_report(
    report_md: Path,
    report_csv: Path,
    *,
    root_dir: Path,
    source_path: Path,
    output_path: Path,
    row_count: int,
    node_ids: Set[int],
    missing_neighbor_rows: List[Dict[str, str]],
    duplicate_node_rows: List[Dict[str, str]],
    asymmetric_edges: List[Dict[str, str]],
) -> None:
    all_issues = missing_neighbor_rows + duplicate_node_rows + asymmetric_edges
    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "category", "node_id", "value", "message"])
        writer.writeheader()
        writer.writerows(all_issues)

    lines: List[str] = []
    lines.append("# COEX Graph CSV Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- source: {source_path.name} (read-only)")
    lines.append(f"- output: {_project_relative(root_dir, output_path)}")
    lines.append(f"- rows: {row_count}")
    lines.append(f"- unique node_ids: {len(node_ids)}")
    lines.append(f"- validation rows: {len(all_issues)}")
    lines.append("")
    lines.append("## Required Checks")
    lines.append("")
    lines.append(f"- {'PASS' if not duplicate_node_rows else 'FAIL'}: duplicate node_id")
    lines.append(f"- {'PASS' if not missing_neighbor_rows else 'FAIL'}: neighbors exist in node_id set")
    lines.append(f"- {'PASS' if not asymmetric_edges else 'WARN'}: neighbor links are symmetric")
    lines.append("")
    lines.append("## Issues")
    lines.append("")
    if all_issues:
        for issue in all_issues:
            lines.append(
                f"- [{issue['severity']}] {issue['category']} node={issue['node_id']} "
                f"value={issue['value']} - {issue['message']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This script does not modify the source xlsx nodemap.")
    lines.append("- The output CSV is a derived graph/value-map source, not a view-expanded map.")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_graph_csv(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir).resolve()
    source_path = _resolve_path(root_dir, args.source)
    out_dir = _resolve_path(root_dir, args.out_dir)
    report_dir = _resolve_path(root_dir, args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / args.output_name
    report_md = report_dir / args.report_md_name
    report_csv = report_dir / args.report_csv_name

    df = pd.read_excel(source_path, engine="openpyxl")
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in nodemap: {missing_columns}")

    df_out = df.copy()
    for col in df_out.columns:
        df_out[col] = df_out[col].map(_clean_cell)

    node_id_series = df_out["node_id"].astype(str)
    duplicate_node_rows: List[Dict[str, str]] = []
    for node_id in sorted(set(node_id_series[node_id_series.duplicated(keep=False)])):
        duplicate_node_rows.append({
            "severity": "error",
            "category": "duplicate_node_id",
            "node_id": node_id,
            "value": node_id,
            "message": "node_id appears more than once",
        })

    node_ids = {int(v) for v in node_id_series if str(v).strip()}
    neighbor_map: Dict[int, Set[int]] = {}
    missing_neighbor_rows: List[Dict[str, str]] = []
    for _, row in df_out.iterrows():
        node_id = int(row["node_id"])
        neighbors = _parse_neighbors(row["neighbors"])
        neighbor_map[node_id] = set(neighbors)
        for nb in neighbors:
            if nb not in node_ids:
                missing_neighbor_rows.append({
                    "severity": "error",
                    "category": "missing_neighbor_node",
                    "node_id": str(node_id),
                    "value": str(nb),
                    "message": "neighbor does not exist in node_id set",
                })

    asymmetric_edges: List[Dict[str, str]] = []
    for node_id, neighbors in sorted(neighbor_map.items()):
        for nb in sorted(neighbors):
            if nb in neighbor_map and node_id not in neighbor_map[nb]:
                asymmetric_edges.append({
                    "severity": "warning",
                    "category": "asymmetric_neighbor_link",
                    "node_id": str(node_id),
                    "value": str(nb),
                    "message": "neighbor relation is not listed in both directions",
                })

    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    _write_report(
        report_md,
        report_csv,
        root_dir=root_dir,
        source_path=source_path,
        output_path=output_path,
        row_count=len(df_out),
        node_ids=node_ids,
        missing_neighbor_rows=missing_neighbor_rows,
        duplicate_node_rows=duplicate_node_rows,
        asymmetric_edges=asymmetric_edges,
    )

    print(f"Saved graph CSV: {output_path}")
    print(f"Saved report MD: {report_md}")
    print(f"Saved report CSV: {report_csv}")
    print(f"rows={len(df_out)} issues={len(missing_neighbor_rows) + len(duplicate_node_rows) + len(asymmetric_edges)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--source", default="data/coex/source/coex_nodemap_m11.xlsx")
    parser.add_argument("--out_dir", default="data/coex/graph")
    parser.add_argument("--report_dir", default="data/coex/reports")
    parser.add_argument("--output_name", default="coex_nodemap.csv")
    parser.add_argument("--report_md_name", default="coex_nodemap_report.md")
    parser.add_argument("--report_csv_name", default="coex_nodemap_report.csv")
    build_graph_csv(parser.parse_args())


if __name__ == "__main__":
    main()
