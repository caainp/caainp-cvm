from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


VIEW_COLUMNS = [
    "view_id",
    "node_id",
    "floor",
    "view_label",
    "image_path",
    "view_role",
    "direction_to",
    "use_for_localization",
    "memo",
]

REPORT_COLUMNS = [
    "severity",
    "category",
    "view_id",
    "node_id",
    "folder_node",
    "image_path",
    "message",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FILENAME_RE = re.compile(r"^(?P<floor>[^_]+)_(?P<node_id>\d+)_(?P<view_label>.+)_(?P<index>\d+)$")
DIRECTION_RE = re.compile(r"(?:^|_)(?:to|toward|back_to)(?P<target>\d+)", re.IGNORECASE)


def _resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _image_path_for_csv(root_dir: Path, image_path: Path) -> str:
    resolved = image_path.resolve()
    try:
        return resolved.relative_to(root_dir).as_posix()
    except ValueError:
        return "../" + resolved.relative_to(root_dir.parent).as_posix()


def _label_tokens(view_label: str) -> List[str]:
    return [t for t in re.split(r"[_\-\s]+", view_label.lower()) if t]


def _has_token(view_label: str, token: str) -> bool:
    return token.lower() in _label_tokens(view_label)


def _has_door_like(view_label: str) -> bool:
    return any(t in {"door", "doors"} for t in _label_tokens(view_label))


def _infer_direction(view_label: str) -> str:
    match = DIRECTION_RE.search(view_label)
    return match.group("target") if match else ""


def _infer_role(view_label: str) -> str:
    label = view_label.lower()
    if _has_token(view_label, "dest"):
        return "destination"
    if _has_token(view_label, "ent") or _has_door_like(view_label):
        return "door"
    if "conn" in label:
        return "connector"
    if "foyer" in label:
        return "foyer"
    if "context" in label:
        return "context"
    if "axis" in label:
        return "corridor"
    if label.startswith(("to", "toward", "back_to", "from")):
        return "corridor"
    return ""


def _localization_policy(view_label: str) -> Tuple[bool, str]:
    label = view_label.lower()
    memos: List[str] = []
    use = True
    if _has_token(view_label, "dest"):
        use = False
        memos.append("destination confirmation candidate")
    if _has_token(view_label, "ent") or _has_door_like(view_label):
        use = False
        memos.append("room entrance candidate")
    if label.startswith(("from", "axis")) or "context" in label:
        memos.append(f"view_label={view_label}")
    return use, "; ".join(memos)


def _load_node_ids(nodemap_path: Path) -> Tuple[set[str], Dict[str, str]]:
    df = pd.read_excel(nodemap_path, engine="openpyxl")
    node_ids = {str(int(v)) for v in df["node_id"].dropna()}
    node_names: Dict[str, str] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("node_id")):
            continue
        node_id = str(int(row["node_id"]))
        node_names[node_id] = "" if pd.isna(row.get("name_ko")) else str(row.get("name_ko"))
    return node_ids, node_names


def build_views(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir).resolve()
    view_root = _resolve_path(root_dir, args.view_root)
    nodemap_path = _resolve_path(root_dir, args.nodemap)
    out_dir = _resolve_path(root_dir, args.out_dir)
    report_dir = _resolve_path(root_dir, args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    views_csv = out_dir / args.views_name
    report_csv = report_dir / args.report_csv_name
    report_md = report_dir / args.report_md_name

    node_ids, node_names = _load_node_ids(nodemap_path)
    image_paths = sorted(
        p for p in view_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    rows: List[Dict[str, str]] = []
    issues: List[Dict[str, str]] = []
    seen_view_ids: Dict[str, str] = {}

    for path in image_paths:
        image_path = _image_path_for_csv(root_dir, path)
        folder_node = path.parent.name
        view_id = path.stem
        match = FILENAME_RE.match(view_id)

        if view_id in seen_view_ids:
            issues.append({
                "severity": "error",
                "category": "duplicate_view_id",
                "view_id": view_id,
                "node_id": "",
                "folder_node": folder_node,
                "image_path": image_path,
                "message": f"Duplicate view_id. Previous path: {seen_view_ids[view_id]}",
            })
        else:
            seen_view_ids[view_id] = image_path

        if not path.exists():
            issues.append({
                "severity": "error",
                "category": "missing_image_path",
                "view_id": view_id,
                "node_id": "",
                "folder_node": folder_node,
                "image_path": image_path,
                "message": "Image path does not exist",
            })

        if not match:
            issues.append({
                "severity": "error",
                "category": "filename_format_error",
                "view_id": view_id,
                "node_id": "",
                "folder_node": folder_node,
                "image_path": image_path,
                "message": "Expected filename pattern: 3F_<node_id>_<view_label>_<index>",
            })
            continue

        floor = match.group("floor")
        node_id = match.group("node_id")
        view_label = match.group("view_label")

        if node_id not in node_ids:
            issues.append({
                "severity": "error",
                "category": "node_id_missing_in_nodemap",
                "view_id": view_id,
                "node_id": node_id,
                "folder_node": folder_node,
                "image_path": image_path,
                "message": "Filename node_id does not exist in nodemap",
            })

        if folder_node != node_id:
            issues.append({
                "severity": "warning",
                "category": "parent_folder_node_mismatch",
                "view_id": view_id,
                "node_id": node_id,
                "folder_node": folder_node,
                "image_path": image_path,
                "message": "Parent folder must match filename node_id",
            })

        use_for_localization, memo = _localization_policy(view_label)
        if not use_for_localization:
            issues.append({
                "severity": "review",
                "category": "use_for_localization_false_candidate",
                "view_id": view_id,
                "node_id": node_id,
                "folder_node": folder_node,
                "image_path": image_path,
                "message": memo or "Review localization use",
            })

        rows.append({
            "view_id": view_id,
            "node_id": node_id,
            "floor": floor,
            "view_label": view_label,
            "image_path": image_path,
            "view_role": _infer_role(view_label),
            "direction_to": _infer_direction(view_label),
            "use_for_localization": "true" if use_for_localization else "false",
            "memo": memo,
        })

    rows.sort(key=lambda r: (int(r["node_id"]), r["view_id"]))
    issues.sort(key=lambda r: (r["severity"], r["category"], r["image_path"]))

    with views_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=VIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(issues)

    category_counts = Counter(i["category"] for i in issues)
    severity_counts = Counter(i["severity"] for i in issues)
    node_counts = Counter(r["node_id"] for r in rows)

    lines: List[str] = []
    lines.append("# COEX View Validation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- nodemap: {nodemap_path.name} (read-only validation source)")
    lines.append(f"- nodemap nodes: {len(node_ids)}")
    lines.append(f"- scanned images: {len(image_paths)}")
    lines.append(f"- views.csv rows: {len(rows)}")
    lines.append(f"- validation rows: {len(issues)}")
    lines.append(f"- use_for_localization=false candidates: {sum(1 for r in rows if r['use_for_localization'] == 'false')}")
    lines.append("")
    lines.append("## Validation Counts")
    lines.append("")
    if category_counts:
        for category, count in sorted(category_counts.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- no validation issues")
    lines.append("")
    lines.append("## Severity Counts")
    lines.append("")
    if severity_counts:
        for severity, count in sorted(severity_counts.items()):
            lines.append(f"- {severity}: {count}")
    else:
        lines.append("- no validation issues")
    lines.append("")
    lines.append("## Required Checks")
    lines.append("")
    required = [
        ("image path existence", not any(i["category"] == "missing_image_path" for i in issues)),
        ("filename format", not any(i["category"] == "filename_format_error" for i in issues)),
        ("node_id exists in nodemap", not any(i["category"] == "node_id_missing_in_nodemap" for i in issues)),
        ("parent folder matches filename node_id", not any(i["category"] == "parent_folder_node_mismatch" for i in issues)),
        ("duplicate view_id", not any(i["category"] == "duplicate_view_id" for i in issues)),
    ]
    for label, ok in required:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {label}")
    lines.append("")
    lines.append("## Manual Review Candidates")
    lines.append("")
    if issues:
        for issue in issues:
            lines.append(f"- [{issue['severity']}] {issue['category']}: {issue['image_path']} - {issue['message']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Node Coverage")
    lines.append("")
    for node_id, count in sorted(node_counts.items(), key=lambda kv: int(kv[0])):
        lines.append(f"- {node_id}: {count} views ({node_names.get(node_id, '')})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The nodemap was read only for node_id validation.")
    lines.append("- DEST, ENT, and door-like views are preserved and marked as use_for_localization=false candidates.")
    lines.append("- DEST and ENT are matched as label tokens, so words like center_context are not treated as entrance views.")
    lines.append("- direction_to is filled only for to/toward/back_to followed by digits.")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved views: {views_csv}")
    print(f"Saved report CSV: {report_csv}")
    print(f"Saved report MD: {report_md}")
    print(f"views={len(rows)} issues={len(issues)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--view_root", default="view/3F")
    parser.add_argument("--nodemap", default="data/coex/source/coex_nodemap_m11.xlsx")
    parser.add_argument("--out_dir", default="data/coex/localization")
    parser.add_argument("--report_dir", default="data/coex/reports")
    parser.add_argument("--views_name", default="views.csv")
    parser.add_argument("--report_csv_name", default="view_validation_report.csv")
    parser.add_argument("--report_md_name", default="view_validation_report.md")
    build_views(parser.parse_args())


if __name__ == "__main__":
    main()
