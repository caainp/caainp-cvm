from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

try:
    from scripts.coex_view_adapter import localize_query  # type: ignore
    from scripts.map_loader import load_map_csv  # type: ignore
    from scripts.value_map import build_value_map_v2  # type: ignore
except ImportError:
    from coex_view_adapter import localize_query  # type: ignore
    from map_loader import load_map_csv  # type: ignore
    from value_map import build_value_map_v2  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _relative_to_root(root_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(root_dir).as_posix()


def _extract_route_from_plan_json(plan_json: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    current_step_id = int(plan_json.get("current_step", 1))
    steps = plan_json.get("steps", []) or []
    step_by_id = {
        int(step["step_id"]): step
        for step in steps
        if "step_id" in step
    }
    if current_step_id not in step_by_id:
        raise ValueError(f"current_step {current_step_id} not found in plan_json.steps")
    step = step_by_id[current_step_id]
    route_nodes = [int(n) for n in step.get("route_nodes", []) or []]
    target_nodes = [int(n) for n in step.get("target_nodes", []) or []]
    if not route_nodes:
        raise ValueError(f"Step {current_step_id} has empty route_nodes")
    if not target_nodes:
        target_nodes = [int(route_nodes[-1])]
    return route_nodes, target_nodes


def _parse_blocked_edges(blocked_edges: Optional[Iterable[Any]]) -> List[Tuple[int, int]]:
    parsed: List[Tuple[int, int]] = []
    for edge in blocked_edges or []:
        if isinstance(edge, str):
            cleaned = edge.replace("-", ",").replace(":", ",")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Invalid blocked edge: {edge!r}")
            parsed.append((int(parts[0]), int(parts[1])))
        else:
            u, v = edge
            parsed.append((int(u), int(v)))
    return parsed


def _choose_next_node(value_map: Dict[str, Any]) -> Optional[int]:
    neighbor_values = value_map.get("neighbor_values", {}) or {}
    if not neighbor_values:
        return None
    return int(max(neighbor_values.items(), key=lambda kv: float(kv[1]))[0])


def _candidate_payload(node_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"node_id": int(row["node_id"]), "score": float(row["node_score"])}
        for row in node_rows
    ]


def _node_label(node_records: Dict[int, Any], node_id: int) -> str:
    rec = node_records.get(int(node_id))
    if rec is None:
        return str(node_id)
    for key in ("name_ko", "name", "label"):
        value = (getattr(rec, "extra", {}) or {}).get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if getattr(rec, "description", None):
        return str(rec.description).strip()
    return str(node_id)


def _make_move_instruction(next_node: Optional[int], node_records: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
    if next_node is None:
        return {
            "direction_type": "ARRIVED_OR_NO_NEIGHBOR",
            "angle_deg": None,
            "text_ko": "다음 이동 노드를 결정할 수 없습니다.",
        }
    label = _node_label(node_records or {}, int(next_node))
    target_text = f"{label}({next_node})" if label != str(next_node) else str(next_node)
    return {
        "direction_type": "NODE",
        "angle_deg": None,
        "text_ko": f"{target_text} 방향으로 이동하세요.",
    }


def _make_route_summary(route_nodes: List[int], target_nodes: List[int]) -> Dict[str, Any]:
    return {
        "route_nodes": route_nodes,
        "target_nodes": target_nodes,
        "remaining_node_count": len(route_nodes),
    }


def _resolve_route(
    *,
    graph: nx.Graph,
    current_node: int,
    target_node: Optional[int],
    plan_json: Optional[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    if plan_json is not None:
        return _extract_route_from_plan_json(plan_json)
    if target_node is None:
        raise ValueError("target_node is required when plan_json is not provided")
    if target_node not in graph:
        raise ValueError(f"target_node {target_node} is not in graph")
    route_nodes = [int(n) for n in nx.shortest_path(graph, current_node, int(target_node))]
    return route_nodes, [int(target_node)]


def run_coex_cvm_step(
    *,
    image_path: str,
    target_node: Optional[int] = None,
    plan_json: Optional[Dict[str, Any]] = None,
    root_dir: Optional[str] = None,
    graph_csv: str = "data/coex/graph/coex_nodemap.csv",
    embeddings: str = "data/coex/localization/view_embeddings.npy",
    index: str = "data/coex/localization/view_embedding_index.csv",
    include_all_views: bool = False,
    node_topk: int = 5,
    view_topk: int = 10,
    blocked_edges: Optional[Iterable[Any]] = None,
    device: str = "auto",
    model: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    aug_times: int = 1,
) -> Dict[str, Any]:
    """
    Thin COEX CVM wrapper.

    Sources stay separated:
    - views.csv/view_embeddings.npy: localization source
    - coex_nodemap.csv: graph/value-map source
    - output to CSM/AR: node_id-based current/next/route data
    """
    root = Path(root_dir).resolve() if root_dir else _repo_root()
    graph_path = _resolve_path(root, graph_csv)

    view_args = SimpleNamespace(
        root_dir=str(root),
        query_image=image_path,
        embeddings=embeddings,
        index=index,
        node_topk=max(int(node_topk), 1),
        view_topk=max(int(view_topk), 1),
        include_all_views=bool(include_all_views),
        model=model,
        pretrained=pretrained,
        device=device,
        aug_times=int(aug_times),
    )
    view_rows, node_rows = localize_query(view_args)
    if not node_rows:
        raise RuntimeError("No node candidates returned by COEX view localization")

    graph, node_records, _, _ = load_map_csv(str(graph_path))
    current_node = int(node_rows[0]["node_id"])
    if current_node not in graph:
        raise ValueError(f"current_node {current_node} is not in graph")

    route_nodes, target_nodes = _resolve_route(
        graph=graph,
        current_node=current_node,
        target_node=target_node,
        plan_json=plan_json,
    )
    blocked = _parse_blocked_edges(blocked_edges)
    cvm_candidates = _candidate_payload(node_rows)
    value_map = build_value_map_v2(
        current_node=current_node,
        route_nodes=route_nodes,
        target_nodes=target_nodes,
        graph=graph,
        cvm_candidates=cvm_candidates,
        blocked_edges=blocked,
    )
    next_node = _choose_next_node(value_map)

    cvm_result = {
        "current_node": current_node,
        "confidence": float(node_rows[0]["node_score"]),
        "candidates": cvm_candidates,
        "debug": {
            "node_candidates": node_rows,
            "view_candidates": view_rows,
            "include_all_views": bool(include_all_views),
            "graph_csv": _relative_to_root(root, graph_path),
            "localization_source": {
                "embeddings": embeddings,
                "index": index,
            },
        },
    }
    nav_output = {
        "schema_version": 1,
        "current_node": current_node,
        "next_node": next_node,
        "route_nodes": route_nodes,
        "target_nodes": target_nodes,
        "move_instruction": _make_move_instruction(next_node, node_records),
        "route_summary": _make_route_summary(route_nodes, target_nodes),
    }
    return {
        "nav_output": nav_output,
        "cvm_result": cvm_result,
        "value_map": value_map,
        "blocked_edges": blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=str(_repo_root()))
    parser.add_argument("--image", "--query_image", dest="image_path", required=True)
    parser.add_argument("--target_node", type=int, default=None)
    parser.add_argument("--plan_json", default=None, help="Optional PlanState.to_json file")
    parser.add_argument("--graph_csv", default="data/coex/graph/coex_nodemap.csv")
    parser.add_argument("--embeddings", default="data/coex/localization/view_embeddings.npy")
    parser.add_argument("--index", default="data/coex/localization/view_embedding_index.csv")
    parser.add_argument("--include_all_views", action="store_true")
    parser.add_argument("--node_topk", type=int, default=5)
    parser.add_argument("--view_topk", type=int, default=10)
    parser.add_argument("--blocked_edge", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--aug_times", type=int, default=1)
    parser.add_argument("--json_out", default="data/coex/reports/smoke/run_coex_cvm_step_smoke.json")
    parser.add_argument("--no_output", action="store_true")
    args = parser.parse_args()

    plan_json = None
    if args.plan_json:
        plan_json_path = _resolve_path(Path(args.root_dir).resolve(), args.plan_json)
        plan_json = json.loads(plan_json_path.read_text(encoding="utf-8"))

    result = run_coex_cvm_step(
        image_path=args.image_path,
        target_node=args.target_node,
        plan_json=plan_json,
        root_dir=args.root_dir,
        graph_csv=args.graph_csv,
        embeddings=args.embeddings,
        index=args.index,
        include_all_views=bool(args.include_all_views),
        node_topk=int(args.node_topk),
        view_topk=int(args.view_topk),
        blocked_edges=args.blocked_edge,
        device=args.device,
        model=args.model,
        pretrained=args.pretrained,
        aug_times=int(args.aug_times),
    )
    print(json.dumps(result["nav_output"], ensure_ascii=False, indent=2))

    if not args.no_output:
        out_path = _resolve_path(Path(args.root_dir).resolve(), args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
