from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

try:
    from scripts.coex_view_adapter import localize_query  # type: ignore
    from scripts.map_loader import load_map_csv  # type: ignore
    from scripts.value_map import build_value_map_v2  # type: ignore
except ImportError:
    from coex_view_adapter import localize_query  # type: ignore
    from map_loader import load_map_csv  # type: ignore
    from value_map import build_value_map_v2  # type: ignore


TARGET_VIEW_ALIASES: Dict[Tuple[int, int], Tuple[str, ...]] = {
    (3131, 3002): ("toc",),
    (3131, 3062): ("toc",),
}
TERMINAL_DESTINATION_PAIRS = {
    (3014, 3080),
    (3050, 3049),
    *TARGET_VIEW_ALIASES.keys(),
}


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


def _graph_without_blocked_edges(
    graph: nx.Graph,
    blocked_edges: Sequence[Tuple[int, int]],
) -> nx.Graph:
    if not blocked_edges:
        return graph
    nav_graph = graph.copy()
    for u, v in blocked_edges:
        if nav_graph.has_edge(int(u), int(v)):
            nav_graph.remove_edge(int(u), int(v))
    return nav_graph


def _choose_next_node(value_map: Dict[str, Any]) -> Optional[int]:
    neighbor_values = value_map.get("neighbor_values", {}) or {}
    if not neighbor_values:
        return None
    return int(max(neighbor_values.items(), key=lambda kv: float(kv[1]))[0])


def _blocked_edge_set(blocked_edges: Sequence[Tuple[int, int]]) -> set[Tuple[int, int]]:
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in blocked_edges
        if int(u) != int(v)
    }


def _edge_is_blocked(u: int, v: int, blocked_edges: Sequence[Tuple[int, int]]) -> bool:
    return (min(int(u), int(v)), max(int(u), int(v))) in _blocked_edge_set(blocked_edges)


def _choose_route_next_node(
    *,
    current_node: int,
    route_nodes: Sequence[int],
    graph: nx.Graph,
    blocked_edges: Sequence[Tuple[int, int]],
) -> Optional[int]:
    route = [int(n) for n in route_nodes]
    current = int(current_node)
    blocked = _blocked_edge_set(blocked_edges)
    for idx, node_id in enumerate(route[:-1]):
        if int(node_id) != current:
            continue
        next_node = int(route[idx + 1])
        edge = (min(current, next_node), max(current, next_node))
        if graph.has_edge(current, next_node) and edge not in blocked:
            return next_node
        return None
    return None


def _is_arrived(current_node: int, route_nodes: Sequence[int], target_nodes: Sequence[int]) -> bool:
    current = int(current_node)
    targets = {int(n) for n in target_nodes}
    if current in targets:
        return True
    route = [int(n) for n in route_nodes]
    return len(route) == 1 and route[0] == current


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


def _normalize_view_label(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _int_or_none(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _target_visibility_from_views(
    *,
    current_node: int,
    target_nodes: Sequence[int],
    view_rows: Sequence[Dict[str, Any]],
    graph: nx.Graph,
    blocked_edges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    targets = [int(n) for n in target_nodes]
    if len(targets) != 1:
        return {"visible": False, "reason": "target_count_not_one"}
    current = int(current_node)
    target_node = int(targets[0])
    if current == target_node:
        return {"visible": False, "reason": "already_arrived", "target_node": target_node}
    if (current, target_node) not in TERMINAL_DESTINATION_PAIRS:
        return {"visible": False, "reason": "not_terminal_destination", "target_node": target_node}
    if not graph.has_edge(current, target_node):
        return {"visible": False, "reason": "not_adjacent", "target_node": target_node}
    if _edge_is_blocked(current, target_node, blocked_edges):
        return {"visible": False, "reason": "edge_blocked", "target_node": target_node}

    aliases = TARGET_VIEW_ALIASES.get((current, target_node), ())
    for row in view_rows:
        if int(row.get("node_id", -1)) != current:
            continue
        direction_to = _int_or_none(row.get("direction_to"))
        label = _normalize_view_label(row.get("view_label") or row.get("view_id"))
        points_by_direction = direction_to == target_node
        points_by_alias = any(label.startswith(alias) for alias in aliases)
        if points_by_direction or points_by_alias:
            return {
                "visible": True,
                "reason": "direction_to" if points_by_direction else "view_alias",
                "current_node": current,
                "target_node": target_node,
                "view_id": row.get("view_id"),
                "view_label": row.get("view_label"),
                "direction_to": row.get("direction_to"),
                "rank": row.get("rank"),
                "view_score": row.get("view_score"),
            }

    return {"visible": False, "reason": "no_view_evidence", "target_node": target_node}


def _make_move_instruction(
    next_node: Optional[int],
    node_records: Optional[Dict[int, Any]] = None,
    *,
    arrived: bool = False,
    target_in_sight: bool = False,
    final_leg: bool = False,
    target_node: Optional[int] = None,
) -> Dict[str, Any]:
    # Keep the public AR-facing instruction compatible with the existing AR enum.
    # Fine-grained arrival semantics are exposed through nav_output.arrival_state.
    if arrived:
        return {
            "direction_type": "STRAIGHT",
            "angle_deg": 0.0,
            "text_ko": "목적지에 도착했습니다.",
        }
    if target_in_sight and target_node is not None:
        label = _node_label(node_records or {}, int(target_node))
        target_text = f"{label}({target_node})" if label != str(target_node) else str(target_node)
        return {
            "direction_type": "STRAIGHT",
            "angle_deg": 0.0,
            "text_ko": f"{target_text} 목적지가 정면에 보입니다. 직진해 도착하세요.",
        }
    if final_leg and target_node is not None:
        label = _node_label(node_records or {}, int(target_node))
        target_text = f"{label}({target_node})" if label != str(target_node) else str(target_node)
        return {
            "direction_type": "STRAIGHT",
            "angle_deg": 0.0,
            "text_ko": f"{target_text} 방향으로 직진해 마지막 구간을 이동하세요.",
        }
    if next_node is None:
        return {
            "direction_type": "STRAIGHT",
            "angle_deg": 0.0,
            "text_ko": "다음 이동 노드를 결정할 수 없습니다.",
        }
    label = _node_label(node_records or {}, int(next_node))
    target_text = f"{label}({next_node})" if label != str(next_node) else str(next_node)
    return {
        "direction_type": "STRAIGHT",
        "angle_deg": 0.0,
        "text_ko": f"{target_text} 방향으로 이동하세요.",
    }


def _make_route_summary(route_nodes: List[int], target_nodes: List[int]) -> Dict[str, Any]:
    remaining_text = " -> ".join(str(int(node)) for node in route_nodes)
    return {
        "current_step": 1,
        "total_steps": 1,
        "remaining_steps_text": remaining_text,
        "via_nodes": [int(node) for node in target_nodes],
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
    blocked_edges: Sequence[Tuple[int, int]] = (),
) -> Tuple[List[int], List[int]]:
    if plan_json is not None:
        return _extract_route_from_plan_json(plan_json)
    if target_node is None:
        raise ValueError("target_node is required when plan_json is not provided")
    if target_node not in graph:
        raise ValueError(f"target_node {target_node} is not in graph")
    nav_graph = _graph_without_blocked_edges(graph, blocked_edges)
    route_nodes = [int(n) for n in nx.shortest_path(nav_graph, current_node, int(target_node))]
    return route_nodes, [int(target_node)]


def _safe_shortest_path_length(graph: nx.Graph, source: int, target: int) -> Optional[int]:
    try:
        return int(nx.shortest_path_length(graph, int(source), int(target)))
    except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
        return None


def _distance_score(graph: nx.Graph, node_id: int, anchors: Sequence[int]) -> float:
    distances = [
        d
        for anchor in anchors
        if (d := _safe_shortest_path_length(graph, int(node_id), int(anchor))) is not None
    ]
    if not distances:
        return 0.0
    return 1.0 / (1.0 + float(min(distances)))


def _build_context_route_hint(
    *,
    graph: nx.Graph,
    target_node: Optional[int],
    plan_json: Optional[Dict[str, Any]],
    previous_node: Optional[int],
) -> Tuple[List[int], List[int]]:
    if plan_json is not None:
        return _extract_route_from_plan_json(plan_json)
    if previous_node is not None and target_node is not None:
        try:
            route = [int(n) for n in nx.shortest_path(graph, int(previous_node), int(target_node))]
            return route, [int(target_node)]
        except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
            return [], [int(target_node)]
    if target_node is not None:
        return [], [int(target_node)]
    return [], []


def _rerank_current_node(
    *,
    node_rows: List[Dict[str, Any]],
    graph: nx.Graph,
    route_nodes: Sequence[int],
    target_nodes: Sequence[int],
    previous_node: Optional[int],
    rerank_topk: int,
) -> Dict[str, Any]:
    if not node_rows:
        raise RuntimeError("No node candidates returned by COEX view localization")

    raw_current = int(node_rows[0]["node_id"])
    route = [int(n) for n in route_nodes]
    targets = [int(n) for n in target_nodes]
    has_context = bool(route) or previous_node is not None
    if not has_context:
        return {
            "node_id": raw_current,
            "reranked": False,
            "reason": "no_context",
            "raw_current_node": raw_current,
            "candidates": [],
        }

    candidates = node_rows[: max(int(rerank_topk), 1)]
    scores = [float(row["node_score"]) for row in candidates]
    min_score = min(scores)
    max_score = max(scores)
    denom = (max_score - min_score) or 1.0

    previous = int(previous_node) if previous_node is not None else None
    route_set = set(route)
    details: List[Dict[str, Any]] = []

    for rank, row in enumerate(candidates, start=1):
        node_id = int(row["node_id"])
        raw_score = float(row["node_score"])
        clip_norm = (raw_score - min_score) / denom if max_score != min_score else 1.0
        rank_prior = 1.0 - ((rank - 1) / max(len(candidates) - 1, 1))
        visual_score = 0.7 * clip_norm + 0.3 * rank_prior

        if route:
            route_score = 1.0 if node_id in route_set else _distance_score(graph, node_id, route)
        else:
            route_score = 0.0
        previous_score = _distance_score(graph, node_id, [previous]) if previous is not None else 0.0
        target_score = _distance_score(graph, node_id, targets) if targets else 0.0

        if previous is not None:
            total = (
                0.45 * visual_score
                + 0.25 * route_score
                + 0.25 * previous_score
                + 0.05 * target_score
            )
        else:
            total = 0.65 * visual_score + 0.30 * route_score + 0.05 * target_score

        details.append({
            "rank": rank,
            "node_id": node_id,
            "raw_score": raw_score,
            "visual_score": float(visual_score),
            "route_score": float(route_score),
            "previous_score": float(previous_score),
            "target_score": float(target_score),
            "context_score": float(total),
        })

    selected = max(details, key=lambda row: (float(row["context_score"]), -int(row["rank"])))
    return {
        "node_id": int(selected["node_id"]),
        "reranked": int(selected["node_id"]) != raw_current,
        "reason": "context_score",
        "raw_current_node": raw_current,
        "previous_node": previous,
        "route_nodes": route,
        "target_nodes": targets,
        "candidates": details,
    }


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
    previous_node: Optional[int] = None,
    context_rerank: bool = True,
    rerank_topk: int = 10,
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
        node_topk=max(int(node_topk), int(rerank_topk) if context_rerank else 1, 1),
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
    blocked = _parse_blocked_edges(blocked_edges)
    nav_graph = _graph_without_blocked_edges(graph, blocked)
    route_hint_nodes, target_hint_nodes = _build_context_route_hint(
        graph=nav_graph,
        target_node=target_node,
        plan_json=plan_json,
        previous_node=previous_node,
    )
    rerank_debug = (
        _rerank_current_node(
            node_rows=node_rows,
            graph=graph,
            route_nodes=route_hint_nodes,
            target_nodes=target_hint_nodes,
            previous_node=previous_node,
            rerank_topk=rerank_topk,
        )
        if context_rerank
        else {
            "node_id": int(node_rows[0]["node_id"]),
            "reranked": False,
            "reason": "disabled",
            "raw_current_node": int(node_rows[0]["node_id"]),
            "candidates": [],
        }
    )
    current_node = int(rerank_debug["node_id"])
    if current_node not in graph:
        raise ValueError(f"current_node {current_node} is not in graph")

    route_nodes, target_nodes = _resolve_route(
        graph=nav_graph,
        current_node=current_node,
        target_node=target_node,
        plan_json=plan_json,
        blocked_edges=(),
    )
    cvm_candidates = _candidate_payload(node_rows)
    value_map = build_value_map_v2(
        current_node=current_node,
        route_nodes=route_nodes,
        target_nodes=target_nodes,
        graph=graph,
        cvm_candidates=cvm_candidates,
        blocked_edges=blocked,
    )
    arrived = _is_arrived(current_node, route_nodes, target_nodes)
    route_next_node = None if arrived else _choose_route_next_node(
        current_node=current_node,
        route_nodes=route_nodes,
        graph=graph,
        blocked_edges=blocked,
    )
    next_node = None if arrived else (route_next_node or _choose_next_node(value_map))
    target_visibility = (
        {"visible": False, "reason": "already_arrived"}
        if arrived
        else _target_visibility_from_views(
            current_node=current_node,
            target_nodes=target_nodes,
            view_rows=view_rows,
            graph=graph,
            blocked_edges=blocked,
        )
    )
    target_in_sight = bool(target_visibility.get("visible"))
    visible_target_node = _int_or_none(target_visibility.get("target_node"))
    final_leg = (
        not arrived
        and visible_target_node is not None
        and (current_node, visible_target_node) in TERMINAL_DESTINATION_PAIRS
        and next_node == visible_target_node
        and graph.has_edge(current_node, visible_target_node)
        and not _edge_is_blocked(current_node, visible_target_node, blocked)
    )
    arrival_state = (
        "ARRIVED"
        if arrived
        else ("IN_SIGHT" if target_in_sight else ("FINAL_LEG" if final_leg else "ROUTING"))
    )
    selected_node_row = next(
        (row for row in node_rows if int(row["node_id"]) == current_node),
        node_rows[0],
    )

    cvm_result = {
        "current_node": current_node,
        "confidence": float(selected_node_row["node_score"]),
        "candidates": cvm_candidates,
        "debug": {
            "node_candidates": node_rows,
            "view_candidates": view_rows,
            "context_rerank": rerank_debug,
            "route_next_node": route_next_node,
            "target_visibility": target_visibility,
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
        "arrival_state": arrival_state,
        "target_visibility": target_visibility,
        "move_instruction": _make_move_instruction(
            next_node,
            node_records,
            arrived=arrived,
            target_in_sight=target_in_sight,
            final_leg=final_leg,
            target_node=target_visibility.get("target_node"),
        ),
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
    parser.add_argument("--previous_node", type=int, default=None)
    parser.add_argument("--rerank_topk", type=int, default=10)
    parser.add_argument("--no_context_rerank", action="store_true")
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
        previous_node=args.previous_node,
        context_rerank=not bool(args.no_context_rerank),
        rerank_topk=int(args.rerank_topk),
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
