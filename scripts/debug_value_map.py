# scripts/debug_value_map.py

from __future__ import annotations

from pathlib import Path
from pprint import pprint

import networkx as nx

from scripts.map_loader import load_map_csv
from scripts.value_map import build_value_map
from caainp_cvm import _get_csv_path

def main() -> None:
    # 1) Resolve CSV path (repo root is the parent directory of this script)
    csv_path = _get_csv_path()

    print(f"[INFO] CSV path: {csv_path}")

    # 2) Load graph from CSV
    graph, node_records, emb_matrix, node_ids = load_map_csv(str(csv_path))
    assert isinstance(graph, nx.Graph)
    print(
        f"[INFO] graph loaded: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # 3) Example current / route / target node configuration for debugging
    #    It does not need to match the real structure exactly, only valid node IDs.
    current_node = 4101
    route_nodes = [4101, 4104, 4105, 4106, 4150]
    target_nodes = [4150]

    print(f"[INFO] current_node = {current_node}")
    print(f"[INFO] route_nodes   = {route_nodes}")
    print(f"[INFO] target_nodes  = {target_nodes}")

    # 4) Build value map for the given state
    value_map = build_value_map(
        current_node=current_node,
        route_nodes=route_nodes,
        target_nodes=target_nodes,
        graph=graph,
    )

    print("\n[RESULT] Raw ValueMap dict:")
    pprint(value_map)

    # 5) Print neighbor_values sorted in descending order for inspection
    print("\n[RESULT] Neighbor values (sorted desc):")
    items = sorted(
        value_map["neighbor_values"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for nid, val in items:
        print(f"  node {nid}: {val:.3f}")


if __name__ == "__main__":
    main()
