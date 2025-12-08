# scripts/run_cvm_step.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from pprint import pprint
import json
 
from scripts.localize_image import localize_image
from scripts.map_loader import load_map_csv
from scripts.value_map import build_value_map_v2
# -------------------------------------------------------------------
# 1. Extract route/target information from a CSM plan JSON
# -------------------------------------------------------------------

def extract_route_from_plan_json(plan_json: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    Extract route_nodes and target_nodes for the current step
    from a PlanState.to_json() style dictionary.

    Expected JSON structure (simplified):

    {
      "constraints": {...},
      "steps": [
        {
          "step_id": 1,
          "goal_type": "ROOM",
          "goal_room": 410,
          "allowed_moves": [...],
          "description_ko": "...",
          "target_nodes": [...],
          "route_nodes": [...]
        },
        ...
      ],
      "current_step": 1,
      "steps_status": [
        {"step_id": 1, "status": "IN_PROGRESS"},
        ...
      ]
    }
    """
    current_step_id = int(plan_json.get("current_step", 1))
    steps = plan_json.get("steps", [])

    step_by_id: Dict[int, Dict[str, Any]] = {
        int(s["step_id"]): s for s in steps
        if "step_id" in s
    }
    if current_step_id not in step_by_id:
        raise ValueError(f"current_step {current_step_id} not found in plan_json.steps")

    cur_step = step_by_id[current_step_id]
    route_nodes = [int(n) for n in cur_step.get("route_nodes", [])]
    target_nodes = [int(n) for n in cur_step.get("target_nodes", [])]

    if not route_nodes:
        raise ValueError(f"Step {current_step_id} has empty route_nodes")

    return route_nodes, target_nodes


# -------------------------------------------------------------------
# 2. Run CVM pipeline for a single frame
# -------------------------------------------------------------------

def run_cvm_step(
    *,
    image_path: str,
    csv_path: str,
    plan_json: Dict[str, Any],
    prev_node: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run the whole CVM pipeline for a single frame.

    Pipeline:
      1) localize_image: image → current_node (CLIP + OCR + geometry + prior)
      2) load_map_csv  : CSV → graph
      3) extract_route_from_plan_json: plan_json → route_nodes / target_nodes
      4) build_value_map_v2: current node + route/target + graph (+ candidates)
         → neighbor-wise value map

    Returns
    -------
    {
      "cvm_result": {...},   # raw localization result
      "value_map": {...}     # neighbor_values + debug info
    }
    """
    # Determine node_images directory from the CSV path
    root_dir = Path(csv_path).resolve().parent
    node_images_dir = root_dir / "node_images" / "node_images"

    cvm_result = localize_image(
        image_path=image_path,
        csv_path=csv_path,
        device=device,
        use_ocr=True,
        node_images_dir=str(node_images_dir),
        use_geo=True,
        prev_node=prev_node,
        w_clip=1.0,
        w_ocr=0.8,
        w_geo=0.4,
        w_prior=0.2,
        # model_name / pretrained / topk / auto_match_model: use defaults
    )
    current_node = int(cvm_result["current_node"])

    # 2) Load graph (map_loader returns a networkx graph and related metadata)
    graph, node_records, emb_matrix, node_ids = load_map_csv(csv_path)

    # 3) Extract route/target for current step from the CSM plan
    route_nodes, target_nodes = extract_route_from_plan_json(plan_json)

    # 4) Compute value map (Phase 2: route-based + visual candidates)
    blocked_edges: set[tuple[int, int]] = set()

    value_map = build_value_map_v2(
        current_node=current_node,
        route_nodes=route_nodes,
        target_nodes=target_nodes,
        graph=graph,
        cvm_candidates=cvm_result.get("candidates"),
        blocked_edges=blocked_edges,
    )

    return {
        "cvm_result": cvm_result,
        "value_map": value_map,
    }


# -------------------------------------------------------------------
# 3. Simple standalone demo (__main__)
# -------------------------------------------------------------------

def _load_dummy_plan_json() -> Dict[str, Any]:
    """
    Build a minimal dummy plan_json for local testing.

    In the real system, CSM will generate something like:

        plan = create_simple_plan(..., g, start_room=401)
        plan_json = plan.to_json()

    and pass that JSON to CVM via file / network / IPC.
    """
    # Single-step plan example: 401 → 4102 → 4106 → 4150
    # (In practice, 'route_nodes' will come from the real CSM planner.)
    plan_json = {
        "constraints": {
            "use_elevator_only": False,
            "avoid_stairs": False,
            "forbidden_nodes": [],
            "via_rooms": [],
        },
        "steps": [
            {
                "step_id": 1,
                "goal_type": "ROOM",
                "goal_room": 4150,
                "allowed_moves": ["CORRIDOR", "ELEVATOR", "STAIRS"],
                "description_ko": "401에서 4150호 강의실 앞까지 이동",
                "target_nodes": [4150],
                "route_nodes": [401, 4102, 4106, 4150],
            }
        ],
        "current_step": 1,
        "steps_status": [
            {"step_id": 1, "status": "IN_PROGRESS"},
        ],
    }
    return plan_json


def main() -> None:
    # Resolve repository root (caainp-cvm)
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "ai_4f_node_map_fixed_embeded.csv"
    image_path = root / "node_images" / "401.jpg"

    print(f"[INFO] csv_path   = {csv_path}")
    print(f"[INFO] image_path = {image_path}")

    # 1) Load dummy plan JSON (in production, this comes from CSM)
    plan_json = _load_dummy_plan_json()

    # 2) Run pipeline for a single frame
    out = run_cvm_step(
        image_path=str(image_path),
        csv_path=str(csv_path),
        plan_json=plan_json,
        prev_node=None,
        device="cpu",
    )

    print("\n[RESULT] cvm_result:")
    pprint(out["cvm_result"])

    print("\n[RESULT] value_map.neighbor_values (sorted desc):")
    neighbor_values = out["value_map"]["neighbor_values"]
    for nid, val in sorted(neighbor_values.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  node {nid}: {val:.3f}")

    # Optionally print full JSON packet for inspection
    print("\n[RESULT] full packet as JSON:")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
