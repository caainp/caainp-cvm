# nav_engine.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import sys

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    DEFAULT_DEVICE = "cpu"

# -----------------------------
# 0. 경로 설정 (nav_demo와 거의 동일)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent      # .../caainp_cvm (패키지 폴더)
CVM_ROOT = THIS_DIR.parent                      # .../caainp-cvm (프로젝트 루트)

sys.path.append(str(CVM_ROOT))

# from graph_4f import Graph4F
from caainp_csm.graph_4f import Graph4F
# from plan_csm import create_simple_plan, update_state_with_node, PlanState
from caainp_csm.plan_csm import create_simple_plan, update_state_with_node, PlanState
from scripts.run_cvm_step import run_cvm_step


CSV_PATH = CVM_ROOT / "ai_4f_node_map_fixed_embeded.csv"
NODE_IMAGES_DIR = CVM_ROOT / "node_images" / "node_images"

def init_plan(user_text: str, start_room: int) -> Tuple[PlanState, str]:
    """
    자연어 지시와 시작 방 번호를 받아서
    - PlanState 객체
    - CSV 경로 문자열
    을 돌려줌.
    """
    g = Graph4F(str(CSV_PATH))
    plan: PlanState = create_simple_plan(user_text, g, start_room=start_room)
    return plan, str(CSV_PATH)

def make_move_instruction(current_node: int,
                          next_node: Optional[int]) -> Dict[str, Any]:
    """
    current_node에서 next_node로 갈 때
    - direction_type
    - angle_deg
    - text_ko
    를 만들어주는 헬퍼.

    ⚠️ 지금은 아주 단순한 버전 (모든 이동을 STRAIGHT으로 가정).
       나중에 node 좌표/방향을 이용해 LEFT/RIGHT, angle_deg 계산을 넣으면 됨.
    """
    if next_node is None:
        # 더 이상 갈 곳이 없을 때 (도착)
        return {
            "direction_type": "STRAIGHT",
            "angle_deg": 0.0,
            "text_ko": "목적지에 도착했습니다."
        }

    # TODO: current_node, next_node 위치를 이용해서 방향/각도 계산하기
    # 지금은 단순히 직진으로 가정
    return {
        "direction_type": "STRAIGHT",
        "angle_deg": 0.0,
        "text_ko": "다음 랜드마크 방향으로 이동하세요."
    }

def make_route_summary(plan: PlanState) -> Dict[str, Any]:
    """
    PlanState에서 현재 단계/전체 단계/남은 경유지 텍스트를 만들어줌.
    PlanState 구조에 맞게 필요시 필드 이름은 수정해줘야 함.
    """
    current_step = plan.current_step
    total_steps = len(plan.steps)

    # remaining_steps_text 예시: "엘리베이터 → 4층 410호 → 7층 7201호"

    remaining_step_objs = plan.steps[current_step - 1 :]

    # 각 step마다 사람이 읽을 이름을 뽑는 헬퍼 (TODO: 정확한 필드 확인 후 수정)
    def step_label(step_obj) -> str:
        # 예: step_obj.description 이 있다면 그걸 사용
        if hasattr(step_obj, "description"):
            return step_obj.description
        # target_rooms 같은 리스트가 있다면 첫 번째 방 번호를 사용
        if hasattr(step_obj, "target_rooms"):
            return " / ".join(map(str, step_obj.target_rooms))
        # 아무 것도 없으면 step_id
        return f"step{step_obj.step_id}"

    labels = [step_label(s) for s in remaining_step_objs]
    remaining_text = " → ".join(labels) if labels else ""

    # via_nodes는 앞으로 거쳐야 할 주요 노드 id를 넣으면 된다.
    # 여기서는 각 step의 target_nodes 첫 번째 값만 모으는 예시.
    via_nodes = []
    for s in remaining_step_objs:
        if hasattr(s, "target_nodes") and s.target_nodes:
            via_nodes.append(int(s.target_nodes[0]))

    return {
        "current_step": int(current_step),
        "total_steps": int(total_steps),
        "remaining_steps_text": remaining_text,
        "via_nodes": via_nodes,
    }

def compute_nav_output(image_path: str,
                       plan: PlanState,
                       prev_node: Optional[int],
                       csv_path: Optional[str] = None,
                       device: Optional[str] = None) -> Dict[str, Any]:
    """
    한 프레임 단위로:
    - CVM (run_cvm_step) 호출 → 현재 노드 + value map
    - CSM (update_state_with_node) 호출 → 플랜 상태 업데이트
    - NavOutput JSON 생성

    반환값은:
    {
      "nav_output": {...},   # AR에 넘길 JSON
      "plan_json": {...},    # 업데이트된 PlanState JSON
      "debug": {...}         # 선택: cvm_result, value_map 등
    }
    """
    if csv_path is None:
        csv_path = str(CSV_PATH)

    if device is None:
        device = DEFAULT_DEVICE

    # 0) CVM: PlanState → JSON으로만 변환해서 CVM에 넘김
    plan_json = plan.to_json()

    # 1) CVM: 이미지 → 현재 위치 + value map
    cvm_out = run_cvm_step(
        image_path=image_path,
        csv_path=csv_path,
        plan_json=plan_json,
        prev_node=prev_node,
        device=device,
    )

    cvm_result = cvm_out["cvm_result"]      # {"current_node", "confidence", ...}
    value_map = cvm_out["value_map"]        # {"current_node", "neighbor_values", ...}

    current_node = int(cvm_result["current_node"])

    # neighbor_values: {이웃 노드 id: value}
    neighbors = value_map.get("neighbor_values", {})
    next_node = None
    if neighbors:
        # value가 가장 큰 이웃 하나 선택
        next_node = int(max(neighbors.items(), key=lambda kv: kv[1])[0])
    
    # 2) CSM: JSON → PlanState 복원 단계는 건너뛰고,
    #    이미 들고 있는 plan 객체를 그대로 업데이트
    update_state_with_node(plan, current_node=current_node, stay_frames=2)
    updated_plan_json = plan.to_json()

    # 3) move_instruction, route_summary 만들기
    move_instruction = make_move_instruction(current_node, next_node)
    route_summary = make_route_summary(plan)

    # 4) 최종 NavOutput JSON 조립
    nav_output = {
        "schema_version": 1,
        "current_node": current_node,
        "next_node": next_node,
        "move_instruction": move_instruction,
        "route_summary": route_summary,
    }

    return {
        "nav_output": nav_output,
        "plan": plan,                # PlanState 객체 (서버 내부용)
        "plan_json": updated_plan_json,  # 필요하면 AR/로그용으로 사용
        "debug": {
            "cvm_result": cvm_result,
            "value_map": value_map,
        },
    }

if __name__ == "__main__":
    # 1) 플랜 초기화
    plan, csv_path = init_plan(
        user_text="401에서 410호까지 가기",
        start_room=401,
    )

    prev_node = None

    # 2) 테스트용 이미지 시퀀스
    img_sequence = ["401.jpg", "4102.jpg", "4201.jpg"]

    for img_name in img_sequence:
        image_path = NODE_IMAGES_DIR / img_name

        out = compute_nav_output(
            image_path=str(image_path),
            plan=plan,
            prev_node=prev_node,
            csv_path=csv_path,
        )

        nav_output = out["nav_output"]
        plan = out["plan"]                  # 업데이트된 PlanState 계속 유지
        plan_json = out["plan_json"]        # 필요시 로그/디버깅용
        prev_node = nav_output["current_node"]

        from pprint import pprint
        print(f"\n[IMAGE] {img_name}")
        pprint(nav_output)
