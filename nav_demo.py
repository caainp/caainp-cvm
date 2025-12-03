# nav_demo.py  (위치: C:\Users\ljdkr\nav\caainp-cvm\nav_demo.py)

from pathlib import Path
import sys
from pprint import pprint

# -----------------------------
# 0. 경로 설정
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent       # ...\nav\caainp-cvm
NAV_DIR = THIS_DIR.parent                        # ...\nav

CSM_SRC = NAV_DIR / "caainp-csm" / "src"         # graph_4f.py, plan_csm.py 있는 곳
CVM_ROOT = THIS_DIR                              # caainp-cvm 루트

# import 경로에 추가
sys.path.append(str(CSM_SRC))
sys.path.append(str(CVM_ROOT))

print("[DEBUG] CSM_SRC =", CSM_SRC)
print("[DEBUG] CVM_ROOT =", CVM_ROOT)

# -----------------------------
# 1. CSM / CVM 모듈 import
# -----------------------------
from graph_4f import Graph4F
from plan_csm import create_simple_plan, update_state_with_node
from scripts.run_cvm_step import run_cvm_step


# -----------------------------
# 2. 헬퍼: 텍스트 → 플랜
# -----------------------------
def make_plan_from_text(user_text: str, start_room: int):
    """
    사용자 자연어 + 시작 방번호 → (PlanState, plan_json, csv_path)
    """
    csv_path = CVM_ROOT / "ai_4f_node_map_fixed_embeded.csv"
    g = Graph4F(str(csv_path))
    plan = create_simple_plan(user_text, g, start_room=start_room)
    return plan, plan.to_json(), csv_path


# -----------------------------
# 3. 데모 메인 루프
# -----------------------------
def main():
    node_images_dir = CVM_ROOT / "node_images/node_images"

    # 1) CSM: 유저 요청으로 플랜 생성
    user_text = "401에서 410호까지 가기"
    plan, plan_json, csv_path = make_plan_from_text(
        user_text=user_text,
        start_room=401,
    )

    print("\n[PLAN JSON]")
    pprint(plan_json)

    prev_node = None

    # 2) 여러 프레임 시퀀스 처리 시뮬레이션
    img_sequence = ["401.jpg", "4102.jpg", "4201.jpg"]

    for idx, img_name in enumerate(img_sequence, start=1):
        image_path = node_images_dir / img_name
        if not image_path.exists():
            print(f"[WARN] image not found: {image_path}, skip")
            continue

        # 2) CVM: 한 프레임 처리 (현재 위치 + value map)
        out = run_cvm_step(
            image_path=str(image_path),
            csv_path=str(csv_path),
            plan_json=plan_json,
            prev_node=prev_node,
            device="cpu",
        )

        cvm = out["cvm_result"]
        vm = out["value_map"]
        current_node = int(cvm["current_node"])
        prev_node = current_node

        print(f"\n[FRAME {idx}] image={img_name}")
        print("  current_node:", current_node)
        print("  confidence  :", cvm.get("confidence"))

        # 상위 3개 이웃만 출력
        neighbors = vm["neighbor_values"]
        top3 = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("  top neighbors:", top3)

        # 3) CSM: 현재 노드로 플랜 상태 업데이트
        update_state_with_node(plan, current_node=current_node, stay_frames=2)
        print("  current_step:", plan.current_step)
        print("  steps_status:", plan.steps_status)

        # step이 바뀌었을 수 있으니 JSON 다시 뽑기
        plan_json = plan.to_json()


if __name__ == "__main__":
    main()
