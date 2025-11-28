# 🚀 CA-Nav Project  
**Constraint-Aware AR Navigation for AI Engineering Building 4F**

AI공학관 4층을 대상으로, **사전학습된 지도 + 이미지 기반 위치추정(CVM) + 자연어 지시/제약 분석(CSM) + AR 인터페이스**를 결합한 실내 내비게이션 프로토타입입니다.

- CVM 팀: **Camera/Vision Module** – 이미지에서 현재 위치 노드 추정 + value map 생성  
- CSM 팀: **Command/Constraint Module** – 자연어 지시를 단계별 플랜으로 구조화  
- NAV 팀: CVM + CSM 결과를 묶어 **“이번 스텝 안내 JSON”** 생성  
- AR 팀: JSON을 받아 **화살표 + 안내 문구**를 웹 AR로 렌더링

---

## 📂 프로젝트 구조 (Directory Structure)

> 실제 레포 이름 예시: `CA-Nav-Project/`

```text
CA-Nav-Project/
├── README.md
├── requirements.txt          # 📦 백엔드(Python) 공통 라이브러리
├── map_data/                 # 🗺️ 지도 CSV 및 임베딩 데이터
│   └── ai_building_4f.csv
├── modules/
│   ├── cvm/                  # 👁️ CVM 팀 – 위치추정 + value map
│   │   ├── map_loader.py
│   │   ├── embedding_utils.py
│   │   ├── localization.py
│   │   └── value_map.py
│   ├── csm/                  # 🧠 CSM 팀 – 자연어 지시/제약 → 플랜 + 상태관리
│   │   ├── planner.py
│   │   ├── state.py
│   │   └── pathfinder.py
│   └── nav/                  # 🧭 NAV – cvm + csm를 합쳐 최종 Nav JSON 생성
│       └── nav_engine.py
├── shared/                   # 🔁 공통 유틸, 타입 정의 등 (선택)
└── ar-interface/             # 📱 AR.js / A-Frame 코드 (현 레포에 이미 존재)
```

각 팀원은 **본인 담당 폴더 안에서만 작업**하고,  
최종적으로 `modules/nav/nav_engine.py`가 AR에 넘길 JSON을 생성하는 구조입니다.

---

## 🤝 공통 인터페이스 / 데이터 구조 (Interface Agreement)

모듈끼리의 입·출력을 미리 고정해 두어야, 서로 따로 작업해도 나중에 쉽게 합칠 수 있습니다.

### 1. CVM 결과 – `CvmResult`

```python
CvmResult = {
    "current_node": int,   # 현재 위치한 노드 ID
    "confidence": float    # 위치 추정 신뢰도(0~1)
    # 나중에 확장: "recognized": {...}  # OCR/표지판 결과 등
}
```

### 2. CSM 계획/상태 – `PlanState`

```python
PlanState = {
    "constraints": {       # 제약조건 (엘리베이터만, 계단금지 등)
        # 예: "use_elevator_only": True,
        #     "avoid_stairs": True,
        #     "forbidden_nodes": [ ... ]
    },
    "current_step": int,   # 현재 진행 중인 step 번호 (1부터 시작)

    "steps": [
        {
            "step_id": int,
            "target_nodes": [int, ...],    # 이 step의 목표 노드 ID 리스트
            "status": "PENDING" | "IN_PROGRESS" | "DONE",
            "description_ko": str,         # 이 단계 설명 (예: "엘리베이터 앞까지 이동")
            # 선택: "route_nodes": [int, ...]  # pathfinder가 채워 넣는 실제 경로
        },
        ...
    ]
}
```

### 3. CVM value map – `ValueMap`

```python
ValueMap = {
    "current_node": int,
    "neighbor_values": {
        int: float   # {이웃 노드 id: value(클수록 좋은 방향)}
    }
}
```

- 기본 규칙: **목표에 가까워지는 이웃일수록 높은 value**  
  (예: 그래프 거리 기반으로 `value = -distance` 혹은 `exp(-distance)`)

### 4. 최종 네비 출력 – `NavOutput` (백 → 프론트 JSON)

```python
NavOutput = {
    "schema_version": 1,
    "current_node": int,           # 현재 위치 노드 id (CVM 결과)
    "next_node": int | None,       # 이번 스텝에서 이동할 다음 노드 id

    "move_instruction": {          # 화면에 바로 쓰이는 안내 정보
        "direction_type": str,     # "STRAIGHT" | "LEFT" | "RIGHT" | "TURN_BACK" | ...
        "angle_deg": float,        # 화살표 회전 각도 (카메라 정면 기준 0°, 시계방향 +)
        "text_ko": str             # 예: "오른쪽 복도로 이동하세요."
    },

    "route_summary": {             # 남은 단계 요약 + 경유지
        "current_step": int,
        "total_steps": int,
        "remaining_steps_text": str,   # 예: "엘리베이터 → 4층 410호 → 7층 7201호"
        "via_nodes": [int, ...]        # 앞으로 거칠 주요 경유지 노드 id
    }
}
```

> **AR 팀은 `NavOutput`만 믿고 개발**하면 됩니다.  
> - `move_instruction.text_ko` → 안내 문구 UI  
> - `move_instruction.direction_type` / `angle_deg` → 화살표 회전/애니메이션  
> - `route_summary.*` → 상단/하단 “남은 단계/경유지” 표시

---

## 👁️ CVM 팀 – 역할 & 코드 구조

### 목표

> 업로드된 이미지 1장을 보고  
> → 지도 상에서 **현재 위치 노드**를 찾고  
> → 현재 노드 기준 이웃 방향별 **value map**을 만든다.

### 디렉토리 구조

```text
modules/cvm/
  ├─ map_loader.py       # CSV → 그래프/노드 정보 로딩
  ├─ embedding_utils.py  # CLIP 로딩, 이미지 → 임베딩
  ├─ localization.py     # localize_image(image) → CvmResult
  └─ value_map.py        # build_value_map(...) → ValueMap
```

### 핵심 함수 명세

1. `map_loader.py`

```python
def load_map(csv_path: str):
    """
    CSV 지도 파일을 읽어 NetworkX 그래프 + 노드 메타데이터를 반환.
    return: graph, node_dict
    """
```

2. `embedding_utils.py`

```python
def load_clip_model():
    """CLIP 모델/프로세서 로드"""

def encode_image(image) -> np.ndarray:
    """PIL.Image 또는 ndarray 입력 → 정규화된 임베딩 벡터 반환"""
```

3. `localization.py`

```python
def localize_image(image, map_db) -> CvmResult:
    """
    이미지 임베딩과 지도에 저장된 노드 임베딩을 비교하여
    가장 유사한 노드(current_node)와 confidence를 반환.
    """
```

4. `value_map.py`

```python
def build_value_map(current_node: int,
                    target_nodes: list[int],
                    graph) -> ValueMap:
    """
    current_node의 이웃 노드들에 대해
    '목표 target_nodes까지의 그래프 거리'를 기반으로 value를 계산.
    """
```

### 인원 분담 예시

- **CVM-1**  
  - `map_loader.py`, `embedding_utils.py`, `localization.py`  
  - → 지도 + 임베딩 + 위치 추정 흐름 완성
- **CVM-2**  
  - `value_map.py`, `run_cvm_step(...)` 헬퍼, 테스트 코드  
  - → value map + 안정화(필요시 smoothing) 담당

각자 `feature/cvm-localize`, `feature/cvm-valuemap` 같은 브랜치에서 작업 후  
`modules/cvm/` 범위만 PR로 올리는 식으로 협업.

---

## 🧠 CSM 팀 – 역할 & 코드 구조

### 목표

> 자연어 지시 → **제약 조건 포함 단계별 플랜(PlanState)** 생성  
> + CVM에서 오는 `current_node` 기반으로 **step 상태 업데이트 + 경로계획** 수행

### 디렉토리 구조

```text
modules/csm/
  ├─ planner.py      # 자연어 → 초기 PlanState 생성
  ├─ state.py        # PlanState 상태 머신 (step 완료 여부 갱신)
  └─ pathfinder.py   # CSV 그래프 기반 최단경로 (constraints 반영)
```

### 핵심 함수 명세

1. `planner.py`

```python
def parse_command(user_text: str) -> PlanState:
    """
    사용자 지시 문장을 파싱해서
    목적지/경유지/제약조건이 들어간 초기 PlanState 생성.
    (초기 MVP에서는 간단한 규칙/프롬프트로 구현)
    """
```

2. `state.py`

```python
def update_step_state(plan_state: PlanState,
                      current_node: int) -> PlanState:
    """
    CVM이 추정한 current_node를 보고
    각 step의 status (PENDING/IN_PROGRESS/DONE)를 갱신하고,
    current_step을 적절히 증가/유지한다.
    """
```

3. `pathfinder.py`

```python
def compute_route(graph,
                  plan_state: PlanState) -> PlanState:
    """
    제약조건(use_elevator_only, avoid_stairs 등)을 반영하여
    각 step의 target_nodes까지 최단경로(route_nodes)를 계산하고
    PlanState 안에 채워 넣는다.
    """
```

CSM 팀은 **이미지 없이도**  
가상 `current_node` 값을 바꿔 가며 `update_step_state` / `compute_route`를 테스트할 수 있습니다.

---

## 🧭 NAV 모듈 – CVM + CSM 통합

### 목표

> `CvmResult` + `PlanState` → **NavOutput(JSON)** 변환  
> (AR이 바로 쓸 수 있는 형태)

### 디렉토리 구조

```text
modules/nav/
  └─ nav_engine.py
```

### 핵심 함수 예시

```python
from modules.cvm.localization import localize_image
from modules.cvm.value_map import build_value_map
from modules.csm.state import update_step_state
from modules.csm.pathfinder import compute_route

def compute_nav_output(image,
                       plan_state: PlanState,
                       graph) -> NavOutput:
    # 1) CVM: 현재 위치 추정
    cvm_result = localize_image(image, graph)

    # 2) CSM: 상태 업데이트 + 경로계획
    plan_state = update_step_state(plan_state, cvm_result["current_node"])
    plan_state = compute_route(graph, plan_state)

    # 3) 현재 step / 목표 노드
    step = plan_state["steps"][plan_state["current_step"] - 1]
    target_nodes = step["target_nodes"]

    # 4) CVM: value map
    value_map = build_value_map(
        current_node=cvm_result["current_node"],
        target_nodes=target_nodes,
        graph=graph,
    )

    # 5) neighbor_values 기반 next_node 선택 (가장 value 높은 방향)
    neighbor_values = value_map["neighbor_values"]
    next_node = max(neighbor_values, key=neighbor_values.get)

    # 6) next_node 방향 → direction_type / angle_deg (간단 룰로 매핑)
    move_instruction = {
        "direction_type": "RIGHT",   # TODO: 실제 방향 계산 로직
        "angle_deg": 30.0,
        "text_ko": step["description_ko"]  # 필요시 디테일 문구 추가
    }

    route_summary = {
        "current_step": plan_state["current_step"],
        "total_steps": len(plan_state["steps"]),
        "remaining_steps_text": "...",  # CSM에서 만들어주거나 여기서 조합
        "via_nodes": []                 # PlanState 안 route_nodes 기반 생성
    }

    return {
        "schema_version": 1,
        "current_node": cvm_result["current_node"],
        "next_node": next_node,
        "move_instruction": move_instruction,
        "route_summary": route_summary,
    }
```

> 이 `compute_nav_output`이 **“백에서 프론트로 넘어가는 JSON”을 만드는 핵심 엔진**입니다.  

---

## 📱 AR 인터페이스 – 연동 방식

- 위치 추정, 경로 계산, 제약 처리 → **백엔드 모듈(CVM/CSM/NAV)**에서 담당
- AR.js / A-Frame 쪽은 **NavOutput JSON만 입력**으로 사용

### MVP 단계 아이디어

1. 백엔드(Python)는  
   - 로컬/Colab에서 `compute_nav_output()` 결과를  
     - 콘솔 출력하거나  
     - JSON 파일로 저장
2. AR/Web 쪽에서는
   - 미리 저장된 JSON 파일을 fetch해서
   - `move_instruction`와 `route_summary`를 화면에 반영
3. 이후 시간이 되면
   - FastAPI 등으로 `POST /step` API를 만들어
   - 업로드된 사진과 함께 실시간 연동 가능

---

## 🛠 Getting Started

```bash
# 1. 클론
git clone https://github.com/caainp/CA-Nav-Project.git
cd CA-Nav-Project

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. (예정) 간단 데모 실행
# python demos/run_nav_demo.py  # 예: 추후 추가
```

---

## ✅ 팀별 To-Do 요약

### 🗺️ Map & CVM 팀

- [ ] `map_data/`에 최종 4층 CSV + 임베딩 갱신  
- [ ] `localize_image(image) -> CvmResult` 함수 완성  
- [ ] `build_value_map(current_node, target_nodes, graph) -> ValueMap` 구현  
- [ ] 여러 노드에서 테스트 이미지로 위치/방향 검증

### 🧠 CSM 팀

- [ ] 시나리오 A 기준 자연어 → PlanState 파서 구현 (`parse_command`)  
- [ ] step 상태 머신 (`update_step_state`) 구현  
- [ ] 제약을 반영한 최단 경로 (`compute_route`) 구현  
- [ ] 가상 current_node 시퀀스로 시뮬레이션 테스트

### 🧭 NAV 팀 (혹은 CVM+CSM 합동)

- [ ] `compute_nav_output(image, plan_state, graph) -> NavOutput` 구현  
- [ ] 실제 이미지 + 지시 문장으로 end-to-end 테스트

### 📱 AR 팀

- [ ] NavOutput JSON 샘플로 UI 목업 구현  
- [ ] `text_ko` + `direction_type/angle_deg` + `route_summary`를  
      AR.js / A-Frame 상에 오버레이
