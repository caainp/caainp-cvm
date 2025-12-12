# 프로젝트 상세 분석 보고서

## 📋 프로젝트 개요

**프로젝트명**: CA-Nav (Constraint-Aware AR Navigation)  
**목표**: AI공학관 4층을 대상으로 한 실내 AR 내비게이션 시스템  
**현재 상태**: CVM 모듈 구현 완료, CSM/NAV/AR 모듈 미구현

---

## 🏗️ 프로젝트 아키텍처

### 전체 시스템 구조

```
┌─────────────┐
│   사용자    │
│  (카메라)   │
└──────┬──────┘
       │ 이미지
       ▼
┌─────────────────────────────────────┐
│         CVM (Camera/Vision)         │
│  - 이미지 → 현재 위치 노드 추정      │
│  - Value Map 생성                    │
└──────┬──────────────────────────────┘
       │ CvmResult
       ▼
┌─────────────────────────────────────┐
│    CSM (Command/Constraint)         │
│  - 자연어 지시 → 단계별 플랜         │
│  - 제약조건 처리                     │
│  - 경로 계획                         │
└──────┬──────────────────────────────┘
       │ PlanState
       ▼
┌─────────────────────────────────────┐
│         NAV (Navigation)            │
│  - CVM + CSM 통합                   │
│  - NavOutput JSON 생성               │
└──────┬──────────────────────────────┘
       │ NavOutput JSON
       ▼
┌─────────────────────────────────────┐
│         AR Interface                │
│  - 화살표 + 안내 문구 렌더링         │
└─────────────────────────────────────┘
```

---

## ✅ 현재 구현 상태

### 1. CVM (Camera/Vision Module) - ✅ 완료

#### 구현된 기능

**1.1 지도 로딩 (`scripts/map_loader.py`)**
- CSV 파일에서 노드 정보 및 CLIP 임베딩 로드
- NetworkX 그래프 구조 생성
- 노드 메타데이터 파싱 (floor, description, neighbors, type, coordinates 등)
- 임베딩 벡터 정규화 및 매트릭스 구성

**주요 함수:**
```python
load_map_csv(csv_path) -> (graph, node_records, emb_matrix, node_ids)
```

**1.2 이미지 위치 추정 (`scripts/localize_image.py`)**
- **CLIP 기반 유사도 계산**: OpenCLIP 모델을 사용한 이미지 임베딩
- **OCR 통합**: EasyOCR을 통한 방호수 인식 및 보정
- **기하학적 검증**: SIFT 특징점 매칭을 통한 정확도 향상
- **그래프 사전 확률**: 이전 노드 정보를 활용한 시간적 일관성
- **다중 신호 융합**: CLIP, OCR, 기하학적 검증, 그래프 사전 확률의 가중 결합

**핵심 알고리즘:**
```python
localize_image(
    image_path, csv_path,
    use_ocr=True,      # OCR 활성화
    use_geo=True,      # 기하학적 검증
    prev_node=None,    # 이전 노드 (시간적 일관성)
    w_clip=1.0,        # CLIP 가중치
    w_ocr=0.3,         # OCR 가중치
    w_geo=0.4,         # 기하학적 검증 가중치
    w_prior=0.2,       # 그래프 사전 확률 가중치
    w_id=0.6,          # OCR-노드ID 직접 매칭 가중치
    w_consistency=0.4  # 이웃 노드 일관성 가중치
) -> {
    "current_node": int,
    "confidence": float,
    "candidates": [...]
}
```

**1.3 OCR 기능 상세**
- **ROI 기반 텍스트 검출**: 형태학적 기법으로 텍스트 영역 자동 검출
- **전처리 옵션**: CLAHE 대비 향상, 언샤프 마스킹, 적응형 임계값
- **TTA (Test-Time Augmentation)**: 회전 및 스케일 변환으로 정확도 향상
- **방호수 추출**: 정규화 및 패턴 매칭으로 방호수 자동 인식

**1.4 평가 시스템 (`scripts/evaluate_cvm.py`)**
- 대량 이미지에 대한 자동 평가
- Top-1 및 Top-K 정확도 계산
- 노드별 성능 분석
- 실패 케이스 추적

#### 데이터 구조

**CSV 파일 구조** (`ai_4f_node_map_fixed_embeded.csv`):
- `node_id`: 노드 고유 ID
- `floor`: 층 정보
- `description`: 노드 설명 (예: "401호 강의실")
- `neighbors`: 인접 노드 리스트 (세미콜론 구분)
- `type`: 노드 타입 (ROOM, HALLWAY 등)
- `room_range`: 방 범위 (예: "4101-4110")
- `anchor_room`: 기준 방호수
- `clip_embedding`: CLIP 임베딩 벡터 (JSON 배열)

**노드 이미지 구조** (`node_images/node_images/`):
- 파일명 패턴: `{node_id}.jpg`, `{node_id}({variant}).jpg`
- 예: `401.jpg`, `401(1).jpg`, `401(2).jpg`

---

## 📊 현재 구현 상세 분석

### CVM 모듈의 핵심 알고리즘

#### 1. 다중 신호 융합 (Multi-Modal Fusion)

```python
# 1. CLIP 유사도 계산
sims = cosine_sim_matrix(query_embedding, gallery_embeddings)

# 2. OCR 힌트 추출
ocr_nums = ocr_hints_with_roi(image_path, ...)

# 3. 기하학적 검증
geo_score = geometric_verification_score(query, reference_images)

# 4. 그래프 사전 확률
prior_score = graph_prior_score(graph, prev_node, candidate_node)

# 5. 이웃 일관성 보너스
consistency_score = neighbor_consistency_bonus(...)

# 6. 가중 결합
combined_score = (
    w_clip * clip_score +
    w_ocr * ocr_score +
    w_id * id_match_score +
    w_geo * geo_score +
    w_prior * prior_score +
    w_consistency * consistency_score
)
```

#### 2. OCR 처리 파이프라인

```
이미지 입력
    ↓
ROI 검출 (형태학적 기법)
    ↓
전처리 (CLAHE, 언샤프, 적응형 임계값)
    ↓
EasyOCR 텍스트 인식
    ↓
TTA (회전/스케일 변환)
    ↓
방호수 추출 및 정규화
    ↓
노드 ID 매칭
```

#### 3. 기하학적 검증

- SIFT 특징점 검출
- BFMatcher를 통한 특징점 매칭
- RANSAC 호모그래피 추정
- 인라이어 비율 계산

---

## 🎯 개발 방법론

### 현재 개발 방식

1. **모듈화된 구조**: 각 기능이 독립적인 스크립트로 구현
2. **명령줄 인터페이스**: argparse를 통한 유연한 파라미터 조정
3. **평가 중심**: `evaluate_cvm.py`를 통한 체계적인 성능 측정
4. **로깅**: loguru를 통한 상세한 디버깅 정보

### 실행 예시

```bash
# 단일 이미지 위치 추정
python -m scripts.localize_image \
  --image "node_images/node_images/401(1).jpg" \
  --csv "ai_4f_node_map_fixed_embeded.csv" \
  --model "ViT-L-14" \
  --pretrained "laion2b_s32b_b82k" \
  --device cpu \
  --topk 5 \
  --use_ocr \
  --use_geo \
  --node_images_dir "node_images/node_images" \
  --prev_node 427 \
  --w_geo 0.7 \
  --w_prior 0.3

# 대량 평가
python -m scripts.evaluate_cvm \
  --images_dir "node_images/node_images" \
  --csv "ai_4f_node_map_fixed_embeded.csv" \
  --use_ocr \
  --use_geo \
  --device auto
```

---

## 📈 성능 최적화 기법

### 1. 임베딩 차원 자동 매칭
- CSV의 임베딩 차원을 분석하여 적절한 CLIP 모델 자동 선택
- 모델 불일치 방지

### 2. 후보 풀 확장
- 초기 CLIP Top-K 후보를 10배 확장 (예: topk=5 → pool_size=50)
- OCR 힌트로 누락된 노드 강제 주입
- 재랭킹을 통한 정확도 향상

### 3. 정규화 및 스케일링
- 모든 보조 점수를 0~1 범위로 정규화
- 동적 가중치 조정 (OCR이 없으면 OCR 가중치 0)

### 4. 온도 스케일링
- Softmax에 온도 파라미터 적용
- Confidence 계산의 안정성 향상

---
