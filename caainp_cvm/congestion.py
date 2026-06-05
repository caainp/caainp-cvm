"""
COEX corridor congestion analysis via person detection + walkable ROI density.

Uses YOLO object detection (COCO class "person") when ultralytics is installed.
A trapezoid ROI approximates the walkable corridor floor in typical corridor POV images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-contrib-python is required for congestion analysis") from exc


class CongestionLevel(str, Enum):
    EMPTY = "EMPTY"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


# Normalized (x, y) polygon — floor / corridor band for typical forward-facing views.
DEFAULT_CORRIDOR_ROI: Tuple[Tuple[float, float], ...] = (
    (0.05, 0.42),
    (0.95, 0.42),
    (0.98, 0.98),
    (0.02, 0.98),
)

COCO_PERSON_CLASS_ID = 0


@dataclass
class CongestionThresholds:
    """Tune per venue after calibration on reference + live frames."""

    low_max_persons: int = 2
    moderate_max_persons: int = 6
    high_max_persons: int = 12
    low_max_occupancy: float = 0.06
    moderate_max_occupancy: float = 0.14
    high_max_occupancy: float = 0.28
    min_detection_confidence: float = 0.35
    min_box_area_ratio: float = 0.00015  # ignore tiny false positives


@dataclass
class PersonDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    in_roi: bool
    center_x: float
    center_y: float


@dataclass
class CongestionResult:
    image_path: str
    image_width: int
    image_height: int
    person_count_total: int
    person_count_in_roi: int
    occupancy_ratio: float
    density_per_megapixel: float
    level: CongestionLevel
    level_score: float
    detections: List[PersonDetection] = field(default_factory=list)
    node_id: Optional[int] = None
    view_id: Optional[str] = None
    view_role: Optional[str] = None
    direction_to: Optional[str] = None
    model_name: str = ""
    roi_polygon_norm: Tuple[Tuple[float, float], ...] = DEFAULT_CORRIDOR_ROI
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level.value
        payload["detections"] = [asdict(d) for d in self.detections]
        payload["roi_polygon_norm"] = [list(p) for p in self.roi_polygon_norm]
        return payload


def _norm_polygon_to_pixels(
    polygon_norm: Sequence[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    pts = np.array(
        [[int(x * width), int(y * height)] for x, y in polygon_norm],
        dtype=np.int32,
    )
    return pts.reshape((-1, 1, 2))


def _roi_mask(
    shape: Tuple[int, int],
    polygon_norm: Sequence[Tuple[float, float]],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = _norm_polygon_to_pixels(polygon_norm, w, h)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _point_in_polygon(px: float, py: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def classify_congestion(
    *,
    person_count_in_roi: int,
    occupancy_ratio: float,
    thresholds: CongestionThresholds,
) -> Tuple[CongestionLevel, float]:
    """
    Map counts + ROI occupancy to a discrete level and a continuous score in [0, 1].
    """
    if person_count_in_roi <= 0 and occupancy_ratio < thresholds.low_max_occupancy * 0.5:
        return CongestionLevel.EMPTY, 0.0

    occ = float(np.clip(occupancy_ratio, 0.0, 1.0))
    n = int(person_count_in_roi)

    if n <= thresholds.low_max_persons and occ <= thresholds.low_max_occupancy:
        return CongestionLevel.LOW, max(occ / thresholds.low_max_occupancy, n / max(thresholds.low_max_persons, 1)) * 0.25
    if n <= thresholds.moderate_max_persons and occ <= thresholds.moderate_max_occupancy:
        return CongestionLevel.MODERATE, 0.25 + 0.25 * min(1.0, n / thresholds.moderate_max_persons)
    if n <= thresholds.high_max_persons and occ <= thresholds.high_max_occupancy:
        return CongestionLevel.HIGH, 0.55 + 0.2 * min(1.0, occ / thresholds.high_max_occupancy)
    return CongestionLevel.SEVERE, min(1.0, 0.75 + 0.25 * occ)


class PersonDetector:
    """Lazy-loaded YOLO person detector (ultralytics)."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "auto",
        thresholds: Optional[CongestionThresholds] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.thresholds = thresholds or CongestionThresholds()
        self._model: Any = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for person detection. "
                "Install with: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.model_name)
        return self._model

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        roi_polygon_norm: Sequence[Tuple[float, float]] = DEFAULT_CORRIDOR_ROI,
    ) -> List[PersonDetection]:
        model = self._load_model()
        device = self._resolve_device()
        h, w = image_bgr.shape[:2]
        roi_mask = _roi_mask((h, w), roi_polygon_norm)
        roi_area = max(int(np.count_nonzero(roi_mask)), 1)
        image_area = w * h

        results = model.predict(
            image_bgr,
            classes=[COCO_PERSON_CLASS_ID],
            conf=self.thresholds.min_detection_confidence,
            verbose=False,
            device=device,
        )
        detections: List[PersonDetection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            box_area = max(0, x2i - x1i) * max(0, y2i - y1i)
            if box_area < image_area * self.thresholds.min_box_area_ratio:
                continue

            cx = (x1 + x2) / 2.0 / w
            cy = (y2 + y1) / 2.0 / h
            in_roi = _point_in_polygon(cx, cy, roi_polygon_norm)

            # Refine: fraction of box overlapping ROI (segmentation-lite occupancy)
            box_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(box_mask, (x1i, y1i), (x2i, y2i), 255, -1)
            overlap = cv2.bitwise_and(box_mask, roi_mask)
            overlap_ratio = float(np.count_nonzero(overlap)) / float(box_area + 1e-6)
            if overlap_ratio < 0.15:
                in_roi = False

            detections.append(
                PersonDetection(
                    x1=x1i,
                    y1=y1i,
                    x2=x2i,
                    y2=y2i,
                    confidence=float(conf),
                    in_roi=in_roi,
                    center_x=float(cx),
                    center_y=float(cy),
                )
            )
        return detections


def analyze_congestion(
    image_path: str | Path,
    *,
    detector: Optional[PersonDetector] = None,
    roi_polygon_norm: Sequence[Tuple[float, float]] = DEFAULT_CORRIDOR_ROI,
    thresholds: Optional[CongestionThresholds] = None,
    node_id: Optional[int] = None,
    view_id: Optional[str] = None,
    view_role: Optional[str] = None,
    direction_to: Optional[str] = None,
    save_viz_path: Optional[str | Path] = None,
) -> CongestionResult:
    path = Path(image_path).resolve()
    thresholds = thresholds or CongestionThresholds()
    detector = detector or PersonDetector(thresholds=thresholds)

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = image_bgr.shape[:2]
    detections = detector.detect(image_bgr, roi_polygon_norm=roi_polygon_norm)
    in_roi = [d for d in detections if d.in_roi]

    roi_mask = _roi_mask((h, w), roi_polygon_norm)
    roi_area = max(int(np.count_nonzero(roi_mask)), 1)
    occupied = np.zeros((h, w), dtype=np.uint8)
    for d in in_roi:
        cv2.rectangle(occupied, (d.x1, d.y1), (d.x2, d.y2), 255, -1)
    occupied = cv2.bitwise_and(occupied, roi_mask)
    occupancy_ratio = float(np.count_nonzero(occupied)) / float(roi_area)

    person_count_in_roi = len(in_roi)
    person_count_total = len(detections)
    density_per_megapixel = person_count_in_roi / max((w * h) / 1_000_000.0, 1e-6)

    level, level_score = classify_congestion(
        person_count_in_roi=person_count_in_roi,
        occupancy_ratio=occupancy_ratio,
        thresholds=thresholds,
    )

    result = CongestionResult(
        image_path=str(path),
        image_width=w,
        image_height=h,
        person_count_total=person_count_total,
        person_count_in_roi=person_count_in_roi,
        occupancy_ratio=occupancy_ratio,
        density_per_megapixel=density_per_megapixel,
        level=level,
        level_score=float(level_score),
        detections=detections,
        node_id=node_id,
        view_id=view_id,
        view_role=view_role,
        direction_to=direction_to or None,
        model_name=detector.model_name,
        roi_polygon_norm=tuple(roi_polygon_norm),
        debug={
            "roi_area_pixels": roi_area,
            "occupied_pixels": int(np.count_nonzero(occupied)),
        },
    )

    if save_viz_path is not None:
        render_congestion_viz(image_bgr, result, save_viz_path)

    return result


def render_congestion_viz(
    image_bgr: np.ndarray,
    result: CongestionResult,
    out_path: str | Path,
) -> Path:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    pts = _norm_polygon_to_pixels(result.roi_polygon_norm, w, h)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], (80, 200, 80))
    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
    cv2.polylines(vis, [pts], True, (80, 220, 80), 2)

    for d in result.detections:
        color = (0, 220, 0) if d.in_roi else (160, 160, 160)
        cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.confidence:.2f}"
        cv2.putText(
            vis,
            label,
            (d.x1, max(d.y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    header = (
        f"{result.level.value} | in_roi={result.person_count_in_roi} "
        f"occ={result.occupancy_ratio:.2%}"
    )
    if result.node_id is not None:
        header += f" | node={result.node_id}"
    cv2.rectangle(vis, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(vis, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    return out


def suggest_blocked_edges(
    current_node: int,
    result: CongestionResult,
    *,
    block_from_level: CongestionLevel = CongestionLevel.HIGH,
) -> List[Tuple[int, int]]:
    """
    If congestion is high+, suggest blocking the edge toward direction_to (if known).
    Compatible with value_map.build_value_map_v2(blocked_edges=...).
    """
    order = [
        CongestionLevel.EMPTY,
        CongestionLevel.LOW,
        CongestionLevel.MODERATE,
        CongestionLevel.HIGH,
        CongestionLevel.SEVERE,
    ]
    if order.index(result.level) < order.index(block_from_level):
        return []
    if not result.direction_to:
        return []
    try:
        target = int(result.direction_to)
    except ValueError:
        return []
    return [(int(current_node), target)]


def is_passage_view(view_role: Optional[str]) -> bool:
    if not view_role:
        return False
    return str(view_role).strip().lower() in {
        "corridor",
        "connector",
        "context",
        "foyer",
    }
