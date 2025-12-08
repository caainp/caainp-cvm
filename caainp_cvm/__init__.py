from importlib import resources
from pathlib import Path
from .nav_engine import compute_nav_output

CVM_ROOT = Path(__file__).resolve().parent

def _get_csv_path() -> Path:
    """CSV 파일 경로를 반환"""
    try:
        return Path(resources.files("caainp_cvm") / "data" / "ai_4f_node_map_fixed_embeded.csv")
    except (ModuleNotFoundError, TypeError):
        return CVM_ROOT / "ai_4f_node_map_fixed_embeded.csv"

def _get_node_images_dir() -> Path:
    """노드 이미지 디렉토리 경로를 반환"""
    try:
        return Path(resources.files("caainp_cvm") / "data" / "node_images")
    except (ModuleNotFoundError, TypeError):
        return CVM_ROOT / "node_images" / "node_images"


__all__ = ["compute_nav_output", "_get_csv_path", "_get_node_images_dir"]
