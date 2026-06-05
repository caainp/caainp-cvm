from .paths import _get_csv_path, _get_node_images_dir

__all__ = ["compute_nav_output", "_get_csv_path", "_get_node_images_dir"]


def compute_nav_output(*args, **kwargs):
    """Lazy import so CLI tools (e.g. congestion) work without caainp-csm installed."""
    from .nav_engine import compute_nav_output as _impl

    return _impl(*args, **kwargs)
