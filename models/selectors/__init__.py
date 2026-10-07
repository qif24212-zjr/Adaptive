"""Frame selector registry (project-side).

SELECTOR.NAME in config picks the implementation. Future adaptive selector:
    models/selectors/adaptive_selector.py  (same BaseSelector contract)
"""
from xmodaler.utils.registry import Registry

SELECTOR_REGISTRY = Registry("SELECTOR")
SELECTOR_REGISTRY.__doc__ = """
Registry for frame selectors operating on candidate frame features.
"""


def build_selector(cfg):
    name = cfg.SELECTOR.NAME
    assert len(name) > 0, "SELECTOR.NAME must be set"
    return SELECTOR_REGISTRY.get(name)(cfg)


# import implementations so they self-register
from .picknet_style_selector import PickNetStyleSelector  # noqa: F401,E402

__all__ = ["SELECTOR_REGISTRY", "build_selector", "BaseSelector", "PickNetStyleSelector"]

from .base_selector import BaseSelector  # noqa: E402
