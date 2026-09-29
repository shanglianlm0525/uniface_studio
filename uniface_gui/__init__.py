"""Gradio application for interactive UniFace workflows."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

__all__ = ['build_app']

# Prefer the vendored upstream checkout when running from the Studio source tree.
_VENDORED_ROOT = Path(__file__).resolve().parent.parent / 'uniface'
if (_VENDORED_ROOT / 'uniface' / '__init__.py').is_file():
    vendored_path = str(_VENDORED_ROOT)
    if vendored_path in sys.path:
        sys.path.remove(vendored_path)
    sys.path.insert(0, vendored_path)
    loaded_uniface = sys.modules.get('uniface')
    loaded_file = getattr(loaded_uniface, '__file__', None)
    if loaded_uniface is not None and loaded_file is None:
        del sys.modules['uniface']


def build_app() -> Any:
    """Build the UI while keeping Gradio an optional dependency of UniFace."""
    from .app import build_app as _build_app

    return _build_app()
