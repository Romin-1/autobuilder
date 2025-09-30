"""Helpers for dynamically importing PlugNPlay modules by file path."""

from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
from typing import Any, Type, TypeVar

_T = TypeVar("_T")


def load_class(module_path: str | Path, class_name: str) -> Type[_T]:
    """Load ``class_name`` from the python file located at ``module_path``.

    Parameters
    ----------
    module_path:
        Path to the target ``.py`` file.
    class_name:
        Name of the class defined inside ``module_path``.

    Returns
    -------
    Type[_T]
        The class object loaded from the module.

    Raises
    ------
    FileNotFoundError
        If ``module_path`` does not exist.
    AttributeError
        If the module does not define ``class_name``.
    """

    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find module file: {path}")

    module_name = f"pnp_module_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert isinstance(loader, importlib.abc.Loader)
    loader.exec_module(module)  # type: ignore[arg-type]

    try:
        return getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - defensive path
        raise AttributeError(f"Module {path} does not define {class_name}") from exc


def load_instance(module_path: str | Path, class_name: str, *args: Any, **kwargs: Any) -> _T:
    """Instantiate ``class_name`` from ``module_path`` with provided arguments."""

    klass = load_class(module_path, class_name)
    return klass(*args, **kwargs)
