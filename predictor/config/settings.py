"""Settings loading.

Defaults live in ``default.toml`` (the single source of default values). User overrides
are stored as JSON in ``<data_dir>/settings.json`` and per-course overrides in the
course row. Overrides are deep-merged and type-checked against the defaults, so a typo
in a key or a wrong value type is rejected instead of silently ignored.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULTS_PATH = Path(__file__).with_name("default.toml")
ENV_DATA_DIR = "PREDICTOR_DATA_DIR"


class SettingsError(ValueError):
    """Raised when an override does not match the default schema."""


def _load_defaults() -> dict[str, Any]:
    with DEFAULTS_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _type_compatible(default: Any, value: Any) -> bool:
    if default is None:
        return True
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, str):
        # Some string defaults (like top_k = "auto") also accept integers.
        return isinstance(value, (str, int)) and not isinstance(value, bool)
    if isinstance(default, list):
        return isinstance(value, list)
    if isinstance(default, dict):
        return isinstance(value, dict)
    return isinstance(value, type(default))


def deep_merge(base: dict[str, Any], overrides: Mapping[str, Any], *, path: str = "",
               strict: bool = True) -> dict[str, Any]:
    """Return ``base`` updated with ``overrides`` (recursively), validating keys and types."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        dotted = f"{path}.{key}" if path else key
        if key not in out:
            if strict:
                raise SettingsError(f"Unknown setting '{dotted}'")
            out[key] = copy.deepcopy(value)
            continue
        current = out[key]
        if isinstance(current, dict):
            if not isinstance(value, Mapping):
                raise SettingsError(f"Setting '{dotted}' must be a table/object")
            # Threshold tables keyed by backend name accept new keys.
            out[key] = deep_merge(current, value, path=dotted, strict=strict)
        else:
            if not _type_compatible(current, value):
                raise SettingsError(
                    f"Setting '{dotted}' expects {type(current).__name__}, got {type(value).__name__}"
                )
            out[key] = copy.deepcopy(value)
    return out


class Settings(Mapping[str, Any]):
    """Read-only nested settings with attribute access (``settings.ocr.dpi``)."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Settings(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - guard
        raise AttributeError("Settings are read-only; use merged() to derive new settings")

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Settings(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Settings(node) if isinstance(node, dict) else node

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged(self, overrides: Mapping[str, Any] | None) -> "Settings":
        if not overrides:
            return self
        return Settings(deep_merge(self._data, overrides))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Settings({list(self._data)})"


def default_data_dir() -> Path:
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser()
    return Path.home() / "ExamPredictorData"


def user_settings_path(data_dir: Path) -> Path:
    return data_dir / "settings.json"


def load_settings(data_dir: Path | str | None = None,
                  overrides: Mapping[str, Any] | None = None) -> Settings:
    """Load defaults, then ``<data_dir>/settings.json``, then explicit ``overrides``."""
    data = _load_defaults()
    resolved_dir = Path(data_dir).expanduser() if data_dir else None
    if resolved_dir is None:
        configured = data["app"].get("data_dir") or ""
        resolved_dir = Path(configured).expanduser() if configured else default_data_dir()
    user_file = user_settings_path(resolved_dir)
    if user_file.exists():
        try:
            user_overrides = json.loads(user_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SettingsError(f"{user_file} is not valid JSON: {exc}") from exc
        data = deep_merge(data, user_overrides)
    if overrides:
        data = deep_merge(data, overrides)
    data["app"]["data_dir"] = str(resolved_dir)
    return Settings(data)


def save_user_settings(data_dir: Path, overrides: Mapping[str, Any]) -> None:
    """Validate and persist user overrides (only the keys that differ are stored)."""
    deep_merge(_load_defaults(), overrides)  # validation only
    data_dir.mkdir(parents=True, exist_ok=True)
    user_settings_path(data_dir).write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def read_user_settings(data_dir: Path) -> dict[str, Any]:
    path = user_settings_path(data_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def defaults() -> Settings:
    return Settings(_load_defaults())
