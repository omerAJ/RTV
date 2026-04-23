from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CatalogError(RuntimeError):
    pass


@dataclass
class TieCatalogItem:
    id: str
    name: str
    asset_path: Path
    thumbnail_path: Path
    knot_anchor: tuple[float, float]
    knot_width_ref: float
    default_scale: float
    default_offset_x: float
    default_offset_y: float
    default_rotation_deg: float
    asset_bgra: np.ndarray
    top_mask: np.ndarray


@dataclass
class TieCatalog:
    path: Path
    items: list[TieCatalogItem]

    def __post_init__(self) -> None:
        self.by_id = {item.id: item for item in self.items}

    def get(self, tie_id: str | None) -> TieCatalogItem | None:
        if tie_id is None:
            return None
        return self.by_id.get(tie_id)


def load_tie_catalog(path: str | Path) -> TieCatalog:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise CatalogError(f"Catalog file not found: {catalog_path}")

    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Catalog JSON is invalid: {catalog_path}") from exc

    if not isinstance(data, list):
        raise CatalogError("Catalog root must be a JSON array")

    base_dir = catalog_path.parent
    items: list[TieCatalogItem] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(data):
        item = _load_catalog_item(base_dir, raw_entry, index)
        if item.id in seen_ids:
            raise CatalogError(f"Duplicate tie id in catalog: {item.id}")
        seen_ids.add(item.id)
        items.append(item)

    if not items:
        raise CatalogError("Catalog is empty")

    return TieCatalog(path=catalog_path, items=items)


def _load_catalog_item(base_dir: Path, raw_entry: Any, index: int) -> TieCatalogItem:
    if not isinstance(raw_entry, dict):
        raise CatalogError(f"Catalog entry #{index} must be an object")

    required = [
        "id",
        "name",
        "asset_path",
        "thumbnail_path",
        "knot_anchor",
        "knot_width_ref",
        "default_scale",
        "default_offset_x",
        "default_offset_y",
        "default_rotation_deg",
    ]
    missing = [key for key in required if key not in raw_entry]
    if missing:
        raise CatalogError(f"Catalog entry #{index} is missing fields: {', '.join(missing)}")

    tie_id = _as_string(raw_entry["id"], f"entry #{index} id")
    name = _as_string(raw_entry["name"], f"entry #{index} name")

    asset_path = (base_dir / _as_string(raw_entry["asset_path"], f"entry #{index} asset_path")).resolve()
    thumbnail_path = (base_dir / _as_string(raw_entry["thumbnail_path"], f"entry #{index} thumbnail_path")).resolve()
    if not asset_path.is_file():
        raise CatalogError(f"Tie asset not found for '{tie_id}': {asset_path}")
    if not thumbnail_path.is_file():
        raise CatalogError(f"Tie thumbnail not found for '{tie_id}': {thumbnail_path}")

    knot_anchor = _as_point(raw_entry["knot_anchor"], f"entry #{index} knot_anchor")
    knot_width_ref = _as_positive_float(raw_entry["knot_width_ref"], f"entry #{index} knot_width_ref")
    default_scale = _as_positive_float(raw_entry["default_scale"], f"entry #{index} default_scale")
    default_offset_x = _as_float(raw_entry["default_offset_x"], f"entry #{index} default_offset_x")
    default_offset_y = _as_float(raw_entry["default_offset_y"], f"entry #{index} default_offset_y")
    default_rotation_deg = _as_float(raw_entry["default_rotation_deg"], f"entry #{index} default_rotation_deg")

    asset_bgra = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    if asset_bgra is None:
        raise CatalogError(f"Failed to read tie asset for '{tie_id}': {asset_path}")
    if asset_bgra.ndim != 3 or asset_bgra.shape[2] != 4:
        raise CatalogError(f"Tie asset must be a BGRA PNG with transparency: {asset_path}")

    thumb = cv2.imread(str(thumbnail_path), cv2.IMREAD_UNCHANGED)
    if thumb is None:
        raise CatalogError(f"Failed to read tie thumbnail for '{tie_id}': {thumbnail_path}")

    height, width = asset_bgra.shape[:2]
    anchor_x, anchor_y = knot_anchor
    if not (0 <= anchor_x < width and 0 <= anchor_y < height):
        raise CatalogError(f"knot_anchor for '{tie_id}' must lie inside the asset bounds")
    top_mask = _build_top_mask(asset_bgra[:, :, 3], anchor_y, knot_width_ref)

    return TieCatalogItem(
        id=tie_id,
        name=name,
        asset_path=asset_path,
        thumbnail_path=thumbnail_path,
        knot_anchor=knot_anchor,
        knot_width_ref=knot_width_ref,
        default_scale=default_scale,
        default_offset_x=default_offset_x,
        default_offset_y=default_offset_y,
        default_rotation_deg=default_rotation_deg,
        asset_bgra=asset_bgra,
        top_mask=top_mask,
    )


def _build_top_mask(alpha_channel: np.ndarray, anchor_y: float, knot_width_ref: float) -> np.ndarray:
    cutoff = int(round(min(alpha_channel.shape[0], max(0.0, anchor_y + knot_width_ref * 1.8))))
    top_mask = np.zeros_like(alpha_channel)
    top_mask[:cutoff, :] = alpha_channel[:cutoff, :]
    return top_mask


def _as_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{label} must be a non-empty string")
    return value


def _as_point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise CatalogError(f"{label} must be a 2-item array")
    return _as_float(value[0], label), _as_float(value[1], label)


def _as_positive_float(value: Any, label: str) -> float:
    result = _as_float(value, label)
    if result <= 0:
        raise CatalogError(f"{label} must be greater than 0")
    return result


def _as_float(value: Any, label: str) -> float:
    if not isinstance(value, (float, int)):
        raise CatalogError(f"{label} must be numeric")
    return float(value)
