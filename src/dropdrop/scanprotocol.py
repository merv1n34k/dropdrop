"""Parse EVOS .scanprotocol files to reconstruct the field tiling grid.

The instrument stores the scan-area rectangle and the field traversal pattern
(e.g. SerpentineVertical) but not an explicit row/column count — that is derived
at acquisition time. We rebuild the grid from the actual field count plus the
scan-area aspect ratio, then map each field index to its (row, col) cell.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _local(tag):
    """Strip XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


def find_scanprotocol(input_dir):
    """Return the first .scanprotocol file in input_dir, or None."""
    matches = sorted(Path(input_dir).glob("*.scanprotocol"))
    return matches[0] if matches else None


def parse_scanprotocol(path):
    """Extract field pattern and scan-area aspect ratio from a .scanprotocol.

    Returns a dict with keys ``field_pattern``, ``area_w``, ``area_h``,
    ``aspect`` — or None if the file can't be parsed.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None

    field_pattern = None
    extents = None  # list of (x, y) points for the first usable Extents block

    for el in root.iter():
        name = _local(el.tag)
        if name == "FieldSequencePattern" and el.text:
            field_pattern = el.text.strip()
        elif name == "Extents" and extents is None:
            points = []
            for pt in el:
                if _local(pt.tag) != "Point":
                    continue
                x = y = None
                for coord in pt:
                    c = _local(coord.tag)
                    if c == "_x" and coord.text:
                        x = float(coord.text)
                    elif c == "_y" and coord.text:
                        y = float(coord.text)
                if x is not None and y is not None:
                    points.append((x, y))
            if len(points) >= 3:
                extents = points

    if not field_pattern or not extents:
        return None

    xs = [p[0] for p in extents]
    ys = [p[1] for p in extents]
    area_w = max(xs) - min(xs)
    area_h = max(ys) - min(ys)
    if area_w <= 0 or area_h <= 0:
        return None

    return {
        "field_pattern": field_pattern,
        "area_w": area_w,
        "area_h": area_h,
        "aspect": area_w / area_h,
    }


def compute_grid(n_fields, aspect):
    """Pick (cols, rows) covering n_fields with cols/rows closest to aspect.

    Prefers a snug grid (minimal empty cells) whose proportions match the
    scanned rectangle.
    """
    if n_fields <= 0:
        return (0, 0)

    best = None
    for rows in range(1, n_fields + 1):
        cols = math.ceil(n_fields / rows)
        empty = cols * rows - n_fields
        ratio_err = abs((cols / rows) - aspect)
        score = (ratio_err, empty)
        if best is None or score < best[0]:
            best = (score, cols, rows)
    return (best[1], best[2])


def field_cells(n_fields, cols, rows, pattern):
    """Map acquisition order (0..n_fields-1) to (row, col) grid cells.

    Honors EVOS serpentine traversal. SerpentineVertical fills column by column,
    snaking (down, then up); SerpentineHorizontal fills row by row, snaking.
    Non-serpentine patterns fall back to plain row-major.
    """
    cells = []
    vertical = "vertical" in pattern.lower()
    serpentine = "serpentine" in pattern.lower()

    if vertical:
        for i in range(n_fields):
            col = i // rows
            pos = i % rows
            row = pos if (col % 2 == 0 or not serpentine) else (rows - 1 - pos)
            cells.append((row, col))
    else:
        for i in range(n_fields):
            row = i // cols
            pos = i % cols
            col = pos if (row % 2 == 0 or not serpentine) else (cols - 1 - pos)
            cells.append((row, col))
    return cells


def build_layout(input_dir, n_fields):
    """Build a tile layout dict for n_fields using a .scanprotocol if available.

    Returns a dict with ``cols``, ``rows``, ``pattern`` and ``cells`` (a list
    of [row, col] in acquisition order), or None if no scanprotocol is found.
    """
    path = find_scanprotocol(input_dir)
    if path is None:
        return None
    info = parse_scanprotocol(path)
    if info is None:
        return None

    cols, rows = compute_grid(n_fields, info["aspect"])
    cells = field_cells(n_fields, cols, rows, info["field_pattern"])
    return {
        "cols": cols,
        "rows": rows,
        "pattern": info["field_pattern"],
        "cells": [list(c) for c in cells],
    }
