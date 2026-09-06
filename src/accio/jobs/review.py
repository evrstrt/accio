"""Manifest rows plus the decisions log -> the groups the review UI shows."""

from pathlib import Path
from typing import Callable

from ..core.export import resolve_pick
from .pipeline import top_classes


def dropped_count(anchors: set[str], state: dict[str, dict]) -> int:
    """Groups a reviewer dropped, among the anchors that currently exist."""
    return sum(1 for a in anchors if state.get(a, {}).get("dropped"))


def segmentation(seg: dict, rows: list[dict], face_url: Callable[[str], str],
                 mask_url: Callable[[str], str]) -> dict:
    """The segmentation record with one frame per masked face, in walk order."""
    t_of = {r["path"]: (float(r["t_sec"]), int(r["yaw"])) for r in rows}
    classes = seg.get("classes", {})
    frames = [{
        "face": name,
        "tSec": t_of.get(name, (0.0, 0))[0],
        "yaw": t_of.get(name, (0.0, 0))[1],
        "url": face_url(name),
        "mask": mask_url(f"{Path(name).stem}.png"),
        "classes": shares,
    } for name, shares in classes.items()]
    frames.sort(key=lambda f: (f["tSec"], f["yaw"]))
    return {"model": seg.get("model", ""), "frames": frames,
            "labels": seg.get("labels", {}),  # mask index -> class name
            "classMix": top_classes(classes)}


def groups(rows: list[dict], state: dict[str, dict],
           url: Callable[[str], str]) -> tuple[list[dict], list[dict]]:
    """(faces in manifest order, one group per kept anchor).

    Decisions are stored by face name; the API speaks manifest indices.
    """
    faces = [{
        "idx": i,
        "panoIdx": int(r["pano_idx"]),
        "tSec": float(r["t_sec"]),
        "yaw": int(r["yaw"]),
        "url": url(r["path"]),
        "kept": r["kept"] == "1",
        "anchor": int(r["anchor"]) if r["anchor"] else None,
        "cosine": float(r["cosine"]) if r["cosine"] else None,
        "sharpness": float(r["sharpness"]),
    } for i, r in enumerate(rows)]
    idx_of = {r["path"]: i for i, r in enumerate(rows)}
    out = []
    for f in faces:
        if not f["kept"]:
            continue
        s = state.get(rows[f["idx"]]["path"], {"pick": None, "dropped": False})
        # a pick holds only while the picked face is still in this group
        out.append({
            "anchor": f,
            "members": sorted((m for m in faces if m["anchor"] == f["idx"]),
                              key=lambda m: -m["cosine"]),
            "auto": f["idx"],
            "pick": resolve_pick(rows, idx_of, f["idx"], s["pick"]),
            "dropped": s["dropped"],
        })
    return faces, out
