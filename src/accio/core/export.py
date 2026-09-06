"""Reviewed picks -> a self-contained zip for annotation.

Each frame carries the walk metadata and pipeline settings in its EXIF
UserComment (JSON, no recompression); the manifest repeats them per row and
walk.json holds them once. Filenames are deterministic
({site}_{building}_{walk}_t012.5_y090.jpg) and zip entries use a fixed
timestamp, so the same review exports byte-identical archives.
"""

import csv
import io
import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

import piexif
import piexif.helper

from .params import PipelineParams

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed entry mtime, for byte-identical archives
MANIFEST_FIELDS = ("file", "walk", "site", "building", "stage", "shot_date",
                   "t_sec", "yaw", "pano_idx", "source_face", "pick_idx",
                   "overridden", "classes")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def name_prefix(walk_id: str, meta: dict) -> str:
    parts = [slug(meta.get("site", "")), slug(meta.get("building", "")), walk_id]
    return "_".join(p for p in parts if p)


def export_name(prefix: str, t_sec: float, yaw: int) -> str:
    return f"{prefix}_t{t_sec:06.1f}_y{yaw:03d}.jpg"


def resolve_pick(rows: list[dict], idx_of: dict[str, int], anchor_idx: int,
                 pick: str | None) -> int:
    """The manifest index a stored pick means today, or the anchor.

    A reselect regroups without touching the decisions log, so a pick may now
    name a face outside the anchor's group; honouring it would export that
    frame twice under one name.
    """
    if not pick:
        return anchor_idx
    j = idx_of.get(pick)
    if j is None or (j != anchor_idx and rows[j]["anchor"] != str(anchor_idx)):
        return anchor_idx
    return j


def effective_picks(rows: list[dict],
                    state: dict[str, dict]) -> list[tuple[int, int, dict]]:
    """(anchor_idx, pick_idx, manifest_row) per surviving group, in walk order.

    Review state is keyed by face name, so it survives a re-run renumbering
    the manifest.
    """
    idx_of = {r["path"]: i for i, r in enumerate(rows)}
    picks = []
    for i, r in enumerate(rows):
        if r["kept"] != "1":
            continue
        s = state.get(r["path"], {"pick": None, "dropped": False})
        if s["dropped"]:
            continue
        pick = resolve_pick(rows, idx_of, i, s["pick"])
        picks.append((i, pick, rows[pick]))
    return picks


def review_counts(rows: list[dict], state: dict[str, dict]) -> tuple[int, int]:
    """(dropped, overridden) among the groups that currently exist.

    Decisions on faces that are no longer anchors stay in the log but are not
    counted.
    """
    idx_of = {r["path"]: i for i, r in enumerate(rows)}
    dropped = overridden = 0
    for i, r in enumerate(rows):
        if r["kept"] != "1" or r["path"] not in state:
            continue
        s = state[r["path"]]
        if s["dropped"]:
            dropped += 1
        elif resolve_pick(rows, idx_of, i, s["pick"]) != i:
            overridden += 1
    return dropped, overridden


def stamp_exif(jpg: bytes, payload: dict) -> bytes:
    """Insert payload as EXIF UserComment JSON without recompressing pixels."""
    exif = piexif.dump({"Exif": {
        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(json.dumps(payload))
    }})
    out = io.BytesIO()
    piexif.insert(exif, jpg, out)
    return out.getvalue()


def write_zip(dest, walk_id: str, walk_dir: Path, rows: list[dict],
              state: dict[str, dict], meta: dict, embed_model: str,
              params: PipelineParams, decisions: list[dict] | None = None,
              segmentation: dict | None = None) -> int:
    """Write the archive to `dest` (a path or open binary file); returns the
    frame count.

    Streaming keeps the peak at one frame: measured 37 MB for a 4.7-minute
    walk, ~235 MB for a half-hour one.
    """
    prefix = name_prefix(walk_id, meta)
    pipeline = dict(asdict(params), embed_model_used=embed_model)
    common = {"walk": walk_id, **meta, "pipeline": pipeline}
    classes = (segmentation or {}).get("classes", {})

    manifest = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:

        def write(name: str, data: bytes | str):
            z.writestr(zipfile.ZipInfo(name, date_time=ZIP_EPOCH), data)

        for anchor_idx, pick_idx, r in effective_picks(rows, state):
            t_sec, yaw = float(r["t_sec"]), int(r["yaw"])
            name = export_name(prefix, t_sec, yaw)
            overridden = pick_idx != anchor_idx
            payload = {**common, "t_sec": t_sec, "yaw": yaw,
                       "pano_idx": int(r["pano_idx"]), "source_face": r["path"],
                       "overridden": overridden}
            jpg = (walk_dir / "faces" / r["path"]).read_bytes()
            write(f"frames/{name}", stamp_exif(jpg, payload))
            mask = walk_dir / "masks" / f"{Path(r['path']).stem}.png"
            if mask.exists():
                write(f"masks/{Path(name).stem}.png", mask.read_bytes())
            manifest.append({
                "file": f"frames/{name}", "walk": walk_id,
                "site": meta.get("site", ""), "building": meta.get("building", ""),
                "stage": meta.get("stage", ""), "shot_date": meta.get("shotDate", ""),
                "t_sec": t_sec, "yaw": yaw, "pano_idx": int(r["pano_idx"]),
                "source_face": r["path"], "pick_idx": pick_idx,
                "overridden": int(overridden),
                # "wall 0.58; ceiling 0.28; floor 0.10", largest first
                "classes": "; ".join(f"{k} {v:.2f}" for k, v in
                                     list(classes.get(r["path"], {}).items())[:6]),
            })

        s = io.StringIO()
        w = csv.DictWriter(s, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)
        write("manifest.csv", s.getvalue())

        # review is the one stage a re-run cannot reproduce, so its log ships
        dropped, overridden = review_counts(rows, state)
        write("decisions.json", json.dumps({
            "walk": walk_id, "dropped": dropped, "overridden": overridden,
            "log": [{k: v for k, v in d.items() if k != "walkId"}
                    for d in (decisions or [])],
        }, indent=1))
        if segmentation:
            write("segmentation.json", json.dumps(segmentation, indent=1))
        write("walk.json", json.dumps({**common, "frames": len(manifest),
                                       "dropped": dropped,
                                       "overridden": overridden,
                                       "segmented": bool(classes)}, indent=1))
    return len(manifest)
