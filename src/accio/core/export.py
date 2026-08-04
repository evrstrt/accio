"""Step 3: reviewed picks -> a self-contained zip for annotation.

The app is a processing tool, not a store: everything the dataset needs
travels in the zip. Each frame carries the walk metadata and the exact
pipeline settings in its EXIF UserComment (JSON, no recompression), the
manifest repeats them per row, and walk.json holds them once. Filenames
are deterministic ({site}_{building}_{walk}_t012.5_y090.jpg) and zip
entries use a fixed timestamp, so the same review always exports
byte-identical archives.
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

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed entry mtime: determinism over honesty


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def name_prefix(walk_id: str, meta: dict) -> str:
    parts = [slug(meta.get("site", "")), slug(meta.get("building", "")), walk_id]
    return "_".join(p for p in parts if p)


def export_name(prefix: str, t_sec: float, yaw: int) -> str:
    return f"{prefix}_t{t_sec:06.1f}_y{yaw:03d}.jpg"


def effective_picks(rows: list[dict], state: dict[str, dict]) -> list[tuple[int, dict]]:
    """(pick_idx, manifest_row) per surviving group, in walk order.

    A group survives unless dropped in review; its exported frame is the
    reviewer's pick, defaulting to the dedup anchor. Review state is keyed by
    face name, so it still resolves after a re-run renumbers the manifest.
    """
    idx_of = {r["path"]: i for i, r in enumerate(rows)}
    picks = []
    for i, r in enumerate(rows):
        if r["kept"] != "1":
            continue
        s = state.get(r["path"], {"pick": None, "dropped": False})
        if s["dropped"]:
            continue
        pick = idx_of.get(s["pick"], i) if s["pick"] else i
        picks.append((pick, rows[pick]))
    return picks


def stamp_exif(jpg: bytes, payload: dict) -> bytes:
    """Insert payload as EXIF UserComment JSON without recompressing pixels."""
    exif = piexif.dump({"Exif": {
        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(json.dumps(payload))
    }})
    out = io.BytesIO()
    piexif.insert(exif, jpg, out)
    return out.getvalue()


def build_zip(walk_id: str, walk_dir: Path, rows: list[dict],
              state: dict[int, dict], meta: dict, embed_model: str,
              params: PipelineParams) -> bytes:
    prefix = name_prefix(walk_id, meta)
    pipeline = dict(asdict(params), embed_model_used=embed_model)
    common = {"walk": walk_id, **meta, "pipeline": pipeline}

    buf = io.BytesIO()
    manifest = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:

        def write(name: str, data: bytes | str):
            z.writestr(zipfile.ZipInfo(name, date_time=ZIP_EPOCH), data)

        for pick_idx, r in effective_picks(rows, state):
            t_sec, yaw = float(r["t_sec"]), int(r["yaw"])
            name = export_name(prefix, t_sec, yaw)
            payload = {**common, "t_sec": t_sec, "yaw": yaw,
                       "pano_idx": int(r["pano_idx"]), "source_face": r["path"]}
            jpg = (walk_dir / "faces" / r["path"]).read_bytes()
            write(f"frames/{name}", stamp_exif(jpg, payload))
            manifest.append({
                "file": f"frames/{name}", "walk": walk_id,
                "site": meta.get("site", ""), "building": meta.get("building", ""),
                "stage": meta.get("stage", ""), "shot_date": meta.get("shotDate", ""),
                "t_sec": t_sec, "yaw": yaw, "pano_idx": int(r["pano_idx"]),
                "source_face": r["path"], "pick_idx": pick_idx,
            })

        s = io.StringIO()
        w = csv.DictWriter(s, fieldnames=list(manifest[0].keys()) if manifest else
                           ["file", "walk"])
        w.writeheader()
        w.writerows(manifest)
        write("manifest.csv", s.getvalue())
        write("walk.json", json.dumps({**common, "frames": len(manifest)}, indent=1))
    return buf.getvalue()
