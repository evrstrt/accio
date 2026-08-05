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


def effective_picks(rows: list[dict],
                    state: dict[str, dict]) -> list[tuple[int, int, dict]]:
    """(anchor_idx, pick_idx, manifest_row) per surviving group, in walk order.

    A group survives unless dropped in review; its exported frame is the
    reviewer's pick, defaulting to the dedup anchor. Review state is keyed by
    face name, so it still resolves after a re-run renumbers the manifest. The
    anchor comes back too, so the export can say which frames a human moved.
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
        picks.append((i, pick, rows[pick]))
    return picks


def review_counts(rows: list[dict], state: dict[str, dict]) -> tuple[int, int]:
    """(dropped, overridden) among the groups that currently exist.

    A re-run can change which faces are anchors. Decisions on faces that are no
    longer anchors are kept in the log but do not count against this run, or
    the funnel would report drops the export cannot show.
    """
    anchors = {r["path"] for r in rows if r["kept"] == "1"}
    dropped = overridden = 0
    for anchor, s in state.items():
        if anchor not in anchors:
            continue
        if s["dropped"]:
            dropped += 1
        elif s["pick"] and s["pick"] != anchor:
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


def build_zip(walk_id: str, walk_dir: Path, rows: list[dict],
              state: dict[str, dict], meta: dict, embed_model: str,
              params: PipelineParams, decisions: list[dict] | None = None,
              segmentation: dict | None = None) -> bytes:
    """The zip is the deliverable, so everything the dataset needs is in it:
    the frames, what produced them, and what a human changed by hand."""
    prefix = name_prefix(walk_id, meta)
    pipeline = dict(asdict(params), embed_model_used=embed_model)
    common = {"walk": walk_id, **meta, "pipeline": pipeline}
    # a pre-annotation, if the walk has one: the mask travels beside its frame
    # under the same name, and the class mix goes in the manifest row
    classes = (segmentation or {}).get("classes", {})

    buf = io.BytesIO()
    manifest = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:

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
                # the one column a pipeline hash cannot reproduce
                "overridden": int(overridden),
                # "wall 0.58; ceiling 0.28; floor 0.10", largest first
                "classes": "; ".join(f"{k} {v:.2f}" for k, v in
                                     list(classes.get(r["path"], {}).items())[:6]),
            })

        s = io.StringIO()
        w = csv.DictWriter(s, fieldnames=list(manifest[0].keys()) if manifest else
                           ["file", "walk"])
        w.writeheader()
        w.writerows(manifest)
        write("manifest.csv", s.getvalue())

        # Review is the only stage a re-run cannot reproduce, so its log leaves
        # with the frames: a pattern in the overrides is evidence the auto-pick
        # rule needs changing, and that evidence is worth nothing here.
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
    return buf.getvalue()
