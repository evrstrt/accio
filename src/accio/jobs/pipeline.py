"""The pipeline end to end: one .insv walk -> deduplicated label candidates.

run_walk() writes out/<walk>/pano/, out/<walk>/faces/,
out/<walk>/manifest.csv with a kept flag plus anchor/cosine for every
absorbed face, and out/<walk>/embeddings.npz with one row per manifest
row. The ingest job calls this; there is no other entry point.
"""

import csv
from itertools import batched
from pathlib import Path

import cv2
import numpy as np

from ..core import blur, extract, faces
from ..core.dedup import greedy_dedup
from ..core.embed import Embedder
from ..core.params import PipelineParams

EMBED_CHUNK = 64  # faces held in memory at once on the way to the embedder


def face_name(pano_idx: int, yaw: int) -> str:
    return f"y{yaw:03d}_{pano_idx:05d}.jpg"


def run_walk(video: Path, out_root: Path, params: PipelineParams,
             embedder: Embedder, progress=print) -> dict:
    walk = video.stem
    out = out_root / walk
    progress(f"stitching at {params.extract.fps:g} fps")
    panos = extract.stitch(video, out / "pano", params.extract)

    progress(f"blur-gating {len(panos)} panos")
    scores = [blur.score_pano(cv2.imread(str(p.path)), params.gate) for p in panos]
    sharp = [p for p, keep in zip(panos, blur.windowed_keep(scores, params.gate.window))
             if keep]

    progress(f"rendering {len(sharp)} x {len(params.faces.yaws)} faces")
    faces_dir = out / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    records = []  # (pano, yaw, path) in capture order
    for p in sharp:
        pano_img = cv2.imread(str(p.path))
        for yaw, img in faces.render_faces(pano_img, params.faces).items():
            path = faces_dir / face_name(p.index, yaw)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append((p, yaw, path))

    progress(f"embedding {len(records)} faces")
    embeddings = np.concatenate([
        embedder.embed([cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                        for _, _, path in chunk])
        for chunk in batched(records, EMBED_CHUNK)
    ])
    # embeddings outlive the walk: registry dedup against re-walks, leak checks
    # and future selection all reuse them. Cosines are only comparable between
    # walks embedded by the same model, so the model name travels with the rows.
    np.savez(out / "embeddings.npz",
             embeddings=embeddings, model=params.embed.model_name)
    result = greedy_dedup(embeddings, params.dedup)

    kept = set(result.kept)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pano_idx", "t_sec", "yaw", "path", "kept", "anchor", "cosine"])
        for i, (p, yaw, path) in enumerate(records):
            anchor, cos = ("", "")
            if i in result.anchor_of:
                a, c = result.anchor_of[i]
                anchor, cos = str(a), f"{c:.4f}"
            w.writerow([p.index, f"{p.t_sec:.1f}", yaw, path.name,
                        int(i in kept), anchor, cos])

    progress(f"{len(panos)} panos -> {len(sharp)} sharp -> "
             f"{len(records)} faces -> {len(kept)} kept")
    return dict(walk=walk, panos=len(panos), sharp=len(sharp),
                faces=len(records), kept=len(kept))
