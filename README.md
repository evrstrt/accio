# accio

Frame preparation app for 360 site walks: drop an .insv, review the deduplicated
clusters, swap or drop picks, export a tracked set for annotation.

## Flow

1. **Ingest.** An .insv plus walk metadata (site, building, flat, stage, operator,
   mount height, date). The original video is copied into the app's data root
   first: it is the raw ground truth, kept so walks can be re-stitched when
   parameters improve. The pipeline runs as a background job: MediaSDK stitch
   (optflow + flowstate) with decimation, gnomonic faces, relative blur gate,
   DINOv3 embed, greedy dedup per walk. Faces, embeddings and auto-picks land
   in the store.
2. **Review.** Kept set in walk order; each duplicate group with its absorbed
   members and cosines. The reviewer swaps the auto-pick or drops a group.
   Every override is logged; a pattern in the overrides gets folded back into
   the rule.
3. **Export.** Final picks copied with deterministic names
   (`site_building_walk_t012.5_y090.jpg`) and appended to the dataset registry.
   The registry carries the building column, so train/val/test splits are always
   computed from it, never hand-assigned.

## Architecture

```
src/accio/
  core/      the science: extract, faces, blur gate, embed, dedup
  store/     persistence: walks, faces, picks, overrides, dataset registry
  jobs/      background ingest runner with progress
  server/    FastAPI routes, thin, no logic
web/         frontend (React + Vite + TypeScript)
tests/
```

The app owns one data root (`ACCIO_DATA`, default `./data`):

```
data/
  videos/      original .insv uploads, the raw ground truth
  walks/<id>/  pano/, faces/, manifest.csv, embeddings.npz
  accio.db     walk metadata + review decisions
```

Three rules:

- `core/` is pure functions: paths and arrays in, results out, no globals,
  unit-testable per stage.
- The store is the single source of truth. The GUI never computes science; it
  reads state and writes picks.
- The server is a thin translation layer, so the frontend can grow without
  touching the pipeline.

## Decisions

- **Frontend: React + Vite + TypeScript.** The review screen is image-grid heavy
  and the app is meant to grow (registry dashboard, split view, balance stats).
- **Storage: SQLite.** One file, transactional; per-class counts, split stats and
  override patterns are queries. Images stay on disk, the DB holds paths and
  metadata.
- **Embedding: DINOv3, CLS token, cosine dedup at tau 0.94, per walk.** The
  values the dedup POC validated (threshold gap: different walls 0.936, same
  wall one step later 0.965; patch-mean collapses on bare concrete). The
  embedder sits behind a small interface so a swap stays cheap.
- **One entry point.** The ingest job calls `jobs.pipeline.run_walk()`; the GUI
  is a layer over the store and never computes science.

## Running it

```
uv run uvicorn accio.server.app:app   # API on :8000, data root ./data (ACCIO_DATA)
cd web && npm run dev                 # UI on :5173
```

## Deploying it

```
ACCIO_DATA_DIR=/srv/accio/data docker compose up -d --build   # everything on :8000
```

The image carries the API, the worker and the built frontend. The stitcher
stays outside it: MediaSDK is its own amd64 image, and accio starts it as a
**sibling** container through the host's Docker socket.

That is the one thing worth understanding before changing the compose file.
A sibling's `-v` paths are resolved by the host's daemon, which cannot see
inside accio's filesystem, and Docker does not fail on a bind mount whose
source is missing. It creates an empty directory. So a wrong path gives a
successful mount of nothing and a stitch that exports zero frames with no
stated reason. The compose file avoids this by mounting the data root **at
the same path inside and out**, which makes every path the process computes
true on both sides. Mount it elsewhere and you must set `ACCIO_HOST_DATA` to
the host's name for it. A stitch that produces nothing probes the mount and
says which case it hit.

| Variable | Default | |
|---|---|---|
| `ACCIO_DATA` | `./data` | videos, walks, `accio.db` |
| `ACCIO_HOST_DATA` | `ACCIO_DATA` | what the host calls it, if not the same |
| `ACCIO_WEB_DIST` | `web/dist` | served at `/` when it exists |
| `HF_HOME` | `~/.cache/huggingface` | ~1.2 GB of weights; a volume in compose |

Torch resolves to the CPU build on Linux, which drops ~2.5 GB of unused CUDA
wheels. `pyproject.toml` says where to undo that if the server gets a GPU.
