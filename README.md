# accio

Frame preparation for 360 site walks. Drop an Insta360 `.insv`, the app
stitches it, cuts each panorama into four pinhole faces, embeds them with
DINOv3 and collapses near-duplicates per walk. You review the groups, swap or
drop picks, and export a zip for annotation.

## Layout

```
src/accio/
  core/      extract, faces, blur, embed, calibrate, dedup, segment, export
  jobs/      the pipeline as stages, and the worker that runs it
  server/    FastAPI routes
  store/     SQLite: walk metadata and review decisions
web/         React + Vite frontend
tests/
```

Data lives under one root (`ACCIO_DATA`, default `./data`):

```
data/
  videos/      the uploaded .insv files
  walks/<id>/  pano/, faces/, manifest.csv, embeddings.npz, calibration.json, ...
  accio.db
```

## Running

```
uv run uvicorn accio.server.app:app   # API on :8000
cd web && npm install && npm run dev  # UI on :5173, proxies /api to :8000
uv run pytest
```

The stitcher is the Insta360 MediaSDK in its own Docker image
(`insta360-mediasdk:3.1.1`, amd64). `docker` and `ffprobe` must be on PATH.

## Deploying

```
ACCIO_DATA_DIR=/srv/accio/data docker compose up -d --build
```

One image with the API, the worker and the built frontend, on `127.0.0.1:8000`.
There is no auth, so put a proxy or a tunnel in front of it.

accio starts the stitcher as a sibling container through the host's Docker
socket. The stitcher's bind mounts are resolved by the host, so the data root
has to be visible under the same path on both sides. The compose file mounts it
at the same path inside and out for that reason. If you mount it somewhere
else, set `ACCIO_HOST_DATA` to the host path, or the stitcher gets an empty
directory and exports nothing.

| Variable | Default | |
|---|---|---|
| `ACCIO_DATA` | `./data` | data root |
| `ACCIO_HOST_DATA` | same as `ACCIO_DATA` | the host's path to it, if different |
| `ACCIO_WEB_DIST` | `web/dist` | built frontend, served at `/` if present |
| `ACCIO_MAX_UPLOAD_GB` | `40` | per-ingest upload limit |
| `HF_HOME` | `~/.cache/huggingface` | model weights, about 1.2 GB |

On Linux, torch is pinned to the CPU wheel index in `pyproject.toml`. Remove
that block and re-lock if the server has a GPU.
