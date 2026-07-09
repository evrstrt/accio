# accio

Frame preparation app for 360 site walks: drop an .insv, review the deduplicated
clusters, swap or drop picks, export a tracked set for annotation.

## Flow

1. **Ingest.** An .insv plus walk metadata (site, building, flat, stage, operator,
   mount height, date). The pipeline runs as a background job: decimate, gnomonic
   faces, relative blur gate, DINOv3 embed, greedy dedup per walk. Faces,
   embeddings and auto-picks land in the store.
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
ACCIO_OUT=$PWD/out uv run uvicorn accio.server.app:app   # API on :8000
cd web && npm run dev                                     # UI on :5173
```
