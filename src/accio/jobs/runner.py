"""Background work: one worker thread, a queue of walks.

One worker: the embedder holds the GPU, and two emulated MediaSDK containers
at once do not fit on a laptop. Settings changes queue here too, since a
re-run from the gate re-renders and re-embeds every face.
"""

import itertools
import queue
import threading
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core import extract
from ..core.embed import Embedder, TimmEmbedder
from ..core.segment import OpenVocabSegmenter, SemanticSegmenter, Segmenter
from ..core.params import PipelineParams
from .pipeline import (ERROR_FILE, STAGES, clear_job, read_job, rerun,
                       run_walk, save_failure, save_job)


@dataclass
class Job:
    id: int
    walkId: str
    status: str = "queued"   # queued | running | done | error
    error: str = ""
    video: Path | None = field(default=None, repr=False)
    first: str | None = None  # re-run entry stage; None for a fresh ingest
    params: PipelineParams | None = field(default=None, repr=False)
    stages: dict[str, str] = field(
        default_factory=lambda: {s: "queued" for s in STAGES})
    stats: dict[str, int] = field(default_factory=dict)

    def public(self) -> dict:
        running = next((s for s, st in self.stages.items() if st == "running"), "")
        return {"id": self.id, "walkId": self.walkId, "status": self.status,
                "stage": running, "stages": dict(self.stages),
                "stats": dict(self.stats), "error": self.error,
                "first": self.first or "",
                "params": asdict(self.params) if self.params else None}


JOBS_KEPT = 50


class Runner:
    def __init__(self, out_root: Path, params: PipelineParams | None = None):
        self.out_root = out_root
        self.params = params or PipelineParams()
        self._embedders: dict[str, Embedder] = {}
        self._segmenters: dict[str, Segmenter] = {}
        # submit() runs on request threads while busy() and list() iterate
        self._jobs_lock = threading.Lock()
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._q: queue.Queue[Job] = queue.Queue()
        self.recover()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def recover(self) -> None:
        """Write a failure for walks whose run the process did not survive.

        Only the except block in _run_one writes error.json; a SIGKILL or OOM
        leaves nothing. The recorded stage is where Retry re-enters: the pano
        index is written at the end of the stitch, so its absence means the
        kill landed inside the stitch.
        """
        if not self.out_root.exists():
            return
        for walk in self.out_root.iterdir():
            if not walk.is_dir():
                continue
            job = read_job(walk)
            if job is not None:
                # '' is a fresh ingest, which falls through to the pano check
                stitched = (walk / "pano" / extract.PANO_INDEX).exists()
                save_failure(
                    walk, job.get("stage")
                    or ("gate" if stitched else "stitch"),
                    "the server stopped before this run finished",
                    "No failure was recorded because the process did not live "
                    "to write one. Retry runs it again from where it entered.")
                clear_job(walk)
                continue
            if not (walk / "source.json").exists():
                continue
            if (walk / "manifest.csv").exists() or (walk / ERROR_FILE).exists():
                continue
            # walks from before the job sentinel existed
            stitched = (walk / "pano" / extract.PANO_INDEX).exists()
            save_failure(
                walk, "gate" if stitched else "stitch",
                "the server stopped mid-run",
                "No failure was recorded because the process did not live to "
                "write one. Retry picks up "
                + ("from the gate; the stitch had already finished."
                   if stitched else
                   "from the stitch, reusing whatever frames already landed."))

    def embedder(self, params: PipelineParams) -> Embedder:
        """One embedder per backbone, kept between jobs; weights load on first use."""
        name = params.embed.model_name
        if name not in self._embedders:
            self._embedders[name] = TimmEmbedder(params.embed)
        return self._embedders[name]

    def segmenter(self, params: PipelineParams) -> Segmenter | None:
        """At most one segmenter is held; Grounding DINO plus SAM ViT-H is ~2.4 GB.

        The key includes the classes: an open-vocabulary model resolves its
        prompt into token spans at load time.
        """
        if not params.segment.enabled:
            return None
        name = params.segment.model_name
        key = f"{name}|{'|'.join(params.segment.classes)}"
        if key not in self._segmenters:
            self._segmenters.clear()
            self._free_device()
            self._segmenters[key] = (
                OpenVocabSegmenter(params.segment) if params.segment.kind == "open"
                else SemanticSegmenter(params.segment))
        return self._segmenters[key]

    @staticmethod
    def _free_device() -> None:
        """The caching allocator keeps blocks after the reference is dropped."""
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def submit(self, video: Path | None, first: str | None = None,
               params: PipelineParams | None = None, walk_id: str = "") -> Job:
        """A fresh ingest (first=None) or a re-run entering at `first`.

        `video` may be None for a re-run of a stage that does not read it.
        """
        if video is None and not (walk_id and first):
            raise ValueError("a job without a video needs a walk id and a stage")
        job = Job(id=next(self._ids), walkId=walk_id or video.stem, video=video,
                  first=first, params=params)
        # on disk before it is queued, so a process death leaves a trace
        save_job(self.out_root / job.walkId, first)
        with self._jobs_lock:
            self.jobs[job.id] = job
            self._prune()
        self._q.put(job)
        return job

    def _prune(self) -> None:
        finished = [j.id for j in self.jobs.values()
                    if j.status not in ("queued", "running")]
        for jid in sorted(finished)[:-JOBS_KEPT]:
            del self.jobs[jid]

    def busy(self, walk_id: str) -> bool:
        with self._jobs_lock:
            return any(j.walkId == walk_id and j.status in ("queued", "running")
                       for j in self.jobs.values())

    def list(self) -> list[dict]:
        with self._jobs_lock:
            jobs = sorted(self.jobs.values(), key=lambda j: -j.id)
        return [j.public() for j in jobs]

    def alive(self) -> bool:
        return self._thread.is_alive()

    def _work(self) -> None:
        # a failing handler (disk full) must not kill the worker thread
        while True:
            job = self._q.get()
            try:
                self._run_one(job)
            except Exception:
                traceback.print_exc()
            finally:
                self._q.task_done()

    def _run_one(self, job: Job) -> None:
        job.status = "running"

        def progress(stage: str, status: str, **counts) -> None:
            job.stages[stage] = status
            job.stats.update(counts)

        try:
            params = job.params or self.params
            seg = self.segmenter(params)
            if job.first is None:
                run_walk(job.video, self.out_root, params,
                         self.embedder(params), progress=progress,
                         segmenter=seg)
            else:
                rerun(job.video, self.out_root / job.walkId, params,
                      self.embedder(params), job.first, progress=progress,
                      segmenter=seg)
            job.status = "done"
        except Exception as e:
            job.status = "error"
            # negative: the innermost frames, where the failing line is
            job.error = traceback.format_exc(limit=-8)
            broke = ""
            for stage, state in job.stages.items():
                if state == "running":
                    job.stages[stage] = "error"
                    broke = broke or stage
            save_failure(self.out_root / job.walkId, broke, str(e) or
                         type(e).__name__, job.error)
        finally:
            clear_job(self.out_root / job.walkId)
