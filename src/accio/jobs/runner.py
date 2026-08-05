"""Background work: one worker thread, a queue of walks.

Jobs wrap jobs.pipeline with status the UI polls. One worker on purpose: the
embedder holds the GPU, and two emulated MediaSDK containers at once help
nobody on a laptop. A settings change queues here too rather than blocking a
request, because re-running from Gate re-renders and re-embeds every face.
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
from .pipeline import ERROR_FILE, STAGES, rerun, run_walk, save_failure


@dataclass
class Job:
    id: int
    walkId: str
    status: str = "queued"   # queued | running | done | error
    error: str = ""
    video: Path = field(default=Path(), repr=False)
    # the stage a re-run enters at; None for a fresh ingest, which runs it all
    first: str | None = None
    params: PipelineParams | None = field(default=None, repr=False)
    # per-stage state and the counts each one emitted, so the canvas can show
    # the run advancing rather than one opaque wait
    stages: dict[str, str] = field(
        default_factory=lambda: {s: "queued" for s in STAGES})
    stats: dict[str, int] = field(default_factory=dict)

    def public(self) -> dict:
        running = next((s for s, st in self.stages.items() if st == "running"), "")
        return {"id": self.id, "walkId": self.walkId, "status": self.status,
                "stage": running, "stages": dict(self.stages),
                "stats": dict(self.stats), "error": self.error,
                "first": self.first or "",
                # the settings this run is using, which are not the ones saved
                # next to the walk until it finishes
                "params": asdict(self.params) if self.params else None}


class Runner:
    def __init__(self, out_root: Path, params: PipelineParams | None = None):
        self.out_root = out_root
        self.params = params or PipelineParams()
        self._embedders: dict[str, Embedder] = {}
        self._segmenters: dict[str, Segmenter] = {}
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._q: queue.Queue[Job] = queue.Queue()
        self.recover()
        threading.Thread(target=self._work, daemon=True).start()

    def recover(self) -> None:
        """Mark walks whose run the process did not survive.

        Only the except block writes error.json, so a SIGKILL, an OOM or a
        closed laptop leaves a walk with panoramas, no manifest and no failure.
        Every route then refuses it: rerun 409s with "has no frames", retry
        409s with "no failure to retry", and the canvas shows every stage
        queued forever. The one action that works is Delete, which also unlinks
        the multi-gigabyte original. Writing the failure down turns that into
        one Retry click.

        The stage recorded is what Retry re-enters at, so it is worth getting
        right: panos.json is written at the end of the stitch, so its presence
        means the expensive part is done and the retry starts at the gate. Its
        absence means the kill landed inside the stitch, where the half-written
        export is still digit-named and genuinely does get reused.
        """
        if not self.out_root.exists():
            return
        for walk in self.out_root.iterdir():
            if not walk.is_dir() or not (walk / "source.json").exists():
                continue
            if (walk / "manifest.csv").exists() or (walk / ERROR_FILE).exists():
                continue
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
        """One embedder per backbone, kept between jobs.

        Per backbone, not one for the runner: a job that asked for a different
        model would otherwise embed with the loaded one and record the name it
        asked for, which is a lie no downstream check would catch. Weights load
        on first use, so naming a backbone costs nothing until it runs.
        """
        name = params.embed.model_name
        if name not in self._embedders:
            self._embedders[name] = TimmEmbedder(params.embed)
        return self._embedders[name]

    def segmenter(self, params: PipelineParams) -> Segmenter | None:
        """The current segmenter, and none at all until a walk asks for one:
        the weights are a few hundred megabytes nobody should pay for by
        default.

        Exactly one is held. The key includes the classes, because an
        open-vocabulary model resolves its prompt into token spans when it
        loads, so a different vocabulary is a different segmenter whatever its
        weights are. Keeping them all was the problem: Grounding DINO plus SAM
        ViT-H is around 2.4 GB, and editing the class list is the entire point
        of an open-vocabulary model, so the workflow it exists for was the one
        that allocated until the device ran out.
        """
        if not params.segment.enabled:
            return None
        name = params.segment.model_name
        key = f"{name}|{'|'.join(params.segment.classes)}"
        if key not in self._segmenters:
            self._segmenters.clear()          # drop the old weights first
            self._free_device()
            self._segmenters[key] = (
                OpenVocabSegmenter(params.segment) if params.segment.kind == "open"
                else SemanticSegmenter(params.segment))
        return self._segmenters[key]

    @staticmethod
    def _free_device() -> None:
        """Dropping the reference is not enough on an accelerator: the caching
        allocator keeps the blocks until it is told otherwise."""
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass          # freeing is an optimisation; never fail a run for it

    def submit(self, video: Path, first: str | None = None,
               params: PipelineParams | None = None, walk_id: str = "") -> Job:
        """A fresh ingest (first=None) or a re-run entering at `first`."""
        job = Job(id=next(self._ids), walkId=walk_id or video.stem, video=video,
                  first=first, params=params)
        self.jobs[job.id] = job
        self._q.put(job)
        return job

    def busy(self, walk_id: str) -> bool:
        return any(j.walkId == walk_id and j.status in ("queued", "running")
                   for j in self.jobs.values())

    def list(self) -> list[dict]:
        return [j.public() for j in sorted(self.jobs.values(), key=lambda j: -j.id)]

    def _work(self) -> None:
        # the loop must outlive anything inside it. The handler below writes to
        # disk, and the likeliest reason a run failed is that the disk is full,
        # which makes the handler fail too. That used to kill the thread, and
        # then submit() kept accepting work that nothing would ever pick up:
        # every walk sat queued forever with no error and no clue, until
        # somebody thought to restart the server.
        while True:
            try:
                self._run_one(self._q.get())
            except Exception:
                traceback.print_exc()

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
            job.error = traceback.format_exc(limit=2)
            for stage, state in job.stages.items():
                if state == "running":
                    job.stages[stage] = "error"
            # on disk too: jobs live in memory, the walk does not
            broke = next((s for s, st in job.stages.items()
                          if st == "error"), "")
            save_failure(self.out_root / job.walkId, broke, str(e) or
                         type(e).__name__, job.error)
