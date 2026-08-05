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

from ..core.embed import Embedder, TimmEmbedder
from ..core.segment import SegformerSegmenter, Segmenter
from ..core.params import PipelineParams
from .pipeline import STAGES, rerun, run_walk, save_failure


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
        threading.Thread(target=self._work, daemon=True).start()

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
        """One segmenter per model, and none at all until a walk asks for it:
        the weights are a few hundred megabytes nobody should pay for by
        default."""
        if not params.segment.enabled:
            return None
        name = params.segment.model_name
        if name not in self._segmenters:
            self._segmenters[name] = SegformerSegmenter(params.segment)
        return self._segmenters[name]

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
        while True:
            job = self._q.get()
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
