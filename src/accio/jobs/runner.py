"""Background ingest: one worker thread, a queue of walks.

Jobs wrap jobs.pipeline.run_walk with status the UI polls. One worker on
purpose: the embedder holds the GPU, and two emulated MediaSDK containers at
once help nobody on a laptop.
"""

import itertools
import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from ..core.embed import Dinov3Embedder
from ..core.params import PipelineParams
from .pipeline import run_walk


STAGES = ("stitch", "gate", "faces", "embed", "select")


@dataclass
class Job:
    id: int
    walkId: str
    status: str = "queued"   # queued | running | done | error
    error: str = ""
    video: Path = field(default=Path(), repr=False)
    # per-stage state and the counts each one emitted, so the canvas can show
    # the run advancing rather than one opaque wait
    stages: dict[str, str] = field(
        default_factory=lambda: {s: "queued" for s in STAGES})
    stats: dict[str, int] = field(default_factory=dict)

    def public(self) -> dict:
        running = next((s for s, st in self.stages.items() if st == "running"), "")
        return {"id": self.id, "walkId": self.walkId, "status": self.status,
                "stage": running, "stages": dict(self.stages),
                "stats": dict(self.stats), "error": self.error}


class Runner:
    def __init__(self, out_root: Path, params: PipelineParams | None = None):
        self.out_root = out_root
        self.params = params or PipelineParams()
        self.embedder = Dinov3Embedder(self.params.embed)  # loads on first job
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._q: queue.Queue[Job] = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, video: Path) -> Job:
        job = Job(id=next(self._ids), walkId=video.stem, video=video)
        self.jobs[job.id] = job
        self._q.put(job)
        return job

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
                run_walk(job.video, self.out_root, self.params, self.embedder,
                         progress=progress)
                job.status = "done"
            except Exception:
                job.status = "error"
                job.error = traceback.format_exc(limit=2)
                for stage, state in job.stages.items():
                    if state == "running":
                        job.stages[stage] = "error"
