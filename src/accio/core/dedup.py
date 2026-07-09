"""Stage 5: greedy cosine dedup, per walk.

Walk the faces in capture order. A face whose max cosine to every already-kept
face is under tau is kept; otherwise it is absorbed by the kept face it
matched, and that anchor + cosine are recorded. The anchor mapping is what
the review UI shows as duplicate groups, and what the annotator's swap
overrides operate on.

Dedup runs per walk only: cross-walk near-duplicates are different walls that
look alike, and merging them would cost coverage.
"""

from dataclasses import dataclass

import numpy as np

from .params import DedupParams


@dataclass(frozen=True)
class DedupResult:
    kept: list[int]                              # indices into the input order
    anchor_of: dict[int, tuple[int, float]]      # dropped idx -> (kept idx, cosine)

    @property
    def dropped(self) -> list[int]:
        return sorted(self.anchor_of)

    @property
    def group_of(self) -> dict[int, list[tuple[int, float]]]:
        groups: dict[int, list[tuple[int, float]]] = {k: [] for k in self.kept}
        for i, (k, sim) in self.anchor_of.items():
            groups[k].append((i, sim))
        return groups


def greedy_dedup(embeddings: np.ndarray, params: DedupParams) -> DedupResult:
    """embeddings: (N, D) L2-normalised, in capture order."""
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D) embeddings, got shape {embeddings.shape}")
    kept_rows = np.empty_like(embeddings)  # kept vectors packed at the front,
    kept: list[int] = []                   # so the loop matvecs a view, no copies
    anchor_of: dict[int, tuple[int, float]] = {}
    for i in range(len(embeddings)):
        if kept:
            sims = kept_rows[:len(kept)] @ embeddings[i]
            j = int(np.argmax(sims))
            if float(sims[j]) >= params.tau:
                anchor_of[i] = (kept[j], float(sims[j]))
                continue
        kept_rows[len(kept)] = embeddings[i]
        kept.append(i)
    return DedupResult(kept=kept, anchor_of=anchor_of)
