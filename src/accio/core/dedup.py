"""Stage 5: greedy cosine dedup, per walk.

Walk the faces in capture order. A face whose max cosine to every already-kept
face is under tau is kept; otherwise it is absorbed by the kept face it
matched, and that anchor + cosine are recorded. The anchor mapping is what
the review UI shows as duplicate groups, and what the annotator's swap
overrides operate on.

Dedup runs per walk only: cross-walk near-duplicates are different walls that
look alike, and merging them would cost coverage.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DedupResult:
    kept: list[int]                              # indices into the input order
    anchor_of: dict[int, tuple[int, float]]      # dropped idx -> (kept idx, cosine)
    group_of: dict[int, list[tuple[int, float]]] = field(repr=False, default_factory=dict)

    @property
    def dropped(self) -> list[int]:
        return sorted(self.anchor_of)


def greedy_dedup(embeddings: np.ndarray, tau: float) -> DedupResult:
    """embeddings: (N, D) L2-normalised, in capture order."""
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D) embeddings, got shape {embeddings.shape}")
    kept: list[int] = []
    anchor_of: dict[int, tuple[int, float]] = {}
    for i in range(len(embeddings)):
        if not kept:
            kept.append(i)
            continue
        sims = embeddings[kept] @ embeddings[i]
        j = int(np.argmax(sims))
        if float(sims[j]) < tau:
            kept.append(i)
        else:
            anchor_of[i] = (kept[j], float(sims[j]))

    group_of: dict[int, list[tuple[int, float]]] = {k: [] for k in kept}
    for i, (k, sim) in anchor_of.items():
        group_of[k].append((i, sim))
    return DedupResult(kept=kept, anchor_of=anchor_of, group_of=group_of)
