"""Greedy cosine dedup, per walk.

Faces are visited sharpest first. A face under tau against every kept face is
kept; otherwise it is absorbed by the kept face it matched, and that anchor
and cosine are recorded.

Sharpest first matters: it makes the anchor the sharpest member, so the set
that was compared is the set that ships. Visiting in capture order and then
swapping each group to its sharpest member left 175 of 224 shipped frames
above tau against another shipped frame on the 7th Floor walk. Re-deduping the
shipped set instead chains merges (A into B, B into C) at cosines no threshold
authorised, and took the 0717 walk's false-merge rate from 2.4% to 6.3%.

Per walk only: cross-walk near-duplicates are different walls that look alike.
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


def greedy_dedup(embeddings: np.ndarray, params: DedupParams,
                 sharpness: np.ndarray,
                 pano_of: np.ndarray | None = None) -> DedupResult:
    """embeddings: (N, D) L2-normalised, in capture order.

    Faces cut from the same panorama never absorb each other: they are one
    position looking four ways, and the seam overlap or a symmetric room can
    push two headings over tau (worst seen 0.9534 against a tau of 0.9317).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D) embeddings, got shape {embeddings.shape}")
    if len(sharpness) != len(embeddings):
        raise ValueError(f"got {len(sharpness)} sharpness scores for "
                         f"{len(embeddings)} embeddings")
    if pano_of is not None and len(pano_of) != len(embeddings):
        raise ValueError(f"got {len(pano_of)} panorama ids for "
                         f"{len(embeddings)} embeddings")
    kept_rows = np.empty_like(embeddings)
    kept: list[int] = []
    kept_pano = np.empty(len(embeddings), dtype=np.int64)
    anchor_of: dict[int, tuple[int, float]] = {}
    for i in np.argsort(-np.asarray(sharpness, dtype=float), kind="stable"):
        i = int(i)
        station = int(pano_of[i]) if pano_of is not None else i
        if kept:
            sims = kept_rows[:len(kept)] @ embeddings[i]
            sims[kept_pano[:len(kept)] == station] = -1.0
            j = int(np.argmax(sims))
            if float(sims[j]) >= params.tau:
                anchor_of[i] = (kept[j], float(sims[j]))
                continue
        kept_rows[len(kept)] = embeddings[i]
        kept_pano[len(kept)] = station
        kept.append(i)
    return DedupResult(kept=sorted(kept), anchor_of=anchor_of)


def drop_solo(result: DedupResult, ratio: np.ndarray,
              floor: float) -> DedupResult:
    """Drop groups of one whose only frame is a smear (ratio under floor).

    A group of one has no sharper member to stand in for it, so this is the
    one place a sharpness threshold applies. It costs coverage: nothing else
    in the walk looks like that frame.
    """
    if floor <= 0:
        return result
    gone = {k for k, members in result.group_of.items()
            if not members and ratio[k] < floor}
    if not gone:
        return result
    return DedupResult(kept=[k for k in result.kept if k not in gone],
                       anchor_of=dict(result.anchor_of))
