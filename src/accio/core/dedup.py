"""Stage 5: greedy cosine dedup, per walk.

Walk the faces in capture order. A face whose max cosine to every already-kept
face is under tau is kept; otherwise it is absorbed by the kept face it
matched, and that anchor + cosine are recorded. The anchor mapping is what
the review UI shows as duplicate groups, and what the annotator's swap
overrides operate on.

Dedup runs per walk only: cross-walk near-duplicates are different walls that
look alike, and merging them would cost coverage.
"""

from dataclasses import dataclass, replace

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


def known_negative_cosines(embeddings: np.ndarray, pano_idx: list[int],
                           yaws: list[int]) -> np.ndarray:
    """Cosines between pairs that are different views by construction.

    Opposite faces of one panorama look in opposite directions and, at 110
    degrees of field of view, share no pixels at all. Whatever those score is
    what "different" looks like for this walk, this camera and this backbone.
    """
    by_pano: dict[int, dict[int, int]] = {}
    for i, (p, y) in enumerate(zip(pano_idx, yaws)):
        by_pano.setdefault(p, {})[y % 360] = i
    out = []
    for faces in by_pano.values():
        for y, i in faces.items():
            j = faces.get((y + 180) % 360)
            if j is not None and i < j:
                out.append(float(embeddings[i] @ embeddings[j]))
    return np.array(out)


def auto_tau(embeddings: np.ndarray, pano_idx: list[int], yaws: list[int],
             quantile: float = 99.0, floor: float = 0.60,
             ceiling: float = 0.99) -> float:
    """A threshold from this walk's own scale, not a number carried between
    walks. Deterministic: a pure function of the embeddings and the face
    geometry, and whatever it resolves to is written into params.json.

    NOT VALIDATED, and off by default. Measured on GCMR and ASHV footage
    (Aug 2026), opposite faces turn out to be the easiest possible negatives:
    they score 0.38 to 0.58 on a walk whose consecutive same-wall frames sit
    at 0.90, so a threshold placed just above them merges nearly everything
    (148 faces down to 5). Two walls that merely look alike, a floor apart,
    are far more similar than a wall and the wall behind it. A usable rule
    needs negatives that are hard, not negatives that are certain: pairs far
    apart in the same walk, or labelled ones.
    """
    neg = known_negative_cosines(embeddings, pano_idx, yaws)
    if len(neg) < 8:
        return DedupParams().tau
    return round(float(np.clip(np.percentile(neg, quantile), floor, ceiling)), 4)


def resolve(embeddings: np.ndarray, params: DedupParams,
            pano_idx: list[int], yaws: list[int]) -> DedupParams:
    """Turn an 'auto' rule into the number it resolved to, so the manifest,
    params.json and the export all record the threshold actually used."""
    if params.rule != "auto":
        return params
    return replace(params, tau=auto_tau(embeddings, pano_idx, yaws))


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
