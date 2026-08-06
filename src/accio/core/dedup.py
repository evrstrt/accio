"""Stage 5: greedy cosine dedup, per walk.

Walk the faces in capture order. A face whose max cosine to every already-kept
face is under tau is kept; otherwise it is absorbed by the kept face it
matched, and that anchor + cosine are recorded. The anchor mapping is what
the review UI shows as duplicate groups, and what the annotator's swap
overrides operate on.

Grouping and representing are separate questions. The anchor is whichever
member arrived first, which makes it a stable name for the group and a bad
choice of frame: an operator entering a bay mid-turn produces a smeared frame,
settles, and takes several clean ones, all of which absorb into the smear. So
the anchor stays the group's identity and `sharpest` picks what actually gets
exported.

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


def greedy_dedup(embeddings: np.ndarray, params: DedupParams,
                 pano_of: np.ndarray | None = None) -> DedupResult:
    """embeddings: (N, D) L2-normalised, in capture order.

    pano_of names the panorama each face was cut from. Faces of one panorama
    can never absorb each other: they are one position looking four ways, and
    the sphere at a station is coverage, not redundancy. A symmetric room, or
    the 20 degrees of overlap every seam carries by design, pushes two headings
    over any threshold, and merging them deletes a heading nothing will shoot
    again. Measured on the 7th Floor walk, worst same-panorama cosine 0.9534
    against a calibrated tau of 0.9317: without the guard, two stations lost a
    heading. Omit it and every face counts as its own station.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D) embeddings, got shape {embeddings.shape}")
    if pano_of is not None and len(pano_of) != len(embeddings):
        raise ValueError(f"got {len(pano_of)} panorama ids for "
                         f"{len(embeddings)} embeddings")
    kept_rows = np.empty_like(embeddings)  # kept vectors packed at the front,
    kept: list[int] = []                   # so the loop matvecs a view, no copies
    kept_pano = np.empty(len(embeddings), dtype=np.int64)
    anchor_of: dict[int, tuple[int, float]] = {}
    for i in range(len(embeddings)):
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
    return DedupResult(kept=kept, anchor_of=anchor_of)


def drop_solo(result: DedupResult, ratio: np.ndarray, picks: dict[int, int],
              floor: float) -> DedupResult:
    """Forget groups of one whose only frame is a smear.

    Everywhere else blur is already handled: a group is several looks at one
    place, and `sharpest` exports the best of them, with the wall's texture
    held constant because it is the same wall. A group of one has no such
    choice. It is the single path by which an unusable frame reaches a
    labeller, and so the only place a sharpness threshold earns its keep.

    Judged on `ratio`, sharpness over the recent norm for that face's own
    heading, so a plain wall is not punished for being plain.

    Dropping one is a real loss of coverage: nothing else in the walk looks
    like it, which is what made it a group of one. That is the trade being
    made, a view seen once and too smeared to label against an annotator's
    time and a wrong mask, and it is made here in the open rather than
    upstream by accident.
    """
    if floor <= 0:
        return result
    gone = {k for k, members in result.group_of.items()
            if not members and ratio[picks.get(k, k)] < floor}
    if not gone:
        return result
    return DedupResult(kept=[k for k in result.kept if k not in gone],
                       anchor_of=dict(result.anchor_of))


def sharpest(result: DedupResult, sharpness: np.ndarray) -> dict[int, int]:
    """anchor -> the member of its group to export.

    The sharpest, by the per-face score render_faces recorded. Ties go to the
    anchor, so a group whose members are equally sharp keeps the frame the
    reviewer is looking at rather than shuffling for no reason.

    No normalisation, and none needed: a group is one place, so every member
    is looking at the same wall and the texture that makes the raw score
    ambiguous is constant across the comparison.
    """
    picks = {}
    for anchor, members in result.group_of.items():
        best = anchor
        for i, _cos in members:
            if sharpness[i] > sharpness[best]:
                best = i
        picks[anchor] = best
    return picks
