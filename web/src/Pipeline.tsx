import { STAGES } from './api'
import { backboneLabel } from './Inspector'
import type { Job, StageState, WalkDetail } from './api'

// The pipeline as blocks on the canvas: each one says what it emitted, so the
// funnel reads spatially instead of as a single line. A walk that is still
// running has no manifest yet, so the blocks read from the job instead and
// fill in as each stage lands. The chain is fixed and so is the layout; a
// block is a button that opens its settings, nothing more.
//
// Calibrate is the one block off the main line. Every other stage narrows the
// walk; that one measures it, and what it emits is a threshold rather than
// frames. So the frames run Embed -> Select down the column and the number
// comes in from the side, on a dashed link.

const W = 232, H = 76, GAP_Y = 36
const ASIDE_X = 72   // how far a side node sits off the column

type Pos = { x: number; y: number }

type Block = {
  id: string
  title: string
  sub: (w: WalkDetail) => string
  stat: (w: WalkDetail) => string
  // shown while the run is in flight, from whatever the job has reported
  live?: (stats: Record<string, number>) => string
  idle: string          // the sub line before the walk's own settings exist
  aside?: true          // sits off the column: measures the walk, does not cut it
}

/** "4320 frames · 2:24", the denominator the rest of the funnel cuts down */
function footage(frames?: number, seconds?: number): string {
  if (!frames) return 'uploaded'
  const s = Math.round(seconds ?? 0)
  const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  return `${frames.toLocaleString()} frames · ${clock}`
}

const BLOCKS: Block[] = [
  {
    id: 'video',
    title: 'Video',
    sub: (w) => [w.meta?.site, w.meta?.building].filter(Boolean).join(' · ')
      || 'dual-fisheye .insv',
    stat: (w) => footage(w.stages.frames, w.stages.seconds),
    live: (c) => footage(c.frames, c.seconds),
    idle: 'dual-fisheye .insv',
  },
  {
    id: 'stitch',
    title: 'Stitch',
    sub: (w) => `MediaSDK optflow · ${w.pipeline.extract.fps}fps`,
    stat: (w) => `${w.stages.panos} panos`,
    live: (c) => (c.panos ? `${c.panos} panos` : ''),
    // the sample rate is the interesting part: 4320 frames in, 144 out
    idle: 'MediaSDK optflow',
  },
  {
    id: 'gate',
    title: 'Gate',
    sub: (w) => `sharpest per ${w.pipeline.gate.window}`,
    stat: (w) => `${w.stages.sharp} sharp`,
    live: (c) => (c.sharp ? `${c.sharp} sharp` : ''),
    idle: 'sharpest per window',
  },
  {
    id: 'faces',
    title: 'Faces',
    sub: (w) => `gnomonic · ${w.pipeline.faces.yaws.length} × ${w.pipeline.faces.fov_deg}°`,
    stat: (w) => `${w.stages.faces} faces`,
    live: (c) => (c.faces ? `${c.faces} faces` : ''),
    idle: 'gnomonic',
  },
  {
    id: 'embed',
    title: 'Embed',
    sub: (w) => backboneLabel(w.pipeline.embed.model_name) || 'unknown',
    stat: (w) => `${w.stages.faces} vectors`,
    live: (c) => (c.faces ? `${c.faces} vectors` : ''),
    idle: 'DINOv3 ViT-B/16',
  },
  {
    id: 'calibrate',
    title: 'Calibrate',
    sub: (w) => `p${w.pipeline.calib.quantile} of ${w.pipeline.calib.samples} panos`,
    stat: (w) => `${w.stages.reference} over ${w.stages.pairs} pairs`,
    live: (c) => (c.pairs ? `${c.reference} over ${c.pairs} pairs` : ''),
    idle: 'identical frames',
    aside: true,
  },
  {
    id: 'select',
    title: 'Select',
    sub: (w) => `greedy cosine · τ ${w.pipeline.dedup.tau}`
      + (w.pipeline.dedup.rule === 'calibrated' ? ' calibrated' : ''),
    stat: (w) => `${w.stages.absorbed} absorbed`,
    live: (c) => (c.absorbed != null ? `${c.absorbed} absorbed` : ''),
    idle: 'greedy cosine',
  },
  {
    id: 'review',
    title: 'Review',
    sub: (w) => (w.stages.dropped ? `${w.stages.dropped} dropped` : 'no overrides'),
    stat: (w) => `${w.stages.kept} kept`,
    idle: 'human overrides',
  },
]

/** What each block shows and how it looks, from a finished walk or a live job.
    Video and Review bracket the machine stages: Video is done as soon as the
    upload is in, Review only becomes available once the run finishes. */
function stageView(b: Block, walk: WalkDetail | null, job: Job | null) {
  // A run in flight wins over the manifest: re-running an existing walk must
  // show this run's progress, not the counts the last one left behind.
  const live = job !== null && job.status !== 'done'
  const state: StageState =
    b.id === 'video' ? 'done'
      : b.id === 'review' ? (live || !walk ? 'queued' : 'done')
      : live ? job.stages[b.id] ?? 'queued'
      : walk ? 'done' : 'queued'
  // a stage above the one a re-run entered at was not touched, so what the
  // manifest says about it is still true; at and below, only the job knows
  const reused = live && !!job.first
    && STAGES.indexOf(b.id) < STAGES.indexOf(job.first)
  // and the settings on screen are the ones running, not the ones last saved
  const shown = live && job.params && walk
    ? { ...walk, pipeline: { ...walk.pipeline, ...job.params } }
    : walk
  return {
    state,
    sub: shown ? b.sub(shown) : b.idle,
    stat: live && !reused
      ? (job && b.live?.(job.stats)) || ''
      : (state === 'done' && walk ? b.stat(walk) : ''),
  }
}

/** "Faces, Embed, Select": what editing this stage will actually re-run. */
export function rerunLabel(from: string): string {
  const i = STAGES.indexOf(from)
  if (i < 0) return ''
  return STAGES.slice(i)
    .map((id) => BLOCKS.find((b) => b.id === id)?.title ?? id)
    .join(', ')
}

/** Who feeds whom. The column top to bottom, plus the side node hanging off
    Embed: the frames skip it and only its threshold reaches Select. */
const EDGES: [string, string][] = [
  ['video', 'stitch'], ['stitch', 'gate'], ['gate', 'faces'], ['faces', 'embed'],
  ['embed', 'select'], ['select', 'review'],
  ['embed', 'calibrate'], ['calibrate', 'select'],
]

const isAside = (id: string) => !!BLOCKS.find((b) => b.id === id)?.aside

// how wide the whole graph is. The graph is centred by CSS at this width
// rather than by measuring the canvas: nothing to re-measure when the
// inspector opens and takes 420 px of it.
const SPAN = BLOCKS.some((b) => b.aside) ? W + ASIDE_X + W : W

/** One column, top to bottom: the walk enters at the top and falls through
    the stages, so reading down the canvas is reading the funnel. A side node
    takes its own row but sits out to the right, so the column stays a column.
    Fixed, in the graph's own coordinates. */
const LAYOUT: Record<string, Pos> = Object.fromEntries(BLOCKS.map((b, i) => [
  b.id, { x: b.aside ? W + ASIDE_X : 0, y: i * (H + GAP_Y) },
]))

const HEIGHT = Math.max(...BLOCKS.map((b) => LAYOUT[b.id].y)) + H

/** Bezier between two blocks, bottom to top: the chain runs downward, and the
    side-to-side form is there for a side node level with its neighbour. */
function link(a: Pos, b: Pos): string {
  if (b.y >= a.y + H) {
    const x1 = a.x + W / 2, y1 = a.y + H, x2 = b.x + W / 2, y2 = b.y
    const off = Math.max(20, (y2 - y1) / 2)
    return `M${x1},${y1} C${x1},${y1 + off} ${x2},${y2 - off} ${x2},${y2}`
  }
  const fwd = b.x > a.x
  const x1 = fwd ? a.x + W : a.x
  const x2 = fwd ? b.x : b.x + W
  const y1 = a.y + H / 2, y2 = b.y + H / 2
  const off = Math.max(40, Math.abs(x2 - x1) / 2) * (fwd ? 1 : -1)
  return `M${x1},${y1} C${x1 + off},${y1} ${x2 - off},${y2} ${x2},${y2}`
}

/** A link to or from the side node. It leaves the column through a side and
    turns, so a branch reads as a branch: straight down the chain is the walk,
    out and back is the measurement. */
function branch(a: Pos, b: Pos, leaving: boolean): string {
  if (leaving) {                       // column -> aside: out sideways, then down
    const right = b.x >= a.x
    const x1 = right ? a.x + W : a.x, y1 = a.y + H / 2
    const x2 = b.x + W / 2, y2 = b.y
    return `M${x1},${y1} C${x1 + (right ? 56 : -56)},${y1} ${x2},${y1} ${x2},${y2}`
  }
  const right = a.x >= b.x             // aside -> column: down, then back in
  const x1 = a.x + W / 2, y1 = a.y + H
  const x2 = right ? b.x + W : b.x, y2 = b.y + H / 2
  return `M${x1},${y1} C${x1},${y1 + 56} ${x2 + (right ? 56 : -56)},${y2} ${x2},${y2}`
}

export default function Pipeline({ walk, job, selected, dirtyFrom, onSelect }: {
  walk: WalkDetail | null
  job: Job | null
  selected: string | null
  dirtyFrom: string | null
  onSelect: (id: string | null) => void
}) {
  // a staged edit dirties its own stage and everything downstream of it
  const dirtyAt = dirtyFrom ? BLOCKS.findIndex((b) => b.id === dirtyFrom) : -1

  return (
    <div className="canvas">
      <div className="graph" style={{ width: SPAN, height: HEIGHT }}>
        <svg className="links" width={SPAN} height={HEIGHT}>
          <defs>
            <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4"
                    markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,1 L6,4 L0,7" />
            </marker>
          </defs>
          {EDGES.map(([from, to]) => {
            const off = isAside(from) || isAside(to)
            return (
              <path key={`${from}-${to}`} markerEnd="url(#arrow)"
                    className={off ? 'aside' : undefined}
                    d={off ? branch(LAYOUT[from], LAYOUT[to], isAside(to))
                           : link(LAYOUT[from], LAYOUT[to])} />
            )
          })}
        </svg>
        {BLOCKS.map((b, i) => {
          const v = stageView(b, walk, job)
          const dirty = dirtyAt >= 0 && i >= dirtyAt
          return (
            <button
              key={b.id}
              className={`node ${v.state}${dirty ? ' dirty' : ''}${
                selected === b.id ? ' selected' : ''}`}
              style={{ left: LAYOUT[b.id].x, top: LAYOUT[b.id].y }}
              onClick={() => onSelect(selected === b.id ? null : b.id)}
            >
              <div className="node-head">
                <span className="node-title">{b.title}</span>
                {v.state !== 'done' && <span className={`node-dot ${v.state}`} />}
              </div>
              <div className="node-sub">{v.sub}</div>
              <div className="node-stat">{v.stat || '\u00a0'}</div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
