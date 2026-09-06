import { STAGES } from './api'
import type { Job, StageState, WalkDetail } from './api'
import { backboneLabel, segmenterLabel, TITLES } from './labels'

// the pipeline as a fixed column of blocks, read from the manifest or, while a
// run is in flight, from the job. Calibrate sits off the column: it emits a
// threshold, not frames.

const W = 232, H = 76, GAP_Y = 36
const ASIDE_X = 72   // side node offset from the column

type Pos = { x: number; y: number }

type Block = {
  id: string
  sub: (w: WalkDetail) => string
  stat: (w: WalkDetail) => string
  // shown while the run is in flight, from the job's stats
  live?: (stats: Record<string, number>) => string
  idle: string          // sub line before the walk's own settings exist
  aside?: true          // sits off the column
}

/** "4320 frames · 2:24" */
function footage(frames?: number, seconds?: number): string {
  if (!frames) return 'uploaded'
  const s = Math.round(seconds ?? 0)
  const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  return `${frames.toLocaleString()} frames · ${clock}`
}

const BLOCKS: Block[] = [
  {
    id: 'video',
    sub: (w) => [w.meta?.site, w.meta?.building].filter(Boolean).join(' · ')
      || 'dual-fisheye .insv',
    stat: (w) => footage(w.stages.frames, w.stages.seconds),
    live: (c) => footage(c.frames, c.seconds),
    idle: 'dual-fisheye .insv',
  },
  {
    id: 'stitch',
    sub: (w) => `MediaSDK optflow · ${w.pipeline.extract.fps}fps`,
    stat: (w) => `${w.stages.panos} panos`,
    live: (c) => (c.panos ? `${c.panos} panos` : ''),
    idle: 'MediaSDK optflow',
  },
  {
    id: 'gate',
    sub: (w) => `drop under ${w.pipeline.gate.dead.toFixed(2)}× median`,
    stat: (w) => `${w.stages.sharp} legible`,
    live: (c) => (c.sharp ? `${c.sharp} legible` : ''),
    idle: 'unusable footage only',
  },
  {
    id: 'faces',
    sub: (w) => `gnomonic · ${w.pipeline.faces.yaws.length} × ${w.pipeline.faces.fov_deg}°`,
    stat: (w) => `${w.stages.faces} faces`,
    live: (c) => (c.faces ? `${c.faces} faces` : ''),
    idle: 'gnomonic',
  },
  {
    id: 'embed',
    sub: (w) => backboneLabel(w.pipeline.embed.model_name) || 'unknown',
    stat: (w) => `${w.stages.faces} vectors`,
    live: (c) => (c.faces ? `${c.faces} vectors` : ''),
    idle: 'DINOv3 ViT-B/16',
  },
  {
    id: 'calibrate',
    sub: (w) => `${w.pipeline.calib.false_merge_pct}% false merges allowed`,
    stat: (w) => `τ ${w.stages.calibTau} over ${w.stages.farPairs} far pairs`,
    live: (c) => (c.farPairs ? `τ ${c.calibTau} over ${c.farPairs} far pairs` : ''),
    idle: 'what elsewhere scores',
    aside: true,
  },
  {
    id: 'select',
    sub: (w) => `greedy cosine · τ ${w.pipeline.dedup.tau}`
      + (w.pipeline.dedup.rule === 'calibrated' ? ' calibrated' : ''),
    stat: (w) => `${w.stages.absorbed} absorbed`,
    live: (c) => (c.absorbed != null ? `${c.absorbed} absorbed` : ''),
    idle: 'greedy cosine',
  },
  {
    id: 'segment',
    sub: (w) => (w.pipeline.segment.enabled
      ? segmenterLabel(w.pipeline.segment.model_name) : 'off'),
    stat: (w) => (w.stages.segmented
      ? `${w.stages.segmented} masked` : 'not run'),
    live: (c) => (c.segmented ? `${c.segmented} masked` : ''),
    idle: 'semantic classes',
  },
  {
    id: 'review',
    sub: (w) => (w.stages.dropped ? `${w.stages.dropped} dropped` : 'no overrides'),
    stat: (w) => `${w.stages.kept} kept`,
    idle: 'human overrides',
  },
]

/** A block's state and text, from the finished walk or the live job. */
function stageView(b: Block, walk: WalkDetail | null, job: Job | null) {
  // a run in flight wins over the manifest
  const live = job !== null && job.status !== 'done'
  // a walk with no frames never finished; without a recorded failure, no stage is known to have completed
  const incomplete = !live && !!walk && walk.faces === 0
  const broke = incomplete ? STAGES.indexOf(walk!.error?.stage ?? '') : -1
  const rest = (id: string): StageState => {
    if (!incomplete) return 'done'
    if (broke < 0) return 'queued'
    const i = STAGES.indexOf(id)
    return i < broke ? 'done' : i === broke ? 'error' : 'queued'
  }
  const state: StageState =
    b.id === 'video' ? 'done'
      : b.id === 'review' ? (live || !walk || incomplete ? 'queued' : 'done')
      : live ? job.stages[b.id] ?? 'queued'
      : walk ? rest(b.id) : 'queued'
  // stages above the one a re-run entered at keep their manifest counts
  const reused = live && !!job.first
    && STAGES.indexOf(b.id) < STAGES.indexOf(job.first)
  // show the settings that are running, not the ones last saved
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

const EDGES: [string, string][] = [
  ['video', 'stitch'], ['stitch', 'gate'], ['gate', 'faces'], ['faces', 'embed'],
  ['embed', 'select'], ['select', 'segment'], ['segment', 'review'],
  ['embed', 'calibrate'], ['calibrate', 'select'],
]

const isAside = (id: string) => !!BLOCKS.find((b) => b.id === id)?.aside

// centred by CSS at this width, so nothing is re-measured when the inspector opens
const SPAN = BLOCKS.some((b) => b.aside) ? W + ASIDE_X + W : W

// one row per block; a side node keeps its row but sits to the right
const LAYOUT: Record<string, Pos> = Object.fromEntries(BLOCKS.map((b, i) => [
  b.id, { x: b.aside ? W + ASIDE_X : 0, y: i * (H + GAP_Y) },
]))

const HEIGHT = Math.max(...BLOCKS.map((b) => LAYOUT[b.id].y)) + H

/** Bezier between two blocks: bottom to top, or side to side when level. */
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

/** Link to or from the side node: leaves the column sideways and turns. */
function branch(a: Pos, b: Pos, leaving: boolean): string {
  if (leaving) {                       // column -> aside: out, then down
    const right = b.x >= a.x
    const x1 = right ? a.x + W : a.x, y1 = a.y + H / 2
    const x2 = b.x + W / 2, y2 = b.y
    return `M${x1},${y1} C${x1 + (right ? 56 : -56)},${y1} ${x2},${y1} ${x2},${y2}`
  }
  const right = a.x >= b.x             // aside -> column: down, then in
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
                <span className="node-title">{TITLES[b.id]}</span>
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
