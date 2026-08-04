import { useCallback, useEffect, useRef, useState } from 'react'
import type { Job, StageState, WalkDetail } from './api'

// The pipeline as blocks on the canvas: each one says what it emitted, so the
// funnel reads spatially instead of as a single line. A walk that is still
// running has no manifest yet, so the blocks read from the job instead and
// fill in as each stage lands. The chain is fixed; blocks move, nothing
// rewires.

const W = 232, H = 76, GAP_Y = 36
const SNAP = 8   // the dot pitch, so blocks land on the grid

type Pos = { x: number; y: number }

type Block = {
  id: string
  title: string
  sub: (w: WalkDetail) => string
  stat: (w: WalkDetail) => string
  // shown while the run is in flight, from whatever the job has reported
  live?: (stats: Record<string, number>) => string
  idle: string          // the sub line before the walk's own settings exist
}

/** "4320 frames · 2:24", the denominator the rest of the funnel cuts down */
function footage(frames?: number, seconds?: number): string {
  if (!frames) return 'uploaded'
  const s = Math.round(seconds ?? 0)
  const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  return `${frames.toLocaleString()} frames · ${clock}`
}

/** "vit_base_patch16_dinov3.lvd1689m" -> "DINOv3 ViT-B/16" */
function modelLabel(name: string): string {
  const m = /vit_(base|small|large)_patch(\d+)_(dinov\d)/.exec(name)
  if (!m) return name || 'unknown'
  const size = { base: 'B', small: 'S', large: 'L' }[m[1]] ?? m[1]
  return `${m[3].replace('dinov', 'DINOv')} ViT-${size}/${m[2]}`
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
    sub: (w) => modelLabel(w.pipeline.embed_model_used || w.pipeline.embed.model_name),
    stat: (w) => `${w.stages.faces} vectors`,
    live: (c) => (c.faces ? `${c.faces} vectors` : ''),
    idle: 'DINOv3 ViT-B/16',
  },
  {
    id: 'select',
    title: 'Select',
    sub: (w) => `greedy cosine · τ ${w.pipeline.dedup.tau}`
      + (w.pipeline.dedup.rule === 'auto' ? ' auto' : ''),
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
  const done = state === 'done'
  return {
    state,
    sub: walk ? b.sub(walk) : b.idle,
    stat: done && walk ? b.stat(walk) : (job && b.live?.(job.stats)) || '',
  }
}

const LAYOUT_KEY = 'accio.pipeline.layout'

// the stages a machine re-runs; Video is the upload and Review is people
const MACHINE = ['stitch', 'gate', 'faces', 'embed', 'select']

/** "Faces, Embed, Select": what editing this stage will actually re-run. */
export function rerunLabel(from: string): string {
  const i = MACHINE.indexOf(from)
  if (i < 0) return ''
  return MACHINE.slice(i)
    .map((id) => BLOCKS.find((b) => b.id === id)?.title ?? id)
    .join(', ')
}

/** One column, top to bottom: the walk enters at the top and falls through
    the stages, so reading down the canvas is reading the funnel. Centred by
    x, not by centring the canvas itself, so dragging a block never shifts
    the ones you did not touch. */
function defaultLayout(x: number): Record<string, Pos> {
  const out: Record<string, Pos> = {}
  BLOCKS.forEach((b, i) => { out[b.id] = { x, y: i * (H + GAP_Y) } })
  return out
}

function savedLayout(): Record<string, Pos> | null {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? 'null')
    if (saved && BLOCKS.every((b) => saved[b.id])) return saved
  } catch { /* fall through to the default chain */ }
  return null
}

/** Bezier between two blocks: bottom to top for the normal downward flow,
    side to side once a block has been dragged level with its neighbour. */
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

export default function Pipeline({ walk, job, selected, dirtyFrom, onSelect }: {
  walk: WalkDetail | null
  job: Job | null
  selected: string | null
  dirtyFrom: string | null
  onSelect: (id: string | null) => void
}) {
  // a staged edit dirties its own stage and everything downstream of it
  const dirtyAt = dirtyFrom ? BLOCKS.findIndex((b) => b.id === dirtyFrom) : -1
  const [layout, setLayout] = useState<Record<string, Pos> | null>(savedLayout)
  const [dragging, setDragging] = useState<string | null>(null)
  const canvas = useRef<HTMLDivElement>(null)
  const drag = useRef<
    { id: string; dx: number; dy: number; x0: number; y0: number; moved: boolean } | null
  >(null)
  const swallowClick = useRef(false)

  // first run: centre the column in whatever width the canvas got
  useEffect(() => {
    if (layout) return
    const w = canvas.current?.clientWidth ?? 0
    setLayout(defaultLayout(Math.max(0, Math.round((w - W) / 2 / SNAP) * SNAP)))
  }, [layout])

  useEffect(() => {
    if (layout) localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout))
  }, [layout])

  const onPointerDown = useCallback((e: React.PointerEvent, id: string) => {
    if (e.button !== 0 || !layout) return
    const p = layout[id]
    swallowClick.current = false   // never carry a swallow into a new gesture
    drag.current = { id, dx: e.clientX - p.x, dy: e.clientY - p.y,
                     x0: e.clientX, y0: e.clientY, moved: false }
    setDragging(id)
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }, [layout])

  // Two guards. The buttons check: if a pointerup is ever missed (capture
  // lost, released off-window) the block would keep following the cursor with
  // nothing held. The threshold: without it a click that jitters a pixel snaps
  // the block to the next 8 px, so clicking around slowly scrambles the chain.
  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    if (!(e.buttons & 1)) { drag.current = null; setDragging(null); return }
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < 4) return
    const x = Math.round((e.clientX - d.dx) / SNAP) * SNAP
    const y = Math.round((e.clientY - d.dy) / SNAP) * SNAP
    setLayout((l) => {
      if (!l || (l[d.id].x === x && l[d.id].y === y)) return l
      d.moved = true
      return { ...l, [d.id]: { x: Math.max(0, x), y: Math.max(0, y) } }
    })
  }, [])

  // Activation is a real click, not a synthesised one on pointerup: a dragged
  // block swallows the click that follows it, everything else opens.
  const onPointerUp = useCallback((e: React.PointerEvent) => {
    swallowClick.current = drag.current?.moved ?? false
    drag.current = null
    setDragging(null)
    ;(e.target as HTMLElement).releasePointerCapture?.(e.pointerId)
  }, [])

  const onClick = useCallback((block: Block) => {
    if (swallowClick.current) { swallowClick.current = false; return }
    onSelect(selected === block.id ? null : block.id)
  }, [onSelect, selected])

  if (!layout) return <div className="canvas" ref={canvas} />

  const height = Math.max(...BLOCKS.map((b) => layout[b.id].y)) + H

  return (
    <div className="canvas" ref={canvas} style={{ height }}>
      <svg className="links" width="100%" height={height}>
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4"
                  markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,1 L6,4 L0,7" />
          </marker>
        </defs>
        {BLOCKS.slice(0, -1).map((b, i) => (
          <path key={b.id} markerEnd="url(#arrow)"
                d={link(layout[b.id], layout[BLOCKS[i + 1].id])} />
        ))}
      </svg>
      {BLOCKS.map((b, i) => {
        const v = stageView(b, walk, job)
        const dirty = dirtyAt >= 0 && i >= dirtyAt
        return (
          <div
            key={b.id}
            className={`node ${v.state}${dirty ? ' dirty' : ''}${
              selected === b.id ? ' selected' : ''}${
              dragging === b.id ? ' dragging' : ''}`}
            style={{ left: layout[b.id].x, top: layout[b.id].y }}
            onPointerDown={(e) => onPointerDown(e, b.id)}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onClick={() => onClick(b)}
          >
            <div className="node-head">
              <span className="node-title">{b.title}</span>
              {v.state !== 'done' && <span className={`node-dot ${v.state}`} />}
            </div>
            <div className="node-sub">{v.sub}</div>
            <div className="node-stat">{v.stat || '\u00a0'}</div>
          </div>
        )
      })}
    </div>
  )
}
