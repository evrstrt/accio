import { useCallback, useEffect, useRef, useState } from 'react'
import type { WalkDetail } from './api'

// The pipeline as blocks on the canvas: each one says what it emitted, so the
// funnel reads spatially instead of as a single line. Read-only for now, the
// chain is fixed; blocks move, nothing rewires.

const W = 232, H = 76, GAP_Y = 36
const SNAP = 8   // the dot pitch, so blocks land on the grid

type Pos = { x: number; y: number }

type Block = {
  id: string
  title: string
  sub: (w: WalkDetail) => string
  stat: (w: WalkDetail) => string
  opens?: 'review' | 'export'
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
    id: 'walk',
    title: 'Walk',
    sub: (w) => [w.meta?.site, w.meta?.building].filter(Boolean).join(' · ')
      || 'dual-fisheye .insv',
    stat: (w) => w.meta?.shotDate || 'no date',
  },
  {
    id: 'stitch',
    title: 'Stitch',
    sub: (w) => `MediaSDK optflow · ${w.pipeline.extract.fps}fps`,
    stat: (w) => `${w.stages.panos} panos`,
  },
  {
    id: 'gate',
    title: 'Gate',
    sub: (w) => `sharpest per ${w.pipeline.gate.window}`,
    stat: (w) => `${w.stages.sharp} sharp`,
  },
  {
    id: 'faces',
    title: 'Faces',
    sub: (w) => `gnomonic · ${w.pipeline.faces.yaws.length} × ${w.pipeline.faces.fov_deg}°`,
    stat: (w) => `${w.stages.faces} faces`,
  },
  {
    id: 'embed',
    title: 'Embed',
    sub: (w) => modelLabel(w.pipeline.embed_model_used || w.pipeline.embed.model_name),
    stat: (w) => `${w.stages.faces} vectors`,
  },
  {
    id: 'select',
    title: 'Select',
    sub: (w) => `greedy cosine · τ ${w.pipeline.dedup.tau}`,
    stat: (w) => `${w.stages.absorbed} absorbed`,
  },
  {
    id: 'review',
    title: 'Review',
    sub: (w) => (w.stages.dropped ? `${w.stages.dropped} dropped` : 'no overrides'),
    stat: (w) => `${w.stages.kept} kept`,
    opens: 'review',
  },
  {
    id: 'export',
    title: 'Export',
    sub: () => 'zip · EXIF stamped',
    stat: (w) => `${w.stages.kept} frames`,
    opens: 'export',
  },
]

const LAYOUT_KEY = 'accio.pipeline.layout'

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

export default function Pipeline({ walk, onOpen }: {
  walk: WalkDetail
  onOpen: (what: 'review' | 'export') => void
}) {
  const [layout, setLayout] = useState<Record<string, Pos> | null>(savedLayout)
  const [dragging, setDragging] = useState<string | null>(null)
  const canvas = useRef<HTMLDivElement>(null)
  const drag = useRef<{ id: string; dx: number; dy: number; moved: boolean } | null>(null)
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
    drag.current = { id, dx: e.clientX - p.x, dy: e.clientY - p.y, moved: false }
    setDragging(id)
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }, [layout])

  // buttons check as well as the ref: if a pointerup is ever missed (capture
  // lost, pointer released off-window) the block would otherwise keep
  // following the cursor with no button held.
  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    if (!(e.buttons & 1)) { drag.current = null; setDragging(null); return }
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
    if (block.opens) onOpen(block.opens)
  }, [onOpen])

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
      {BLOCKS.map((b) => (
        <div
          key={b.id}
          className={`node${b.opens ? ' opens' : ''}${dragging === b.id ? ' dragging' : ''}`}
          style={{ left: layout[b.id].x, top: layout[b.id].y }}
          onPointerDown={(e) => onPointerDown(e, b.id)}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onClick={() => onClick(b)}
        >
          <div className="node-title">{b.title}</div>
          <div className="node-sub">{b.sub(walk)}</div>
          <div className="node-stat">{b.stat(walk)}</div>
        </div>
      ))}
    </div>
  )
}
