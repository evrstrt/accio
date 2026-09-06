import { useEffect, useRef, useState } from 'react'
import { thumb } from './api'
import type { Decision, Face, Group, WalkDetail } from './api'
import { backboneLabel } from './labels'

// members within this much of tau get the borderline highlight
const BORDERLINE_MARGIN = 0.015

const tileAt = (grid: HTMLElement | null, i: number) =>
  [...(grid?.children ?? [])]
    .filter((el) => el.classList.contains('tile'))[i] as HTMLElement | undefined

function pickFace(g: Group): Face {
  return g.pick === g.anchor.idx
    ? g.anchor
    : g.members.find((m) => m.idx === g.pick) ?? g.anchor
}

function Funnel({ walk }: { walk: WalkDetail }) {
  const kept = walk.groups.filter((g) => !g.dropped).length
  const dropped = walk.groups.length - kept
  const absorbed = walk.faces - walk.groups.length
  return (
    <div className="funnel">
      <span><span className="n">{walk.faces}</span> <span className="label">faces</span></span>
      <span className="arrow">→</span>
      <span><span className="n">{absorbed}</span> <span className="label">absorbed</span></span>
      {dropped > 0 && (
        <>
          <span className="arrow">→</span>
          <span><span className="n drop-n">{dropped}</span> <span className="label">dropped</span></span>
        </>
      )}
      <span className="arrow">→</span>
      <span><span className="n kept-n">{kept}</span> <span className="label">kept</span></span>
    </div>
  )
}

function MetaLine({ walk }: { walk: WalkDetail }) {
  const m = walk.meta
  if (!m) return null
  const bits = [
    m.site && `site ${m.site}`,
    m.building && `building ${m.building}`,
    m.stage,
    m.operator,
    m.mountHeightCm != null && `${m.mountHeightCm} cm`,
    m.shotDate,
  ].filter(Boolean)
  if (bits.length === 0) return null
  return <div className="meta-line">{bits.join(' · ')}</div>
}

function RunLine({ walk }: { walk: WalkDetail }) {
  const { dedup, embed } = walk.pipeline
  return (
    <div className="run-line">
      τ {dedup.tau}{dedup.rule === 'calibrated' ? ' calibrated' : ''}
      <span className="sep">·</span>
      {backboneLabel(embed.model_name)}
    </div>
  )
}

function GroupRow({ group, tau, onDecision }: {
  group: Group
  tau: number
  onDecision: (d: Decision) => void
}) {
  const { anchor, members, pick, dropped } = group
  const candidates = [anchor, ...members]
  const best = Math.max(...candidates.map((f) => f.sharpness)) || 1
  const shown = candidates.find((f) => f.idx === pick) ?? anchor
  return (
    <div className={`group-row${dropped ? ' dropped' : ''}`}>
      <div className="group-head">
        {/* names the same frame as the tile that opened it */}
        <span>
          y{anchor.yaw}, from t={anchor.tSec.toFixed(1)}s
          {members.length > 0 ? <>
            {`, ${members.length + 1} candidates, exporting `}
            <b>t={shown.tSec.toFixed(1)}s</b>
            {shown.idx === group.auto ? ' as the sharpest' : ' by your override'}
          </> : ', no duplicates absorbed'}
        </span>
        <button
          className={`drop-btn${dropped ? ' restore' : ''}`}
          onClick={() =>
            onDecision({ anchorIdx: anchor.idx, action: dropped ? 'restore' : 'drop' })}
        >
          {dropped ? 'Restore Group' : 'Drop Group'}
        </button>
      </div>
      <div className="strip">
        {candidates.map((f, n) => {
          const isPick = f.idx === pick && !dropped
          // sharpness is only comparable within a group; the absolute number depends on the texture in view
          const rel = f.sharpness / best
          return (
            <button
              key={f.idx}
              className={`member${isPick ? ' is-pick' : ''}`}
              disabled={dropped}
              onClick={() => {
                if (!isPick) onDecision({ anchorIdx: anchor.idx, action: 'pick', pickIdx: f.idx })
              }}
              title={`sharpness ${f.sharpness.toFixed(0)}`}
            >
              {n < 9 && <span className="key">{n + 1}</span>}
              <img src={thumb(f.url)} alt={`t=${f.tSec.toFixed(1)}s y${f.yaw}`}
                   loading="lazy" />
              {candidates.length > 1 && (
                <span className="sharp" aria-hidden>
                  <span style={{ width: `${Math.round(100 * rel)}%` }} />
                </span>
              )}
              <span
                className={`cos${isPick ? ' pick-label' : ''}${
                  !isPick && f.cosine !== null
                    && f.cosine < tau + BORDERLINE_MARGIN ? ' borderline' : ''
                }`}
              >
                {isPick ? (f.idx === group.auto ? 'sharpest' : 'override')
                  : f.cosine === null ? `t=${f.tSec.toFixed(1)}s` : f.cosine.toFixed(3)}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function KeptGrid({ walk, onDecision }: {
  walk: WalkDetail
  onDecision: (d: Decision) => void
}) {
  const [open, setOpen] = useState<number | null>(null)
  const [focus, setFocus] = useState(0)
  const [cols, setCols] = useState(1)
  const gridRef = useRef<HTMLDivElement>(null)

  const groups = walk.groups
  // a reload can shrink the groups under the stored index
  const at = Math.min(focus, Math.max(0, groups.length - 1))

  // column count from the resolved grid template; the open panel goes after its row
  useEffect(() => {
    const el = gridRef.current
    if (!el) return
    const measure = () => setCols(Math.max(1, getComputedStyle(el)
      .gridTemplateColumns.split(' ').filter(Boolean).length))
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // keyboard review: arrows move, enter opens, x drops, 1-9 swap the pick
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (!groups.length) return
      const t = e.target as HTMLElement | null
      // keys from outside the grid belong to whatever has focus there; body
      // counts as inside, since Safari does not focus a clicked button
      const unfocused = !t || t === document.body
      if (!unfocused && !gridRef.current?.contains(t)) return
      const g = groups[at]
      const move = (d: number) => {
        e.preventDefault()
        setFocus(Math.min(groups.length - 1, Math.max(0, at + d)))
      }
      // legacy names (Right/Left/.../Return) for automation layers
      const key = {Right: 'ArrowRight', Left: 'ArrowLeft', Down: 'ArrowDown',
                   Up: 'ArrowUp', Return: 'Enter'}[e.key] ?? e.key
      if (key === 'ArrowRight') move(1)
      else if (key === 'ArrowLeft') move(-1)
      else if (key === 'ArrowDown') move(cols)
      else if (key === 'ArrowUp') move(-cols)
      else if (key === 'Enter' || key === ' ') {
        // the open row's own buttons keep their native activation
        if (!unfocused && !t.closest('.tile')) return
        e.preventDefault()
        setOpen((o) => (o === g.anchor.idx ? null : g.anchor.idx))
      } else if (key === 'Escape') setOpen(null)
      else if (e.key === 'x') {
        onDecision({ anchorIdx: g.anchor.idx, action: g.dropped ? 'restore' : 'drop' })
      } else if (/^[1-9]$/.test(e.key) && open === g.anchor.idx && !g.dropped) {
        const target = [g.anchor, ...g.members][Number(e.key) - 1]
        if (target && target.idx !== g.pick) {
          onDecision({ anchorIdx: g.anchor.idx, action: 'pick', pickIdx: target.idx })
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [groups, at, open, cols, onDecision])

  // drive DOM focus so the focused tile and the highlighted one agree
  useEffect(() => {
    const tile = tileAt(gridRef.current, at)
    if (!tile || tile === document.activeElement) return
    // only move focus when it is already in the grid
    const inGrid = gridRef.current?.contains(document.activeElement)
    if (inGrid) tile.focus({ preventScroll: true })
    tile.scrollIntoView({ block: 'nearest' })
  }, [at])

  // the last tile on the same visual row as the open one
  const openAt = open === null ? -1 : groups.findIndex((g) => g.anchor.idx === open)
  const panelAfter = openAt < 0 ? -1
    : Math.min(groups.length - 1, Math.floor(openAt / cols) * cols + cols - 1)

  return (
    <>
      <div className="grid" ref={gridRef}>
        {walk.groups.map((g, i) => {
          const f = pickFace(g)
          const isOpen = open === g.anchor.idx
          // compared to the auto pick, not the anchor: the auto pick is often a member
          const overridden = g.pick !== g.auto
          const tile = (
            <button
              key={g.anchor.idx}
              className={`tile kept${isOpen ? ' open' : ''}${g.dropped ? ' dropped' : ''}`}
              style={{ animationDelay: `${Math.min(i * 12, 360)}ms` }}
              onClick={() => { setFocus(i); setOpen(isOpen ? null : g.anchor.idx) }}
              onFocus={() => setFocus(i)}
              // roving tabindex: one tab stop for the whole grid
              tabIndex={i === at ? 0 : -1}
              title={`pano ${f.panoIdx}, yaw ${f.yaw}`}
            >
              <img src={thumb(f.url)} alt={`pick t=${f.tSec}s`} loading="lazy" />
              {g.dropped && <span className="badge">dropped</span>}
              {!g.dropped && overridden && <span className="badge override">swapped</span>}
              {!g.dropped && !overridden && g.members.length > 0 && (
                <span className="badge">+{g.members.length}</span>
              )}
              <span className="meta">
                <span>t={f.tSec.toFixed(1)}s</span>
                <span>y{f.yaw}</span>
              </span>
            </button>
          )
          return i === panelAfter
            ? [tile, <GroupRow key={`row-${open}`} group={groups[openAt]}
                                tau={walk.pipeline.dedup.tau}
                                onDecision={onDecision} />]
            : tile
        })}
      </div>
    </>
  )
}

export default function Review({ walk, onDecision }: {
  walk: WalkDetail
  onDecision: (d: Decision) => void
}) {
  return (
    <>
      <header className="walk-header">
        <div className="header-row">
          <Funnel walk={walk} />
          <RunLine walk={walk} />
        </div>
        <MetaLine walk={walk} />
        <div className="keys-hint">
          <kbd>←</kbd><kbd>→</kbd> move
          <span className="sep">·</span>
          <kbd>enter</kbd> open
          <span className="sep">·</span>
          <kbd>x</kbd> drop
          <span className="sep">·</span>
          <kbd>1</kbd>–<kbd>9</kbd> pick
        </div>
      </header>
      <KeptGrid key={walk.id} walk={walk} onDecision={onDecision} />
    </>
  )
}
