import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJobs, fetchWalk, fetchWalks, postDecision } from './api'
import type { Decision, Face, Group, Job, WalkDetail, WalkSummary } from './api'
import Ingest from './Ingest'

// cosines this close to tau (0.94) deserve a second look
const BORDERLINE = 0.955

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

function GroupRow({ group, onDecision }: {
  group: Group
  onDecision: (d: Decision) => void
}) {
  const { anchor, members, pick, dropped } = group
  const candidates = [anchor, ...members]
  return (
    <div className={`group-row${dropped ? ' dropped' : ''}`}>
      <div className="group-head">
        <span>
          t={anchor.tSec.toFixed(1)}s y{anchor.yaw}
          {members.length > 0
            ? `, ${members.length + 1} candidates, click to change the pick`
            : ', no duplicates absorbed'}
        </span>
        <button
          className={`drop-btn${dropped ? ' restore' : ''}`}
          onClick={() =>
            onDecision({ anchorIdx: anchor.idx, action: dropped ? 'restore' : 'drop' })}
        >
          {dropped ? 'restore group' : 'drop group'}
        </button>
      </div>
      <div className="strip">
        {candidates.map((f, n) => {
          const isPick = f.idx === pick && !dropped
          return (
            <button
              key={f.idx}
              className={`member${isPick ? ' is-pick' : ''}`}
              disabled={dropped}
              onClick={() => {
                if (!isPick) onDecision({ anchorIdx: anchor.idx, action: 'pick', pickIdx: f.idx })
              }}
            >
              {n < 9 && <span className="key">{n + 1}</span>}
              <img src={f.url} alt={`t=${f.tSec.toFixed(1)}s y${f.yaw}`} loading="lazy" />
              <span
                className={`cos${isPick ? ' pick-label' : ''}${
                  !isPick && f.cosine !== null && f.cosine < BORDERLINE ? ' borderline' : ''
                }`}
              >
                {isPick ? (f.idx === anchor.idx ? 'pick (auto)' : 'pick (override)')
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
  const gridRef = useRef<HTMLDivElement>(null)
  useEffect(() => { setOpen(null); setFocus(0) }, [walk.id])

  const groups = walk.groups

  // keyboard review: arrows move, enter opens, x drops, 1-9 swap the pick
  useEffect(() => {
    const tiles = () =>
      [...(gridRef.current?.children ?? [])].filter((el) =>
        el.classList.contains('tile')) as HTMLElement[]
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const t = e.target as HTMLElement | null
      if (t?.closest?.('input, textarea, select')) return
      if (!groups.length) return
      const g = groups[focus]
      const move = (d: number) => {
        e.preventDefault()
        setFocus((f) => Math.min(groups.length - 1, Math.max(0, f + d)))
      }
      const cols = () => {
        const ts = tiles()
        let n = 0
        for (const el of ts) {
          if (el.offsetTop !== ts[0].offsetTop) break
          n++
        }
        return Math.max(1, n)
      }
      // legacy names (Right/Left/.../Return) for automation layers
      const key = {Right: 'ArrowRight', Left: 'ArrowLeft', Down: 'ArrowDown',
                   Up: 'ArrowUp', Return: 'Enter'}[e.key] ?? e.key
      if (key === 'ArrowRight') move(1)
      else if (key === 'ArrowLeft') move(-1)
      else if (key === 'ArrowDown') move(cols())
      else if (key === 'ArrowUp') move(-cols())
      else if (key === 'Enter' || key === ' ') {
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
  }, [groups, focus, open, onDecision])

  useEffect(() => {
    const tile = [...(gridRef.current?.children ?? [])]
      .filter((el) => el.classList.contains('tile'))[focus] as HTMLElement | undefined
    tile?.scrollIntoView({ block: 'nearest' })
  }, [focus])

  return (
    <>
      <div className="section-title">
        picks, walk order · arrows move · enter opens · x drops · 1-9 swap pick
      </div>
      <div className="grid" ref={gridRef}>
        {walk.groups.map((g, i) => {
          const f = pickFace(g)
          const isOpen = open === g.anchor.idx
          const overridden = g.pick !== g.anchor.idx
          const tile = (
            <button
              key={g.anchor.idx}
              className={`tile kept${isOpen ? ' open' : ''}${g.dropped ? ' dropped' : ''}${
                i === focus ? ' focused' : ''}`}
              style={{ animationDelay: `${Math.min(i * 12, 360)}ms` }}
              onClick={() => { setFocus(i); setOpen(isOpen ? null : g.anchor.idx) }}
              title={`pano ${f.panoIdx}, yaw ${f.yaw}`}
            >
              <img src={f.url} alt={`pick t=${f.tSec}s`} loading="lazy" />
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
          return isOpen
            ? [tile, <GroupRow key={`row-${g.anchor.idx}`} group={g} onDecision={onDecision} />]
            : tile
        })}
      </div>
    </>
  )
}

function JobItem({ job }: { job: Job }) {
  return (
    <div className={`job-item ${job.status}`}>
      <span className="job-dot" />
      <span className="job-text">
        <span className="job-id">{job.walkId}</span>
        <span className="job-stage">
          {job.status === 'error' ? 'failed' : job.stage || job.status}
        </span>
      </span>
    </div>
  )
}

export default function App() {
  const [walks, setWalks] = useState<WalkSummary[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [walk, setWalk] = useState<WalkDetail | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [ingesting, setIngesting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshWalks = useCallback(
    () => fetchWalks().then(setWalks).catch((e) => setError(String(e))),
    [])

  useEffect(() => {
    fetchWalks()
      .then((ws) => {
        setWalks(ws)
        if (ws.length > 0) setSelected(ws[0].id)
      })
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (!selected) return
    setWalk(null)
    fetchWalk(selected).then(setWalk).catch((e) => setError(String(e)))
  }, [selected])

  // poll jobs while any are queued or running
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running')
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => {
      fetchJobs().then((js) => {
        setJobs(js)
        const stillActive = js.some((j) => j.status === 'queued' || j.status === 'running')
        if (!stillActive) refreshWalks()
      }).catch(() => {})
    }, 2000)
    return () => clearInterval(t)
  }, [active, refreshWalks])

  const onDecision = useCallback((d: Decision) => {
    if (!selected) return
    postDecision(selected, d)
      .then(() => Promise.all([fetchWalk(selected), fetchWalks()]))
      .then(([w, ws]) => {
        setWalk(w)
        setWalks(ws)
      })
      .catch((e) => setError(String(e)))
  }, [selected])

  return (
    <div className="frame">
      <header className="topbar">
        <div className="tabs">
          <button
            className={`tab${!ingesting ? ' active' : ''}`}
            onClick={() => setIngesting(false)}
          >
            Review
          </button>
          <button
            className={`tab${ingesting ? ' active' : ''}`}
            onClick={() => setIngesting(true)}
          >
            New Walk
          </button>
        </div>
        <div className="crumb">
          accio <span className="sep">/</span>{' '}
          <b>{ingesting ? 'new walk' : selected ?? 'no walk'}</b>
        </div>
        {!ingesting && selected && walk && (
          <div className="top-actions">
            <a
              className="top-btn"
              href={`/api/walks/${encodeURIComponent(selected)}/export`}
              download
            >
              Export
            </a>
          </div>
        )}
      </header>
      <div className="app">
      <nav className="rail">
        <div className="rail-label">Walks</div>
        {jobs.filter((j) => j.status !== 'done').map((j) => <JobItem key={j.id} job={j} />)}
        {walks?.map((w) => (
          <button
            key={w.id}
            className={`walk-item${!ingesting && w.id === selected ? ' active' : ''}`}
            onClick={() => { setIngesting(false); setSelected(w.id) }}
          >
            <span className="walk-id">{w.id}</span>
            <span className="walk-stats">
              <span className="kept-n">{w.kept}</span>/{w.faces}
            </span>
          </button>
        ))}
      </nav>
      <main className="main">
        {error && <div className="error">{error}</div>}
        {!error && ingesting && (
          <Ingest
            onSubmitted={(job) => {
              setJobs((js) => [job, ...js])
              setIngesting(false)
            }}
          />
        )}
        {!error && !ingesting && walk && (
          <>
            <header className="walk-header">
              <h1 className="walk-title">{walk.id}</h1>
              <MetaLine walk={walk} />
              <Funnel walk={walk} />
            </header>
            <KeptGrid walk={walk} onDecision={onDecision} />
          </>
        )}
        {!error && !ingesting && !walk && selected && <div className="empty">loading…</div>}
        {!error && !ingesting && !selected && walks?.length === 0 && (
          <div className="empty">no walks yet, click “+ new walk”</div>
        )}
      </main>
      </div>
    </div>
  )
}
