import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchCalibration, fetchJobs, fetchWalk, fetchWalks, postDecision,
         postRerun } from './api'
import type { Calibration, Decision, Face, Group, Job, Pending, WalkDetail,
              WalkSummary } from './api'
import CalibrationModal from './Calibration'
import Ingest from './Ingest'
import Inspector from './Inspector'
import Pipeline, { rerunLabel } from './Pipeline'

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
          {dropped ? 'Restore Group' : 'Drop Group'}
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
                {isPick ? (f.idx === anchor.idx ? 'auto pick' : 'override')
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

/** Same row as a finished walk: the name, then a dot instead of the counts. */
function JobItem({ job, active, onClick }: {
  job: Job
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      className={`walk-item${active ? ' active' : ''}`}
      onClick={onClick}
      title={job.status === 'error' ? `Failed: ${job.error.split('\n').pop()}`
        : job.stage || job.status}
    >
      <span className="walk-id">{job.walkId}</span>
      <span className={`job-dot ${job.status}`} />
    </button>
  )
}

export default function App() {
  const [walks, setWalks] = useState<WalkSummary[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [walk, setWalk] = useState<WalkDetail | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [ingesting, setIngesting] = useState(false)
  const [view, setView] = useState<'pipeline' | 'review'>('pipeline')
  const [inspect, setInspect] = useState<string | null>(null)
  // settings edited but not yet applied, Railway style: the canvas marks what
  // they would re-run and nothing happens until Apply
  const [pending, setPending] = useState<Pending>({})
  const [applying, setApplying] = useState(false)
  const [calib, setCalib] = useState<Calibration | null>(null)
  const [showCalib, setShowCalib] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const exportUrl = selected
    ? `/api/walks/${encodeURIComponent(selected)}/export` : ''
  const job = jobs.find((j) => j.walkId === selected) ?? null
  const running = job !== null && job.status !== 'done'

  const refreshWalks = useCallback(
    () => fetchWalks().then(setWalks).catch((e) => setError(String(e))),
    [])

  useEffect(() => {
    // jobs too: a run started in another tab (or before a reload) should still
    // show up in the rail and keep its canvas live
    fetchJobs().then(setJobs).catch(() => {})
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
    setView('pipeline')
    setInspect(null)
    setPending({})
    // a queued or running walk has no manifest yet: the canvas runs off the
    // job until it lands, so a 404 here is expected rather than an error
    setCalib(null)
    setShowCalib(false)
    fetchWalk(selected).then(setWalk).catch(() => setWalk(null))
    // measured on every run, so only walks made before it exists lack a record
    fetchCalibration(selected).then(setCalib).catch(() => setCalib(null))
  }, [selected])

  // Poll jobs always, fast while something is in flight and slowly otherwise:
  // a run can start in another tab, and stopping entirely means the rail never
  // notices it.
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running')
  useEffect(() => {
    const t = setInterval(() => {
      fetchJobs().then((js) => {
        setJobs(js)
        const mine = js.find((j) => j.walkId === selected)
        if (mine?.status === 'done' && !walk && selected) {
          fetchWalk(selected).then(setWalk).catch(() => {})
        }
        const stillActive = js.some((j) => j.status === 'queued' || j.status === 'running')
        if (!stillActive) refreshWalks()
      }).catch(() => {})
    }, active ? 2000 : 8000)
    return () => clearInterval(t)
  }, [active, refreshWalks, selected, walk])

  const dirtyFrom = pending.tau != null || pending.rule ? 'select' : null

  const apply = useCallback(() => {
    if (!selected) return
    setApplying(true)
    postRerun(selected, pending)
      .then(() => Promise.all([fetchWalk(selected), fetchWalks()]))
      .then(([w, ws]) => { setWalk(w); setWalks(ws); setPending({}) })
      .then(() => fetchCalibration(selected).then(setCalib).catch(() => {}))
      .catch((e) => setError(String(e)))
      .finally(() => setApplying(false))
  }, [selected, pending])

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
        <div className="crumb">
          <span className="crumb-dim">accio</span>
          <span className="sep">/</span>
          {view === 'review' && !ingesting ? (
            <>
              <button className="crumb-link" onClick={() => setView('pipeline')}>
                {selected}
              </button>
              <span className="sep">/</span>
              <b>Review</b>
            </>
          ) : (
            <b>{ingesting ? 'New Walk' : selected ?? 'No Walk'}</b>
          )}
        </div>
        <div className="top-actions">
          {!ingesting && selected && walk && !running && (
            <>
              <button
                className={`top-btn${view === 'review' ? ' active' : ''}`}
                onClick={() => setView(view === 'review' ? 'pipeline' : 'review')}
              >
                Review
              </button>
              <a className="top-btn" href={exportUrl} download>Export</a>
            </>
          )}
        </div>
      </header>
      <div className="app">
      <nav className="rail">
        <div className="rail-label">
          <span>Walks</span>
          <span className="rail-count">{walks?.length ?? 0}</span>
          <button
            className={`icon-btn${ingesting ? ' active' : ''}`}
            onClick={() => setIngesting(true)}
            title="New Walk"
            aria-label="New Walk"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        </div>
        {jobs.filter((j) => j.status !== 'done').map((j) => (
          <JobItem key={j.id} job={j} active={j.walkId === selected}
                   onClick={() => { setIngesting(false); setSelected(j.walkId) }} />
        ))}
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
      <main className={`main${ingesting ? ' center' : ''}`}>
        {/* Zed's dotted backdrop, their markup rather than a CSS gradient:
            an 8 px pattern with an r=0.75 circle in blue-300 at 60%. */}
        <svg className="dots">
          <defs>
            <pattern id="dot" width="8" height="8" patternUnits="userSpaceOnUse">
              <circle cx="4" cy="4" r="0.75" fill="currentColor" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#dot)" />
        </svg>
        {error && <div className="error">{error}</div>}
        {!error && ingesting && (
          <Ingest
            onSubmitted={(job) => {
              setJobs((js) => [job, ...js])
              setIngesting(false)
              setSelected(job.walkId)   // watch it run on its own canvas
            }}
          />
        )}
        {!error && !ingesting && (walk || running) && view === 'pipeline' && (
          <>
            {dirtyFrom && (
              <div className="pill">
                <span className="pill-text">
                  1 change <span className="pill-dim">·</span> re-runs{' '}
                  {rerunLabel(dirtyFrom)}
                </span>
                <button className="pill-discard" onClick={() => setPending({})}>
                  Discard
                </button>
                <button className="pill-apply" onClick={apply} disabled={applying}>
                  {applying ? 'Applying…' : 'Apply'}
                </button>
              </div>
            )}
            <Pipeline walk={walk} job={job} selected={inspect}
                      dirtyFrom={dirtyFrom} onSelect={setInspect} />
          </>
        )}
        {!error && !ingesting && walk && view === 'review' && (
          <>
            <header className="walk-header">
              <MetaLine walk={walk} />
              <Funnel walk={walk} />
            </header>
            <KeptGrid walk={walk} onDecision={onDecision} />
          </>
        )}
        {!error && !ingesting && !walk && !running && selected && (
          <div className="empty">Loading…</div>
        )}
        {!error && !ingesting && !selected && walks?.length === 0 && (
          <div className="empty">No walks yet. Add one with the + above.</div>
        )}
      </main>
      {view === 'pipeline' && !ingesting && inspect && walk && !running && (
        <aside className="inspector">
          <div className="inspector-head">
            <span className="inspector-title">{inspect}</span>
            <button
              className="icon-btn"
              onClick={() => setInspect(null)}
              title="Close"
              aria-label="Close"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="insp-body">
            <Inspector stage={inspect} walk={walk} pending={pending} calib={calib}
                       onEdit={setPending} onOpenReview={() => setView('review')}
                       onShowCalibration={() => setShowCalib(true)} />
          </div>
        </aside>
      )}
      </div>
      {showCalib && calib && selected && (
        <CalibrationModal walkId={selected} calib={calib}
                          onClose={() => setShowCalib(false)} />
      )}
    </div>
  )
}
