import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchCalibration, fetchJobs, fetchWalk, fetchWalks, postDecision,
         deleteWalk, fetchSegmentation, patchMeta, postRerun, retryWalk,
         serverSaid, STAGES, STAGE_OF } from './api'
import type { Calibration, Decision, Face, Group, Job, Pending, Section,
              Segmentation, WalkDetail, WalkSummary } from './api'
import CalibrationModal from './Calibration'
import Ingest from './Ingest'
import Loader from './Loader'
import Inspector, { backboneLabel } from './Inspector'
import Pipeline, { rerunLabel } from './Pipeline'
import Segments from './Segments'

// A member that only just cleared the threshold is the one worth a second
// look: it merged, but barely. The margin is relative to tau, which is per
// walk now, so a calibrated threshold moves the highlight with it.
const BORDERLINE_MARGIN = 0.015

/** Fold one edit into the staged set. A value put back to what the walk
    actually has is not a change, so it un-stages instead of piling up.
    `saved` is that section as it stands: the walk's params, or its metadata. */
function stage(pending: Pending, saved: Record<string, any>,
               section: Section, key: string, value: unknown): Pending {
  const next: Record<string, any> = { ...pending }
  const sec = { ...(next[section] ?? {}) }
  // an empty field and a field that was never set are the same thing
  const was = saved?.[key] ?? (typeof value === 'string' ? '' : undefined)
  if (JSON.stringify(was) === JSON.stringify(value)) delete sec[key]
  else sec[key] = value
  if (Object.keys(sec).length === 0) delete next[section]
  else next[section] = sec
  return next as Pending
}

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

/** What produced these groups. Reviewing them is judging a threshold, so the
    threshold and the backbone that scored against it belong on the page. */
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
  return (
    <div className={`group-row${dropped ? ' dropped' : ''}`}>
      <div className="group-head">
        <span>
          t={anchor.tSec.toFixed(1)}s y{anchor.yaw}
          {members.length > 0
            ? `, ${members.length + 1} candidates, sharpest picked, click to change`
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
          // sharpness only means anything next to its siblings: the absolute
          // number swings with how much texture is in view
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
              <img src={f.url} alt={`t=${f.tSec.toFixed(1)}s y${f.yaw}`} loading="lazy" />
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
  useEffect(() => { setOpen(null); setFocus(0) }, [walk.id])

  const groups = walk.groups

  // How many tiles fit across, read from the grid's own resolved template.
  // The expanded panel goes after the row it belongs to rather than straight
  // after its tile, so opening a group does not leave the row half empty.
  useEffect(() => {
    const el = gridRef.current
    if (!el) return
    const measure = () => setCols(Math.max(1, getComputedStyle(el)
      .gridTemplateColumns.split(' ').filter(Boolean).length))
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])

  // keyboard review: arrows move, enter opens, x drops, 1-9 swap the pick
  useEffect(() => {
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
      // legacy names (Right/Left/.../Return) for automation layers
      const key = {Right: 'ArrowRight', Left: 'ArrowLeft', Down: 'ArrowDown',
                   Up: 'ArrowUp', Return: 'Enter'}[e.key] ?? e.key
      if (key === 'ArrowRight') move(1)
      else if (key === 'ArrowLeft') move(-1)
      else if (key === 'ArrowDown') move(cols)
      else if (key === 'ArrowUp') move(-cols)
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
  }, [groups, focus, open, cols, onDecision])

  useEffect(() => {
    const tile = [...(gridRef.current?.children ?? [])]
      .filter((el) => el.classList.contains('tile'))[focus] as HTMLElement | undefined
    tile?.scrollIntoView({ block: 'nearest' })
  }, [focus])

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
          // a human disagreeing with the pipeline, not with the anchor: the
          // auto pick is already usually a member rather than the anchor
          const overridden = g.pick !== g.auto
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

/** Same row as a finished walk: the name, then a dot instead of the counts. */
// A run in flight. Errored jobs are not shown here: the walk's own row carries
// the failure and the context menu that can act on it.
// traceback.format_exc ends in a newline, so .pop() on the untrimmed string
// returns '' and the tooltip read "Failed: " with nothing after it.
function JobItem({ job, active, onClick }: {
  job: Job
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      className={`walk-item${active ? ' active' : ''}`}
      onClick={onClick}
      title={job.status === 'error'
        ? `Failed: ${job.error.trim().split('\n').pop()}`
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
  const [view, setView] = useState<'pipeline' | 'review' | 'masks'>('pipeline')
  const [seg, setSeg] = useState<Segmentation | null>(null)
  const [inspect, setInspect] = useState<string | null>(null)
  // settings edited but not yet applied, Railway style: the canvas marks what
  // they would re-run and nothing happens until Apply
  const [pending, setPending] = useState<Pending>({})
  const [applying, setApplying] = useState(false)
  const [refused, setRefused] = useState<string | null>(null)
  const [unreachable, setUnreachable] = useState<string | null>(null)
  // right-click on a walk: {id, x, y}, and `armed` once Delete is chosen
  const [menu, setMenu] = useState<
    { walk: WalkSummary; x: number; y: number; armed: boolean } | null>(null)
  const [calib, setCalib] = useState<Calibration | null>(null)
  const [showCalib, setShowCalib] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // the server stopped answering. Distinct from `error`, which is something
  // a request said: this is nobody saying anything, and it clears itself.
  const [offline, setOffline] = useState(false)
  const [exporting, setExporting] = useState(false)

  const exportUrl = selected
    ? `/api/walks/${encodeURIComponent(selected)}/export` : ''
  const job = jobs.find((j) => j.walkId === selected) ?? null
  // An errored job is finished, not in flight. Treating it as running hid the
  // failure notice, locked the inspector, kept Review and Export away, and
  // filtered the walk's own row out of the rail, which is where the context
  // menu carrying Retry and Delete lives. The one recovery path in the app
  // disappeared exactly when it was needed.
  const inFlight = (j: Job) => j.status === 'queued' || j.status === 'running'
  const running = job !== null && inFlight(job)
  const activeJobs = jobs.filter(inFlight)

  const refreshWalks = useCallback(
    () => fetchWalks().then((ws) => { setWalks(ws); setOffline(false) })
      .catch(() => setOffline(true)),
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
    setRefused(null)
    // a queued or running walk has no manifest yet: the canvas runs off the
    // job until it lands, so a 404 here is expected rather than an error
    setCalib(null)
    setShowCalib(false)
    setUnreachable(null)
    setSeg(null)
    // a walk that will not load is not a walk that is still loading: without
    // this the canvas sits on "Loading…" forever and never says why
    fetchWalk(selected)
      .then((w) => { setWalk(w); setUnreachable(null) })
      .catch((e) => {
        setWalk(null)
        setUnreachable(e instanceof Error ? e.message : String(e))
      })
    // measured on every run, so only walks made before it exists lack a record
    fetchCalibration(selected).then(setCalib).catch(() => setCalib(null))
    // 404s until the Segment stage has been turned on for this walk
    fetchSegmentation(selected).then(setSeg).catch(() => setSeg(null))
  }, [selected])

  useEffect(() => {
    if (!menu) return
    const close = () => setMenu(null)
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    // capture, so the menu closes even on a click the target swallows
    window.addEventListener('pointerdown', close, true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('scroll', close, true)
    return () => {
      window.removeEventListener('pointerdown', close, true)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', close, true)
    }
  }, [menu])

  const reload = useCallback((id: string) =>
    Promise.all([fetchWalk(id), fetchWalks()])
      .then(([w, ws]) => { setWalk(w); setWalks(ws) })
      .then(() => fetchCalibration(id).then(setCalib).catch(() => setCalib(null)))
      .then(() => fetchSegmentation(id).then(setSeg).catch(() => setSeg(null))),
    [])

  // Poll jobs always, fast while something is in flight and slowly otherwise:
  // a run can start in another tab, and stopping entirely means the rail never
  // notices it.
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running')
  const watching = useRef<number | null>(null)
  useEffect(() => {
    const t = setInterval(() => {
      fetchJobs().then((js) => {
        setJobs(js)
        const mine = js.find((j) => j.walkId === selected)
        // a job of ours finishing is the moment the frames on disk changed,
        // whether it was an ingest or a settings change
        if (mine && selected && mine.status !== 'done') watching.current = mine.id
        else if (mine?.status === 'done' && selected
                 && (watching.current === mine.id || !walk)) {
          watching.current = null
          reload(selected).catch(() => {})
        }
        const stillActive = js.some((j) => j.status === 'queued' || j.status === 'running')
        setOffline(false)
        if (!stillActive) refreshWalks()
      }).catch(() => setOffline(true))
    }, active ? 2000 : 8000)
    return () => clearInterval(t)
  }, [active, refreshWalks, reload, selected, walk])

  // The inspector is a fixed panel over the right of the canvas, so the canvas
  // has to give up the width or the pipeline centres under it. It opens during
  // a run too: watching a stage is the moment you most want to see what it is
  // set to. Editing is what a run rules out, and that is the panel's business.
  const inspecting = view === 'pipeline' && !ingesting && !!inspect && !!walk

  const edited = Object.values(pending).reduce(
    (n, sec) => n + Object.keys(sec ?? {}).length, 0)
  const dirtyFrom = STAGES.find((s) => Object.keys(pending).some(
    (k) => STAGE_OF[k] === s)) ?? null
  // metadata is staged the same way but re-runs nothing, so the pill says what
  // Apply would do rather than which stages it would cost
  const { meta: metaEdit, ...settings } = pending
  const willRun = Object.keys(settings).length > 0

  const onEdit = useCallback((section: Section, key: string, value: unknown) => {
    if (!walk) return
    const saved = section === 'meta'
      ? (walk.meta ?? {}) : (walk.pipeline as Record<string, any>)[section]
    setPending((prev) => stage(prev, saved, section, key, value))
  }, [walk])

  const apply = useCallback(() => {
    if (!selected) return
    setApplying(true)
    setRefused(null)
    // metadata first: it is a row update either way, and doing it before a
    // re-run means the reload afterwards already carries it
    Promise.resolve(metaEdit && patchMeta(selected, metaEdit))
      .then(() => willRun ? postRerun(selected, settings) : null)
      .then((res) => {
        setPending({})
        // a re-select is already done; anything heavier is now a job, and the
        // canvas follows it stage by stage until it lands
        if (res?.id != null) return fetchJobs().then(setJobs)
        return reload(selected)
      })
      // a refused change keeps the edits staged: the pill says why and the
      // walk on screen is still the one on disk
      .catch((e) => setRefused(e instanceof Error ? e.message : String(e)))
      .finally(() => setApplying(false))
  }, [selected, metaEdit, settings, willRun, reload])

  const onRetry = useCallback((id: string) => {
    setMenu(null)
    setError(null)
    retryWalk(id)
      .then(() => fetchJobs().then(setJobs))
      .catch((e) => setError(String(e)))
  }, [])

  const onDelete = useCallback((id: string) => {
    setMenu(null)
    setError(null)
    deleteWalk(id)
      .then(() => fetchWalks())
      .then((ws) => {
        setWalks(ws)
        setInspect(null)
        setSelected(ws.length ? ws[0].id : null)
        if (!ws.length) setWalk(null)
      })
      .catch((e) => setError(String(e)))
  }, [])

  const onExport = useCallback(() => {
    if (!selected) return
    setError(null)
    setExporting(true)
    fetch(exportUrl)
      .then(async (res) => {
        if (!res.ok) throw new Error(serverSaid(await res.text().catch(() => ''))
          || `${res.status} ${res.statusText}`)
        return res.blob()
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${selected}.zip`
        a.click()
        URL.revokeObjectURL(url)
      })
      .catch((e) => setError(`Export failed: ${e.message}`))
      .finally(() => setExporting(false))
  }, [selected, exportUrl])

  const onDecision = useCallback((d: Decision) => {
    if (!selected) return
    setError(null)
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
        {/* the poll stopped answering. Without this the app freezes in its
            last state and the running dot keeps pulsing at a dead server. */}
        {offline && <div className="offline">server not responding</div>}
        <div className="crumb">
          <span className="crumb-dim">accio</span>
          <span className="sep">/</span>
          {view !== 'pipeline' && !ingesting ? (
            <>
              <button className="crumb-link" onClick={() => setView('pipeline')}>
                {selected}
              </button>
              <span className="sep">/</span>
              <b>{view === 'masks' ? 'Masks' : 'Review'}</b>
            </>
          ) : (
            <b>{ingesting ? 'New Walk' : selected ?? 'No Walk'}</b>
          )}
        </div>
        <div className="top-actions">
          {!ingesting && selected && walk && walk.faces > 0 && !running && (
            <>
              <button
                className={`top-btn${view === 'review' ? ' active' : ''}`}
                onClick={() => setView(view === 'review' ? 'pipeline' : 'review')}
              >
                Review
              </button>
              {seg && (
                <button
                  className={`top-btn${view === 'masks' ? ' active' : ''}`}
                  onClick={() => setView(view === 'masks' ? 'pipeline' : 'masks')}
                >
                  Masks
                </button>
              )}
              {/* A bare download link gave no sign it had started, so an
                  export that reads and EXIF-stamps every kept frame looked
                  like a dead button and got clicked again, queueing another
                  zip. Fetching it means the wait is visible and a refusal
                  arrives as words rather than a downloaded error page. */}
              <button className="top-btn" disabled={exporting}
                      onClick={onExport}>
                {exporting ? 'Exporting…' : 'Export'}
              </button>
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
        {activeJobs.map((j) => (
          <JobItem key={j.id} job={j} active={j.walkId === selected}
                   onClick={() => { setIngesting(false); setSelected(j.walkId) }} />
        ))}
        {/* a walk being re-run is already in the rail above, as a job */}
        {walks?.filter((w) => !activeJobs.some((j) => j.walkId === w.id)).map((w) => (
          <button
            key={w.id}
            className={`walk-item${!ingesting && w.id === selected ? ' active' : ''}${
              menu?.walk.id === w.id ? ' menued' : ''}`}
            onClick={() => { setIngesting(false); setSelected(w.id) }}
            title={w.error ? `Failed at ${w.error.stage}: ${w.error.message}`
              : w.ready ? undefined : 'This run never finished'}
            onContextMenu={(e) => {
              e.preventDefault()
              setMenu({ walk: w, x: e.clientX, y: e.clientY, armed: false })
            }}
          >
            <span className="walk-id">{w.id}</span>
            {/* frames if it has any, and a mark if its last run broke: a walk
                can have both, when a re-run failed over a good earlier one */}
            {w.ready && (
              <span className="walk-stats">
                <span className="kept-n">{w.kept}</span>/{w.faces}
              </span>
            )}
            {(!w.ready || w.error) && <span className="job-dot error" />}
          </button>
        ))}
      </nav>
      {menu && (
        // right-click on a walk. Delete arms in place rather than opening a
        // dialog, so the thing being deleted stays under the cursor.
        <div
          className="ctx"
          style={{ left: menu.x, top: menu.y }}
          onPointerDown={(e) => e.stopPropagation()}
          onContextMenu={(e) => e.preventDefault()}
        >
          {!menu.armed && menu.walk.error && (
            <button className="ctx-item" onClick={() => onRetry(menu.walk.id)}>
              Retry from {menu.walk.error.stage}
            </button>
          )}
          {menu.armed ? (
            <>
              <div className="ctx-warn">
                Deletes {menu.walk.faces} frames, every review decision, and
                the uploaded video.
              </div>
              <button className="ctx-item danger"
                      onClick={() => onDelete(menu.walk.id)}>
                Delete permanently
              </button>
              <button className="ctx-item" onClick={() => setMenu(null)}>
                Cancel
              </button>
            </>
          ) : (
            <button className="ctx-item"
                    onClick={() => setMenu({ ...menu, armed: true })}>
              Delete Walk…
            </button>
          )}
        </div>
      )}
      <main className={`main${ingesting ? ' center' : ''}${
        inspecting ? ' inspected' : ''}${
        view === 'pipeline' && !ingesting ? ' pipeline' : ''}`}>
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
        {/* A failed request used to replace the whole work area and stay there
            until the browser was reloaded, losing your place in the grid. It
            is one request that failed, so it says so and gets out of the way. */}
        {error && (
          <div className="banner">
            <span>{error}</span>
            <button className="icon-btn" onClick={() => setError(null)}
                    aria-label="Dismiss">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        {ingesting && (
          <Ingest
            onSubmitted={(job) => {
              setJobs((js) => [job, ...js])
              setIngesting(false)
              setSelected(job.walkId)   // watch it run on its own canvas
            }}
          />
        )}
        {!ingesting && (walk || running) && view === 'pipeline' && (
          <>
            {walk?.error && !running && (
              <div className="notice">
                <b>{walk.error.stage}</b> failed on the last run: {walk.error.message}
                {walk.faces > 0 && <> The frames below are from the run before it.</>}
              </div>
            )}
            {edited > 0 && (
              <div className={`pill${refused ? ' refused' : ''}`}>
                <span className="pill-text">
                  {refused ?? (
                    <>
                      {edited} change{edited === 1 ? '' : 's'}{' '}
                      <span className="pill-dim">·</span>{' '}
                      {dirtyFrom
                        ? <>re-runs {rerunLabel(dirtyFrom)}</>
                        : 'saves metadata'}
                    </>
                  )}
                </span>
                <button className="pill-discard"
                        onClick={() => { setPending({}); setRefused(null) }}>
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
        {!ingesting && walk && seg && view === 'masks' && (
          <Segments seg={seg} />
        )}
        {!ingesting && walk && view === 'review' && (
          <>
            <header className="walk-header">
              <div className="header-row">
                <Funnel walk={walk} />
                <RunLine walk={walk} />
              </div>
              <MetaLine walk={walk} />
            </header>
            <KeptGrid walk={walk} onDecision={onDecision} />
          </>
        )}
        {!ingesting && !walk && !running && selected && (
          <Loader label={unreachable ? 'Could not open this walk' : 'Opening'}
                  sub={selected} error={unreachable} />
        )}
        {/* walks is null until the first list lands. Without this the app
            opens to an empty dotted field saying nothing, while the rail
            reads "Walks 0" as though that were the answer. */}
        {!ingesting && walks === null && <Loader label="Loading walks" />}
        {!ingesting && !selected && walks?.length === 0 && (
          <div className="empty">No walks yet. Add one with the + to the left.</div>
        )}
      </main>
      {inspecting && (
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
          {running && (
            <div className="insp-locked">
              {job?.stage ? `${job.stage} is running` : 'This walk is running'}
              {' '}· settings are read-only until it lands
            </div>
          )}
          <div className="insp-body">
            <Inspector stage={inspect} walk={walk} pending={pending} calib={calib}
                       locked={running} onEdit={onEdit}
                       onOpenReview={() => setView('review')}
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
