import { useCallback, useEffect, useState } from 'react'
import { deleteWalk, exportZip, fetchJobs, fetchWalks, patchMeta, postRerun,
         retryWalk, STAGES, STAGE_OF } from './api'
import type { Decision, Pending, Section, WalkDetail, WalkSummary } from './api'
import CalibrationModal from './Calibration'
import { inFlight, message, useJobs, useWalk } from './hooks'
import Ingest from './Ingest'
import Inspector from './Inspector'
import type { Edit } from './Inspector'
import { rerunLabel } from './labels'
import Loader from './Loader'
import Pipeline from './Pipeline'
import Rail from './Rail'
import Review from './Review'
import Segments from './Segments'

type Sec<S extends Section> = Required<Pending>[S]

// what a section's values are compared against: the saved metadata, or the settings the walk ran with
function savedOf(walk: WalkDetail, section: Section) {
  return section === 'meta' ? walk.meta : walk.pipeline[section]
}

/** Fold one edit into the staged set; a value put back to what is saved un-stages it. */
function stage<S extends Section, K extends keyof Sec<S>>(
  pending: Pending, walk: WalkDetail, section: S, key: K, value: Sec<S>[K]): Pending {
  const saved: { [P in K]?: unknown } | null = savedOf(walk, section)
  const sec: Sec<S> = Object.assign({}, pending[section])
  // an empty field and a field that was never set are the same thing
  const was = saved?.[key] ?? (typeof value === 'string' ? '' : undefined)
  if (JSON.stringify(was) === JSON.stringify(value)) delete sec[key]
  else sec[key] = value
  const next = { ...pending }
  if (Object.keys(sec).length === 0) delete next[section]
  else next[section] = sec
  return next
}

export default function App() {
  const [walks, setWalks] = useState<WalkSummary[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [ingesting, setIngesting] = useState(false)
  const [view, setView] = useState<'pipeline' | 'review' | 'masks'>('pipeline')
  const [inspect, setInspect] = useState<string | null>(null)
  // edits not yet applied; nothing runs until Apply
  const [pending, setPending] = useState<Pending>({})
  const [applying, setApplying] = useState(false)
  const [refused, setRefused] = useState<string | null>(null)
  // right-click on a walk: {id, x, y}, and `armed` once Delete is chosen
  const [menu, setMenu] = useState<
    { walk: WalkSummary; x: number; y: number; armed: boolean } | null>(null)
  const [showCalib, setShowCalib] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)

  const { walk, setWalk, calib, seg, unreachable, reload, decide } =
    useWalk(selected, setWalks, setError)
  const { jobs, setJobs, offline } = useJobs(
    selected,
    (id) => { reload(id).catch(() => {}) },
    () => fetchWalks().then(setWalks))

  const job = jobs.find((j) => j.walkId === selected) ?? null
  // an errored job is finished, not in flight
  const running = job !== null && inFlight(job)
  const activeJobs = jobs.filter(inFlight)

  useEffect(() => {
    fetchWalks()
      .then((ws) => {
        setWalks(ws)
        if (ws.length > 0) setSelected(ws[0].id)
      })
      .catch((e) => setError(message(e)))
  }, [])

  useEffect(() => {
    if (!selected) return
    setView('pipeline')
    setInspect(null)
    setPending({})
    setRefused(null)
    setShowCalib(false)
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

  // the inspector overlays the right of the canvas, so the canvas gives up the width
  const inspecting = view === 'pipeline' && !ingesting && !!inspect && !!walk

  const edited = Object.values(pending).reduce(
    (n, sec) => n + Object.keys(sec ?? {}).length, 0)
  const dirtyFrom = STAGES.find((s) => Object.keys(pending).some(
    (k) => STAGE_OF[k] === s)) ?? null
  // metadata edits re-run nothing
  const { meta: metaEdit, ...settings } = pending
  const willRun = Object.keys(settings).length > 0

  const onEdit: Edit = (section, key, value) => {
    if (!walk) return
    setPending((prev) => stage(prev, walk, section, key, value))
  }

  const apply = () => {
    if (!selected) return
    setApplying(true)
    setRefused(null)
    // metadata first, so the reload after a re-run already carries it
    Promise.resolve(metaEdit && patchMeta(selected, metaEdit))
      .then(() => willRun ? postRerun(selected, settings) : null)
      .then((res) => {
        setPending({})
        // no job id means the change applied immediately
        if (res?.id != null) return fetchJobs().then(setJobs)
        return reload(selected)
      })
      // a refused change keeps the edits staged
      .catch((e) => setRefused(message(e)))
      .finally(() => setApplying(false))
  }

  const onRetry = (id: string) => {
    setMenu(null)
    setError(null)
    retryWalk(id)
      .then(() => fetchJobs().then(setJobs))
      .catch((e) => setError(message(e)))
  }

  const onDelete = (id: string) => {
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
      .catch((e) => setError(message(e)))
  }

  const onExport = () => {
    if (!selected) return
    setError(null)
    setExporting(true)
    exportZip(selected)
      .then((blob) => {
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${selected}.zip`
        document.body.appendChild(a)
        a.click()
        // revoking in the same tick cancels the download in some browsers
        setTimeout(() => { URL.revokeObjectURL(url); a.remove() }, 0)
      })
      .catch((e) => setError(`Export failed: ${message(e)}`))
      .finally(() => setExporting(false))
  }

  // stable: a dep of the review grid's key handler
  const onDecision = useCallback((d: Decision) => {
    setError(null)
    decide(d).catch((e) => setError(message(e)))
  }, [decide])

  const selectWalk = (id: string) => { setIngesting(false); setSelected(id) }

  return (
    <div className="frame">
      <header className="topbar">
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
              {/* fetched rather than linked, so the wait is visible and a refusal arrives as text */}
              <button className="top-btn" disabled={exporting}
                      onClick={onExport}>
                {exporting ? 'Exporting…' : 'Export'}
              </button>
            </>
          )}
        </div>
      </header>
      <div className="app">
      <Rail walks={walks} activeJobs={activeJobs} selected={selected}
            ingesting={ingesting} menued={menu?.walk.id ?? null}
            onSelect={selectWalk} onNew={() => setIngesting(true)}
            onMenu={(w, x, y) => setMenu({ walk: w, x, y, armed: false })} />
      {menu && (
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
        {/* Zed's dotted backdrop: 8 px pattern, r=0.75 circle */}
        <svg className="dots">
          <defs>
            <pattern id="dot" width="8" height="8" patternUnits="userSpaceOnUse">
              <circle cx="4" cy="4" r="0.75" fill="currentColor" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#dot)" />
        </svg>
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
          <Review walk={walk} onDecision={onDecision} />
        )}
        {!ingesting && !walk && !running && selected && (
          <Loader label={unreachable ? 'Could not open this walk' : 'Opening'}
                  sub={selected} error={unreachable} />
        )}
        {/* walks is null until the first list lands */}
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
