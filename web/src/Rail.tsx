import type { Job, WalkSummary } from './api'

// a run in flight; errored jobs are not listed here, the walk's own row carries the failure
// job.error ends in a newline (traceback.format_exc), so it is trimmed before taking the last line
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

export default function Rail({ walks, activeJobs, selected, ingesting, menued,
                               onSelect, onNew, onMenu }: {
  walks: WalkSummary[] | null
  activeJobs: Job[]
  selected: string | null
  ingesting: boolean
  menued: string | null        // the walk whose context menu is open
  onSelect: (id: string) => void
  onNew: () => void
  onMenu: (walk: WalkSummary, x: number, y: number) => void
}) {
  return (
    <nav className="rail">
      <div className="rail-label">
        <span>Walks</span>
        <span className="rail-count">{walks?.length ?? 0}</span>
        <button
          className={`icon-btn${ingesting ? ' active' : ''}`}
          onClick={onNew}
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
                 onClick={() => onSelect(j.walkId)} />
      ))}
      {/* a walk being re-run is already in the rail above, as a job */}
      {walks?.filter((w) => !activeJobs.some((j) => j.walkId === w.id)).map((w) => (
        <button
          key={w.id}
          className={`walk-item${!ingesting && w.id === selected ? ' active' : ''}${
            menued === w.id ? ' menued' : ''}`}
          onClick={() => onSelect(w.id)}
          title={w.error ? `Failed at ${w.error.stage}: ${w.error.message}`
            : w.ready ? undefined : 'This run never finished'}
          onContextMenu={(e) => {
            e.preventDefault()
            onMenu(w, e.clientX, e.clientY)
          }}
        >
          <span className="walk-id">{w.id}</span>
          {/* a walk can have both: frames from an earlier run and a failed re-run */}
          {w.ready && (
            <span className="walk-stats">
              <span className="kept-n">{w.kept}</span>/{w.faces}
            </span>
          )}
          {(!w.ready || w.error) && <span className="job-dot error" />}
        </button>
      ))}
    </nav>
  )
}
