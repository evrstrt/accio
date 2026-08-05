import { useRef, useState } from 'react'
import { postIngest } from './api'
import type { Job } from './api'

// The capture metadata that makes the dataset balanceable later. Only what
// the file cannot answer is asked: the camera model and the moment of the
// walk are read out of the .insv at ingest. A value typed here still wins,
// for a walk uploaded long after it was shot off a camera clock nobody set.
const FIELDS = [
  { name: 'site', label: 'Site', placeholder: 'GCMR' },
  { name: 'building', label: 'Building', placeholder: 'tower 2' },
  { name: 'floor', label: 'Floor', placeholder: '4, or basement' },
  { name: 'stage', label: 'Stage', placeholder: 'bare-rcc' },
  { name: 'operator', label: 'Operator', placeholder: 'who walked it' },
  { name: 'mount_height_cm', label: 'Mount Height (cm)', placeholder: '175', type: 'number' },
  { name: 'shot_date', label: 'Shot Date', placeholder: '', type: 'date' },
  { name: 'shot_time', label: 'Shot Time', placeholder: '', type: 'time' },
] as const

export default function Ingest({ onSubmitted }: { onSubmitted: (job: Job) => void }) {
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const formRef = useRef<HTMLFormElement>(null)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!files.length) {
      setError('Drop an .insv first')
      return
    }
    const form = new FormData(formRef.current!)
    files.forEach((f) => form.append('files', f))
    setBusy(true)
    setProgress(0)
    setError(null)
    postIngest(form, setProgress)
      .then(onSubmitted)
      .catch((err) => setError(String(err)))
      .finally(() => setBusy(false))
  }

  const gb = files.reduce((n, f) => n + f.size, 0) / 1e9

  return (
    <form ref={formRef} className="ingest" onSubmit={submit}>
      <div className="section-title">New Walk</div>
      <div
        className={`dropzone${dragOver ? ' over' : ''}${files.length ? ' has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          const fs = Array.from(e.dataTransfer.files)
          if (fs.length) setFiles(fs)
        }}
        onClick={() => document.getElementById('ingest-file')?.click()}
      >
        {files.length
          ? `${files.map((f) => f.name).join(' + ')} (${gb.toFixed(2)} GB)`
          : 'Drop the .insv here (both files for dual-lens walks), or click to browse'}
        <input
          id="ingest-file"
          type="file"
          accept=".insv,video/*"
          multiple
          hidden
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
      </div>
      <div className="meta-grid">
        {FIELDS.map((f) => (
          <label key={f.name} className="meta-field">
            <span>{f.label}</span>
            <input
              className="field"
              name={f.name}
              placeholder={f.placeholder}
              type={'type' in f ? f.type : 'text'}
              spellCheck={false}
            />
          </label>
        ))}
      </div>
      <div className="insp-note">
        Camera and time come from the file. Leave the date and time blank
        unless the camera's clock was wrong.
      </div>
      {error && <div className="form-error">{error}</div>}
      {busy && (
        <div className="upload-progress">
          <div className="upload-bar">
            <div className="upload-fill" style={{ width: `${progress * 100}%` }} />
          </div>
          <span className="upload-pct">
            {progress < 1 ? `${Math.round(progress * 100)}%` : 'Processing…'}
          </span>
        </div>
      )}
      <button className="go" type="submit" disabled={busy}>
        {busy ? 'Uploading…' : 'Process Walk'}
        {!busy && (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round"
               strokeLinejoin="round">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        )}
      </button>
    </form>
  )
}
