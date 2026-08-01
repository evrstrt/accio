import { useRef, useState } from 'react'
import { postIngest } from './api'
import type { Job } from './api'

// the item-8 capture metadata: what makes the dataset balanceable later
const FIELDS = [
  { name: 'site', label: 'site', placeholder: 'GCMR' },
  { name: 'building', label: 'building', placeholder: 'tower 2' },
  { name: 'stage', label: 'stage', placeholder: 'bare-rcc' },
  { name: 'operator', label: 'operator', placeholder: 'who walked it' },
  { name: 'mount_height_cm', label: 'mount height (cm)', placeholder: '175', type: 'number' },
  { name: 'shot_date', label: 'shot date', placeholder: '', type: 'date' },
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
      setError('drop an .insv first')
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
      <div className="section-title">new walk</div>
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
          : 'drop the .insv here (both files for dual-lens walks), or click to browse'}
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
      {error && <div className="form-error">{error}</div>}
      {busy && (
        <div className="upload-progress">
          <div className="upload-bar">
            <div className="upload-fill" style={{ width: `${progress * 100}%` }} />
          </div>
          <span className="upload-pct">
            {progress < 1 ? `${Math.round(progress * 100)}%` : 'processing…'}
          </span>
        </div>
      )}
      <button className="go" type="submit" disabled={busy}>
        {busy ? 'uploading…' : 'accio'}
      </button>
    </form>
  )
}
