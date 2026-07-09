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
  const [file, setFile] = useState<File | null>(null)
  const [sourcePath, setSourcePath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const formRef = useRef<HTMLFormElement>(null)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!file && !sourcePath) {
      setError('drop an .insv or give a path')
      return
    }
    const form = new FormData(formRef.current!)
    if (file) form.set('file', file)
    setBusy(true)
    setError(null)
    postIngest(form)
      .then(onSubmitted)
      .catch((err) => setError(String(err)))
      .finally(() => setBusy(false))
  }

  return (
    <form ref={formRef} className="ingest" onSubmit={submit}>
      <div className="section-title">new walk</div>
      <div
        className={`dropzone${dragOver ? ' over' : ''}${file ? ' has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          const f = e.dataTransfer.files[0]
          if (f) setFile(f)
        }}
        onClick={() => document.getElementById('ingest-file')?.click()}
      >
        {file ? file.name : 'drop an .insv here, or click to browse'}
        <input
          id="ingest-file"
          type="file"
          accept=".insv,video/*"
          hidden
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>
      <div className="or">or a path on this machine</div>
      <input
        className="field path"
        name="source_path"
        placeholder="/path/to/walk.insv"
        value={sourcePath}
        onChange={(e) => setSourcePath(e.target.value)}
        spellCheck={false}
      />
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
      <button className="go" type="submit" disabled={busy}>
        {busy ? 'submitting…' : 'accio'}
      </button>
    </form>
  )
}
