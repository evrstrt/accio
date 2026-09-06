import { useState } from 'react'
import type { ReactNode } from 'react'
import type { Calibration, MetaEdit, Pending, Section, WalkDetail } from './api'
import { backboneLabel, segmenterLabel } from './labels'

/** One staged edit: a key of the section's pending shape, with that key's value type. */
export type Edit = <S extends Section, K extends keyof NonNullable<Pending[S]>>(
  section: S, key: K, value: NonNullable<Pending[S]>[K]) => void

// per-stage panels; an edit stages a re-run from its stage, applied on Apply

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="insp-row">
      <span className="insp-label">{label}</span>
      {children}
    </div>
  )
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="insp-group">
      <div className="insp-group-title">{title}</div>
      {children}
    </div>
  )
}

/** A component with a single implementation, shown disabled. */
function Fixed({ value, options }: { value: string; options: string[] }) {
  const all = options.includes(value) ? options : [value, ...options]
  return (
    <select className="insp-control" value={value} disabled
            title="only one implementation so far" onChange={() => {}}>
      {all.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

function Toggle({ on }: { on: boolean }) {
  return <input className="insp-toggle" type="checkbox" checked={on} disabled
                onChange={() => {}} />
}

/** Holds the raw text while editing and commits the parsed number on blur, so
    intermediate values (1, 11, 110) never get staged. */
function Num({ value, step, min, max, disabled, onChange }: {
  value: number
  step?: number
  min?: number
  max?: number
  disabled?: boolean
  onChange: (v: number) => void
}) {
  const [text, setText] = useState<string | null>(null)

  const commit = () => {
    const v = Number(text)
    setText(null)
    if (text === null || text.trim() === '' || !Number.isFinite(v)) return
    // min/max are advisory on a typed value, so clamp here
    const bounded = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v))
    if (bounded !== value) onChange(bounded)
  }

  return (
    <input className="insp-control num" type="number" disabled={disabled}
           step={step ?? 1} min={min} max={max}
           value={text ?? String(value)}
           onChange={(e) => setText(e.target.value)}
           onBlur={commit}
           onKeyDown={(e) => {
             if (e.key === 'Enter') e.currentTarget.blur()
             if (e.key === 'Escape') setText(null)
           }} />
  )
}

function Val({ children }: { children: ReactNode }) {
  return <span className="insp-value">{children}</span>
}

/** One class per line. Raw text until blur: a controlled value that is trimmed
    on change can never hold a trailing space or newline. */
function Classes({ value, disabled, onChange }: {
  value: string[]
  disabled?: boolean
  onChange: (v: string[]) => void
}) {
  const [text, setText] = useState<string | null>(null)
  return (
    <textarea
      className="insp-control"
      rows={8}
      disabled={disabled}
      value={text ?? value.join('\n')}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        const parsed = (text ?? '').split('\n').map((c) => c.trim()).filter(Boolean)
        setText(null)
        if (text !== null && parsed.join('\n') !== value.join('\n')) onChange(parsed)
      }}
      onKeyDown={(e) => { if (e.key === 'Escape') setText(null) }}
    />
  )
}

function Out({ children }: { children: ReactNode }) {
  return <div className="insp-out">{children}</div>
}

/** n headings, offset so the stitch seams at yaw +-90 fall on a face edge. */
function yawRing(n: number): number[] {
  const spacing = 360 / n
  const first = (90 - spacing / 2 + 360) % spacing
  return Array.from({ length: n }, (_, i) => Math.round(first + i * spacing))
}

const RINGS = [3, 4, 6]

export default function Inspector({ stage, walk, pending, calib, locked, onEdit,
                                   onOpenReview, onShowCalibration }: {
  stage: string
  walk: WalkDetail
  pending: Pending
  calib: Calibration | null
  locked: boolean          // a run is in flight
  onEdit: Edit
  onOpenReview: () => void
  onShowCalibration: () => void
}) {
  const { stages: s, pipeline: p, meta } = walk
  // an edit staged during a run would apply against settings the run is about to replace
  const edit: Edit = (...a) => { if (!locked) onEdit(...a) }

  // staged edit if any, else what ran
  const gate = { ...p.gate, ...pending.gate }
  const faces = { ...p.faces, ...pending.faces }
  const emb = { ...p.embed, ...pending.embed }
  const cal = { ...p.calib, ...pending.calib }
  const dedup = { ...p.dedup, ...pending.dedup }
  const m = { ...meta, ...pending.meta }

  if (stage === 'video') {
    const text = (key: Exclude<keyof MetaEdit, 'mountHeightCm'>, label: string,
                  type = 'text') => (
      <Row label={label}>
        <input className="insp-control" type={type} disabled={locked}
               value={m[key] ?? ''}
               onChange={(e) => edit('meta', key, e.target.value)} />
      </Row>
    )
    return (
      <>
        <Group title="source">
          <Row label="File"><Val>{meta?.videoFile ?? walk.id}</Val></Row>
          {/* read from the .insv trailer at ingest */}
          <Row label="Camera"><Val>{meta?.camera || 'unknown'}</Val></Row>
          <Row label="Frame"><Val>{s.source || 'unknown'}</Val></Row>
          <Row label="Lenses">
            {/* dual-fisheye: two circles in one 2:1 frame, or one square frame per lens */}
            <Val>{s.lenses === 2 ? 'fisheye ×2' : s.lenses === 1
              ? 'fisheye ×1, incomplete' : 'unknown'}</Val>
          </Row>
          <Row label="Footage"><Val>{s.frames.toLocaleString()} frames</Val></Row>
        </Group>
        <Group title="capture metadata">
          {text('site', 'Site')}
          {text('building', 'Building')}
          {text('floor', 'Floor')}
          {text('stage', 'Stage')}
          {text('operator', 'Operator')}
          <Row label="Mount height">
            <Num value={m.mountHeightCm ?? 0} min={0} max={1000}
                 disabled={locked} onChange={(v) => edit('meta', 'mountHeightCm', Math.round(v))} />
          </Row>
          {text('shotDate', 'Shot date', 'date')}
          {text('shotTime', 'Shot time', 'time')}
        </Group>
        {s.lenses === 1 && (
          <div className="insp-warn">
            This walk was stitched from one lens, so the back of every panorama
            is the front smeared across it. Upload the _10_ file with the _00_
            and re-ingest.
          </div>
        )}
        <div className="insp-note">
          Written into the EXIF of every exported frame.
        </div>
      </>
    )
  }

  if (stage === 'stitch') {
    return (
      <>
        <Group title="component">
          <Row label="Stitcher"><Fixed value="Insta360 MediaSDK"
                                       options={['Insta360 MediaSDK', 'ffmpeg v360']} /></Row>
          <Row label="Blend"><Fixed value="optflow" options={['optflow', 'template', 'dynamic']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Flowstate levelling"><Toggle on /></Row>
          <Row label="Output width"><Val>{p.extract.pano_width}</Val></Row>
          <Row label="Sample rate"><Val>{p.extract.fps} fps</Val></Row>
          <Row label="Container"><Val>{p.extract.sdk_image}</Val></Row>
        </Group>
        <div className="insp-note">
          Changing these means re-stitching. Re-ingest the video instead.
        </div>
        <Out>{s.panos} panos at {p.extract.fps}fps</Out>
      </>
    )
  }

  if (stage === 'gate') {
    return (
      <>
        <Group title="component">
          <Row label="Score"><Fixed value="variance of Laplacian"
                                    options={['variance of Laplacian', 'Tenengrad']} /></Row>
          <Row label="Rule"><Fixed value="drop the unusable"
                                   options={['drop the unusable', 'sharpest per window', 'none']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Dead">
            <Num value={gate.dead} step={0.05} min={0} max={0.95}
                 disabled={locked} onChange={(v) => edit('gate', 'dead', v)} />
          </Row>
          <Row label="Band top">
            <Num value={gate.band[0]} step={0.05} min={0} max={1}
                 disabled={locked} onChange={(v) => edit('gate', 'band', [v, gate.band[1]])} />
          </Row>
          <Row label="Band bottom">
            <Num value={gate.band[1]} step={0.05} min={0} max={1}
                 disabled={locked} onChange={(v) => edit('gate', 'band', [gate.band[0], v])} />
          </Row>
        </Group>
        <div className="insp-note">
          Drops panoramas under {gate.dead.toFixed(2)} of the walk's median
          sharpness. This only catches a stretch smeared right through; the score
          rates a plain wall low whether or not it is sharp, so raising it costs
          coverage, not blur.
        </div>
        <Out>{s.sharp} legible of {s.panos} panos</Out>
      </>
    )
  }

  if (stage === 'faces') {
    return (
      <>
        <Group title="component">
          <Row label="Projection"><Fixed value="gnomonic"
                                         options={['gnomonic', 'cubemap', 'none (equirect)']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Field of view">
            <Num value={faces.fov_deg} min={20} max={170}
                 disabled={locked} onChange={(v) => edit('faces', 'fov_deg', v)} />
          </Row>
          <Row label="Face size">
            <Num value={faces.size} step={128} min={128} max={4096}
                 disabled={locked} onChange={(v) => edit('faces', 'size', Math.round(v))} />
          </Row>
          <Row label="Faces per pano">
            <select
              className="insp-control" disabled={locked}
              value={String(faces.yaws.length)}
              onChange={(e) => edit('faces', 'yaws', yawRing(Number(e.target.value)))}
            >
              {(RINGS.includes(faces.yaws.length)
                ? RINGS : [...RINGS, faces.yaws.length].sort((a, b) => a - b))
                .map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </Row>
          <Row label="Headings"><Val>{faces.yaws.join(', ')}</Val></Row>
        </Group>
        <div className="insp-note">
          {(360 / faces.yaws.length).toFixed(0)}° apart at {faces.fov_deg}°,
          {' '}{(faces.fov_deg - 360 / faces.yaws.length).toFixed(0)}° of overlap
          per seam. Headings are offset so the stitch seams at yaw ±90 fall on a
          face edge.
        </div>
        <Out>{s.sharp} × {faces.yaws.length} = {s.sharp * faces.yaws.length} faces</Out>
      </>
    )
  }

  if (stage === 'calibrate') {
    const gap = calib ? Math.round(calib.gapSeconds * 1000) : 0
    return (
      <>
        <Group title="component">
          <Row label="Placed against"><Fixed value="far-apart pairs"
                                             options={['far-apart pairs']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="False merges allowed">
            <Num value={cal.false_merge_pct} step={0.5} min={0.1} max={50}
                 disabled={locked}
                 onChange={(v) => edit('calib', 'false_merge_pct', v)} />
          </Row>
          <Row label="Elsewhere is (s) apart">
            <Num value={cal.far_seconds} step={5} min={2} max={600}
                 disabled={locked}
                 onChange={(v) => edit('calib', 'far_seconds', v)} />
          </Row>
          <Row label="Panoramas sampled">
            <Num value={cal.samples} min={2} max={200}
                 disabled={locked} onChange={(v) => edit('calib', 'samples', Math.round(v))} />
          </Row>
        </Group>
        {calib && (
          <>
            <Group title="measured">
              <Row label="Elsewhere scores">
                <Val>{calib.far.median} median, {calib.far.p99} p99</Val>
              </Row>
              <Row label="Far pairs"><Val>{calib.far.n.toLocaleString()}</Val></Row>
              <Row label="Calibrated τ"><Val>{calib.tau ?? 'none'}</Val></Row>
              <Row label="Identical scores">
                <Val>{calib.reference.median} median, {gap} ms apart</Val>
              </Row>
              {calib.tau === null && (
                <div className="insp-warn">Too few far pairs to place τ on.
                  Use the fixed rule, or a longer walk.</div>
              )}
              {calib.referenceError && (
                <div className="insp-warn">The health check did not run. τ is
                  unaffected, but nothing vouches for the stitch here.</div>
              )}
              {!calib.healthy && !calib.referenceError && (
                <div className="insp-warn">Identical frames score low; this walk
                  may be mis-stitched or under-exposed.</div>
              )}
            </Group>
            <button className="insp-btn" onClick={onShowCalibration}>
              How τ was placed
            </button>
          </>
        )}
        <div className="insp-note">
          A bigger false-merge budget cuts harder. Every face stays on disk, so
          an over-cut walk can be re-cut. Changing the budget is instant; the
          window and the sample count re-measure.
        </div>
        <Out>τ {s.calibTau} from {s.farPairs} far pairs</Out>
      </>
    )
  }

  if (stage === 'embed') {
    const swapped = emb.model_name !== p.embed_model_used
    return (
      <>
        <Group title="component">
          <Row label="Backbone">
            {/* input size follows the model: it has to divide by the patch size */}
            <select
              className="insp-control" disabled={locked}
              value={emb.model_name}
              onChange={(e) => edit('embed', 'model_name', e.target.value)}
            >
              {p.backbones.map((b) => (
                <option key={b} value={b}>{backboneLabel(b)}</option>
              ))}
            </select>
          </Row>
          <Row label="Token"><Fixed value="CLS" options={['CLS', 'patch mean']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Input size"><Val>{emb.img_size}</Val></Row>
          <Row label="Batch size">
            <Num value={emb.batch_size} min={1} max={64}
                 disabled={locked} onChange={(v) => edit('embed', 'batch_size', Math.round(v))} />
          </Row>
          <Row label="Weights"><Val>{p.embed_model_used || emb.model_name}</Val></Row>
        </Group>
        <div className="insp-note">
          {swapped
            ? <>Cosines differ between backbones, so Calibrate re-measures
                the threshold. Weights download on first use.</>
            : <>Cosines are only comparable within one backbone.</>}
        </div>
        <Out>{s.faces} vectors, L2 normalised</Out>
      </>
    )
  }

  if (stage === 'select') {
    return (
      <>
        <Group title="component">
          <Row label="Rule"><Fixed value="greedy cosine"
                                   options={['greedy cosine', 'agglomerative', 'coverage']} /></Row>
          <Row label="Threshold">
            <select
              className="insp-control" disabled={locked}
              value={dedup.rule}
              onChange={(e) => {
                const v = e.target.value
                if (v === 'fixed' || v === 'calibrated') edit('dedup', 'rule', v)
              }}
            >
              <option value="fixed">fixed</option>
              <option value="calibrated">calibrated</option>
            </select>
          </Row>
        </Group>
        <Group title="settings">
          <Row label="τ">
            {/* under the calibrated rule the walk derives its own tau */}
            <Num value={dedup.tau} step={0.005} min={0.5} max={0.999}
                 disabled={locked || dedup.rule === 'calibrated'}
                 onChange={(v) => edit('dedup', 'tau', v)} />
          </Row>
          <Row label="Scope"><Fixed value="per walk" options={['per walk', 'across walks']} /></Row>
          <Row label="Solo floor">
            <Num value={dedup.solo_floor} step={0.05} min={0} max={0.95}
                 disabled={locked} onChange={(v) => edit('dedup', 'solo_floor', v)} />
          </Row>
        </Group>
        <div className="insp-note">
          {dedup.rule === 'calibrated'
            ? <>From Calibrate, which measured this walk at {s.calibTau}.</>
            : <>Calibrate measured identical frames on this walk at {s.reference}.</>}
        </div>
        <div className="insp-note">
          Groups anchor on their sharpest frame. A group of one has no
          alternative, so the solo floor drops it when it scores under{' '}
          {(dedup.solo_floor ?? 0).toFixed(2)} of its heading's usual sharpness.
          0 keeps every one; each drop is a view nothing else in the walk shows.
          {s.solo ? <> This run dropped {s.solo}.</> : null}
        </div>
        {walk.runs.length > 1 && (
          <Group title="what each run gave">
            <div className="runs">
              {walk.runs.map((r, i) => (
                <div className={`run${i === 0 ? ' current' : ''}`} key={r.at + i}>
                  <span className="run-model">{backboneLabel(r.backbone)}</span>
                  <span className="run-tau">τ {r.tau}</span>
                  <span className="run-ref">ref {r.reference || '—'}</span>
                  <span className="run-out">{r.absorbed} absorbed</span>
                </div>
              ))}
            </div>
          </Group>
        )}
        <Out>{s.absorbed} absorbed, {s.anchors} anchors</Out>
      </>
    )
  }

  if (stage === 'segment') {
    const seg = { ...p.segment, ...pending.segment }
    const open = p.segmenters[seg.model_name] === 'open'
    return (
      <>
        <Group title="component">
          <Row label="Model">
            <select className="insp-control" disabled={locked}
                    value={seg.model_name}
                    onChange={(e) => edit('segment', 'model_name', e.target.value)}>
              {Object.keys(p.segmenters).map((m) => (
                <option key={m} value={m}>{segmenterLabel(m)}</option>
              ))}
            </select>
          </Row>
          <Row label="Labels">
            <Fixed value={open ? 'whatever you name' : 'ADE20K, 150 classes'}
                   options={[open ? 'whatever you name' : 'ADE20K, 150 classes']} />
          </Row>
        </Group>
        <Group title="settings">
          <Row label="Run it">
            <input className="insp-toggle" type="checkbox" disabled={locked}
                   checked={seg.enabled}
                   onChange={(e) => edit('segment', 'enabled', e.target.checked)} />
          </Row>
          {open && (
            <>
              <Row label="Confidence">
                <Num value={seg.threshold} step={0.05} min={0.05} max={0.95}
                     disabled={locked}
                     onChange={(v) => edit('segment', 'threshold', v)} />
              </Row>
              <div className="insp-classes">
                {/* one per line: a single long prompt makes the detector merge neighbouring phrases */}
                <Classes value={seg.classes} disabled={locked}
                         onChange={(cs) => edit('segment', 'classes', cs)} />
              </div>
            </>
          )}
          {!open && (
            <Row label="Scope"><Fixed value="kept frames"
                                      options={['kept frames', 'every face']} /></Row>
          )}
        </Group>
        {s.classMix.length > 0 && (
          <Group title="what the walk is made of">
            {/* mean share per class over the frames that carry it */}
            <div className="runs">
              {s.classMix.map((c) => (
                <div className="mix" key={c.name}>
                  <span className="mix-name">{c.name}</span>
                  <span className="mix-bar">
                    <span style={{ width: `${Math.round(c.share * 100)}%` }} />
                  </span>
                  <span className="mix-pct">{Math.round(c.share * 100)}%</span>
                </div>
              ))}
            </div>
          </Group>
        )}
        <div className="insp-note">
          {open
            ? <>Finds what you name; anything unnamed stays background. On
                bare RCC it reads openings well and needs a lower confidence
                to find much else.</>
            : <>A fixed label set over every pixel. ADE20K has walls, floors
                and openings, not rebar or formwork; use the open-vocabulary
                model for those.</>}
          {' '}Annotation only: it never changes which frames survive.
        </div>
        <Out>{seg.enabled ? `${s.segmented} of ${s.kept} masked` : 'off'}</Out>
      </>
    )
  }

  if (stage === 'review') {
    return (
      <>
        <Group title="manual stage">
          <div className="insp-note">
            Every decision is logged and exported as decisions.json.
          </div>
        </Group>
        <Group title="decisions">
          <Row label="Groups dropped"><Val>{s.dropped}</Val></Row>
          <Row label="Picks overridden"><Val>{s.overridden}</Val></Row>
        </Group>
        <button className="insp-btn" onClick={onOpenReview}>Open Review</button>
        <Out>{s.kept} kept of {s.anchors}</Out>
      </>
    )
  }

  return null
}
