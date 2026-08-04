import type { ReactNode } from 'react'
import type { Calibration, Pending, Section, WalkDetail } from './api'

// Per-stage panels. Settings are live: editing one stages a re-run from the
// stage it belongs to, and nothing happens until Apply. Choosing a different
// component (a second stitcher, another backbone) is the part that is not
// built yet, so those controls are disabled rather than pretending.

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

/** A component choice we have exactly one implementation of, shown so the
    pipeline says what it is made of, disabled so it cannot claim otherwise. */
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

/** A live number: edits stage a re-run, blanks and junk stage nothing. */
function Num({ value, step, min, max, onChange }: {
  value: number
  step?: number
  min?: number
  max?: number
  onChange: (v: number) => void
}) {
  return (
    <input className="insp-control num" type="number"
           step={step ?? 1} min={min} max={max} value={value}
           onChange={(e) => {
             const v = Number(e.target.value)
             if (e.target.value !== '' && Number.isFinite(v)) onChange(v)
           }} />
  )
}

function Val({ children }: { children: ReactNode }) {
  return <span className="insp-value">{children}</span>
}

function Out({ children }: { children: ReactNode }) {
  return <div className="insp-out">{children}</div>
}

/** Faces evenly around the sphere, offset so the stitch seams at yaw +-90 fall
    on a face edge rather than through the middle of one. */
export function yawRing(n: number): number[] {
  const spacing = 360 / n
  const first = (90 - spacing / 2 + 360) % spacing
  return Array.from({ length: n }, (_, i) => Math.round(first + i * spacing))
}

const RINGS = [3, 4, 6]

export default function Inspector({ stage, walk, pending, calib, onEdit,
                                   onOpenReview, onShowCalibration }: {
  stage: string
  walk: WalkDetail
  pending: Pending
  calib: Calibration | null
  onEdit: (section: Section, key: string, value: unknown) => void
  onOpenReview: () => void
  onShowCalibration: () => void
}) {
  const { stages: s, pipeline: p, meta } = walk

  // what a control shows: the staged edit if there is one, else what ran
  const gate = { ...p.gate, ...pending.gate }
  const faces = { ...p.faces, ...pending.faces }
  const dedup = { ...p.dedup, ...pending.dedup }

  if (stage === 'video') {
    return (
      <>
        <Group title="source">
          <Row label="File"><Val>{meta?.videoFile ?? walk.id}</Val></Row>
          <Row label="Format"><Val>dual-fisheye .insv</Val></Row>
        </Group>
        <Group title="capture metadata">
          <Row label="Site"><input className="insp-control" defaultValue={meta?.site ?? ''} /></Row>
          <Row label="Building"><input className="insp-control" defaultValue={meta?.building ?? ''} /></Row>
          <Row label="Stage"><input className="insp-control" defaultValue={meta?.stage ?? ''} /></Row>
          <Row label="Operator"><input className="insp-control" defaultValue={meta?.operator ?? ''} /></Row>
          <Row label="Mount height"><Num value={meta?.mountHeightCm ?? 0} /></Row>
          <Row label="Shot date"><input className="insp-control" type="date"
                                        defaultValue={meta?.shotDate ?? ''} /></Row>
        </Group>
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
          Changing these re-stitches from the original, which is the whole run
          again. Re-ingest the video instead.
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
          <Row label="Rule"><Fixed value="sharpest per window"
                                   options={['sharpest per window', 'absolute threshold', 'none']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Window">
            <Num value={gate.window} min={1} max={120}
                 onChange={(v) => onEdit('gate', 'window', Math.round(v))} />
          </Row>
          <Row label="Band top">
            <Num value={gate.band[0]} step={0.05} min={0} max={1}
                 onChange={(v) => onEdit('gate', 'band', [v, gate.band[1]])} />
          </Row>
          <Row label="Band bottom">
            <Num value={gate.band[1]} step={0.05} min={0} max={1}
                 onChange={(v) => onEdit('gate', 'band', [gate.band[0], v])} />
          </Row>
        </Group>
        <div className="insp-note">
          One panorama per window survives, so the window sets the spacing:
          {' '}{(gate.window / p.extract.fps).toFixed(1)}s of walk each.
        </div>
        <Out>{s.sharp} sharp of {s.panos} panos</Out>
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
                 onChange={(v) => onEdit('faces', 'fov_deg', v)} />
          </Row>
          <Row label="Face size">
            <Num value={faces.size} step={128} min={128} max={4096}
                 onChange={(v) => onEdit('faces', 'size', Math.round(v))} />
          </Row>
          <Row label="Faces per pano">
            <select
              className="insp-control"
              value={String(faces.yaws.length)}
              onChange={(e) => onEdit('faces', 'yaws', yawRing(Number(e.target.value)))}
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

  if (stage === 'embed') {
    return (
      <>
        <Group title="component">
          <Row label="Backbone"><Fixed value="DINOv3 ViT-B/16"
                                       options={['DINOv3 ViT-B/16', 'DINOv2 ViT-B/14',
                                                 'RADIO', 'CLIP ViT-B/16']} /></Row>
          <Row label="Token"><Fixed value="CLS" options={['CLS', 'patch mean']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Input size"><Val>{p.embed.img_size}</Val></Row>
          <Row label="Batch size"><Val>{p.embed.batch_size}</Val></Row>
          <Row label="Weights"><Val>{p.embed_model_used || p.embed.model_name}</Val></Row>
        </Group>
        <Out>{s.faces} vectors, 768d, L2 normalised</Out>
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
              className="insp-control"
              value={dedup.rule}
              onChange={(e) => onEdit('dedup', 'rule', e.target.value)}
            >
              <option value="fixed">fixed</option>
              {/* nothing to take a threshold from on a walk measured before
                  calibration existed, so the rule would be a no-op */}
              <option value="calibrated" disabled={!calib}>
                calibrated{calib ? '' : ' (not measured)'}
              </option>
            </select>
          </Row>
        </Group>
        <Group title="settings">
          <Row label="τ">
            {/* under the calibrated rule the walk derives its own value, so
                the number is shown but not editable */}
            <input
              className="insp-control num"
              type="number" step={0.005} min={0.5} max={0.999}
              value={dedup.tau}
              disabled={dedup.rule === 'calibrated'}
              onChange={(e) => {
                const v = Number(e.target.value)
                if (e.target.value !== '' && Number.isFinite(v)) {
                  onEdit('dedup', 'tau', v)
                }
              }}
            />
          </Row>
          <Row label="Scope"><Fixed value="per walk" options={['per walk', 'across walks']} /></Row>
        </Group>
        {calib && (
          <Group title="calibration">
            <Row label="Identical scores">
              <Val>{calib.reference.median} median · {calib.reference.n} pairs</Val>
            </Row>
            <Row label="Calibrated τ"><Val>{calib.tau}</Val></Row>
            {!calib.healthy && (
              <div className="insp-warn">Reference is low; this walk may be
                mis-stitched or under-exposed.</div>
            )}
            <button className="insp-btn" onClick={onShowCalibration}>
              How τ was measured
            </button>
          </Group>
        )}
        <Out>{s.absorbed} absorbed, {s.anchors} anchors</Out>
      </>
    )
  }

  if (stage === 'review') {
    return (
      <>
        <Group title="manual stage">
          <div className="insp-note">
            The only stage a pipeline hash cannot reproduce. Every override is
            logged, and the log travels with the export.
          </div>
        </Group>
        <Group title="decisions">
          <Row label="Groups dropped"><Val>{s.dropped}</Val></Row>
          <Row label="Picks overridden"><Val>0</Val></Row>
        </Group>
        <button className="insp-btn" onClick={onOpenReview}>Open Review</button>
        <Out>{s.kept} kept of {s.anchors}</Out>
      </>
    )
  }

  return null
}
