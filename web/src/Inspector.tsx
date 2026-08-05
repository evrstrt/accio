import type { ReactNode } from 'react'
import type { Calibration, Pending, Section, WalkDetail } from './api'

// Per-stage panels. Settings are live: editing one stages a re-run from the
// stage it belongs to, and nothing happens until Apply. Where a stage has only
// one implementation its component control is disabled rather than pretending
// to offer a choice; Embed is the first one that has more than one.

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
function Num({ value, step, min, max, disabled, onChange }: {
  value: number
  step?: number
  min?: number
  max?: number
  disabled?: boolean
  onChange: (v: number) => void
}) {
  return (
    <input className="insp-control num" type="number" disabled={disabled}
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

/** "vit_base_patch16_dinov3.lvd1689m" -> "DINOv3 ViT-B/16" */
export function backboneLabel(name: string): string {
  const m = /vit_(base|small|large)_patch(\d+)_(\w+?)[._]/.exec(name)
  if (!m) return name
  const size = { base: 'B', small: 'S', large: 'L' }[m[1]] ?? m[1]
  const family = { dinov3: 'DINOv3', dinov2: 'DINOv2', clip: 'CLIP' }[m[3]] ?? m[3]
  return `${family} ViT-${size}/${m[2]}`
}

/** "nvidia/segformer-b4-finetuned-ade-512-512" -> "SegFormer-B4 · ADE20K" */
export function segmenterLabel(name: string): string {
  const seg = /segformer-(b\d)-finetuned-ade/.exec(name)
  if (seg) return `SegFormer-${seg[1].toUpperCase()} · ADE20K`
  if (name.includes('mask2former')) return 'Mask2Former-L · ADE20K'
  if (name.includes('oneformer')) return 'OneFormer-L · ADE20K'
  if (name.includes('grounding-dino')) return 'Grounding DINO + SAM · your classes'
  return name
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

export default function Inspector({ stage, walk, pending, calib, locked, onEdit,
                                   onOpenReview, onShowCalibration }: {
  stage: string
  walk: WalkDetail
  pending: Pending
  calib: Calibration | null
  locked: boolean          // a run is in flight: look, do not touch
  onEdit: (section: Section, key: string, value: unknown) => void
  onOpenReview: () => void
  onShowCalibration: () => void
}) {
  const { stages: s, pipeline: p, meta } = walk
  // one gate rather than a `disabled` on each control: a staged edit during a
  // run would be applied against settings the run is about to replace
  const edit: typeof onEdit = (...a) => { if (!locked) onEdit(...a) }

  // what a control shows: the staged edit if there is one, else what ran
  const gate = { ...p.gate, ...pending.gate }
  const faces = { ...p.faces, ...pending.faces }
  const emb = { ...p.embed, ...pending.embed }
  const cal = { ...p.calib, ...pending.calib }
  const dedup = { ...p.dedup, ...pending.dedup }
  const m = { ...meta, ...pending.meta }

  if (stage === 'video') {
    /** Metadata is corrected, not re-derived: it ends up in the EXIF of every
        exported frame, so a typo would otherwise mean re-uploading the video. */
    const text = (key: keyof typeof m, label: string, type = 'text') => (
      <Row label={label}>
        <input className="insp-control" type={type} disabled={locked}
               value={(m as Record<string, string>)[key] ?? ''}
               onChange={(e) => edit('meta', key, e.target.value)} />
      </Row>
    )
    return (
      <>
        <Group title="source">
          <Row label="File"><Val>{meta?.videoFile ?? walk.id}</Val></Row>
          {/* read out of the .insv trailer at ingest, not typed */}
          <Row label="Camera"><Val>{meta?.camera || 'unknown'}</Val></Row>
          <Row label="Frame"><Val>{s.source || 'unknown'}</Val></Row>
          <Row label="Lenses">
            {/* both packings are dual-fisheye: two circles in one 2:1 frame,
                or one square frame per lens. One circle is half a sphere. */}
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
          Stamped into the EXIF of every frame in the export, so it travels with
          the images rather than living only here.
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
                 disabled={locked} onChange={(v) => edit('gate', 'window', Math.round(v))} />
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
          <Row label="Reference"><Fixed value="next raw frame"
                                        options={['next raw frame', 'held pose',
                                                  'repeat walk']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Panoramas sampled">
            <Num value={cal.samples} min={2} max={200}
                 disabled={locked} onChange={(v) => edit('calib', 'samples', Math.round(v))} />
          </Row>
          <Row label="Percentile">
            <Num value={cal.quantile} step={1} min={0.5} max={50}
                 disabled={locked} onChange={(v) => edit('calib', 'quantile', v)} />
          </Row>
        </Group>
        {calib && (
          <>
            <Group title="measured">
              <Row label="Identical scores">
                <Val>{calib.reference.median} median</Val>
              </Row>
              <Row label="Lowest pair"><Val>{calib.reference.min}</Val></Row>
              <Row label="Pairs"><Val>{calib.reference.n}, {gap} ms apart</Val></Row>
              <Row label="Calibrated τ"><Val>{calib.tau}</Val></Row>
              {!calib.healthy && (
                <div className="insp-warn">Reference is low; this walk may be
                  mis-stitched or under-exposed.</div>
              )}
            </Group>
            <button className="insp-btn" onClick={onShowCalibration}>
              How τ was measured
            </button>
          </>
        )}
        <div className="insp-note">
          Sampling more panoramas measures again from the video. Moving the
          percentile only re-reads the pairs already measured, so it is instant.
        </div>
        <Out>τ {s.calibTau} from {s.pairs} pairs</Out>
      </>
    )
  }

  if (stage === 'embed') {
    const swapped = emb.model_name !== p.embed_model_used
    return (
      <>
        <Group title="component">
          <Row label="Backbone">
            {/* the first stage with a real choice. Input size follows the
                model, since it has to divide by the patch size. */}
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
            ? <>A different backbone scores "the same thing" differently, so
                Calibrate re-measures and the threshold moves with it. Weights
                download on first use.</>
            : <>Cosines are only comparable within one backbone. Calibrate
                measures this one's scale for identical frames, which is what
                makes two of them worth comparing.</>}
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
              onChange={(e) => edit('dedup', 'rule', e.target.value)}
            >
              <option value="fixed">fixed</option>
              <option value="calibrated">calibrated</option>
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
              disabled={locked || dedup.rule === 'calibrated'}
              onChange={(e) => {
                const v = Number(e.target.value)
                if (e.target.value !== '' && Number.isFinite(v)) {
                  edit('dedup', 'tau', v)
                }
              }}
            />
          </Row>
          <Row label="Scope"><Fixed value="per walk" options={['per walk', 'across walks']} /></Row>
        </Group>
        <div className="insp-note">
          {dedup.rule === 'calibrated'
            ? <>The threshold comes from the Calibrate stage, which measured
                this walk at {s.calibTau}.</>
            : <>A cosine means nothing on its own. Calibrate measured identical
                frames on this walk at {s.reference}.</>}
        </div>
        {walk.runs.length > 1 && (
          <Group title="what each run gave">
            {/* the walk on disk only shows the last configuration that ran, so
                comparing two backbones would otherwise mean keeping notes */}
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
                {/* one per line: the detector reads them as separate phrases,
                    and a long prompt makes it merge neighbouring ones */}
                <textarea
                  className="insp-control"
                  rows={8}
                  disabled={locked}
                  value={seg.classes.join('\n')}
                  onChange={(e) => edit('segment', 'classes',
                    e.target.value.split('\n').map((c) => c.trim()).filter(Boolean))}
                />
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
            {/* the mean share of each class across the frames that carry one */}
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
            ? <>The classes are the prompt, so it finds what you name rather
                than covering the frame. Anything unnamed stays background.
                Measured on bare RCC it reads openings well and needs the
                confidence dropped to find much else.</>
            : <>A fixed label set over every pixel. ADE20K has the shell of a
                building and nothing a site is made of, so for rebar or
                formwork you want the open-vocabulary model.</>}
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
            The only stage a pipeline hash cannot reproduce. Every decision is
            logged, and the log leaves in the export as decisions.json.
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
