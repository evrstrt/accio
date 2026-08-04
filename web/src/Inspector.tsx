import type { ReactNode } from 'react'
import type { WalkDetail } from './api'

// Per-stage panels. The controls are mock: uncontrolled inputs with no
// handlers, so they move but change nothing. What each stage exposes here is
// the real question (which knobs, which swappable component), and that is
// easier to argue with on screen than in a config file.

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

function Select({ value, options }: { value: string; options: string[] }) {
  const all = options.includes(value) ? options : [value, ...options]
  return (
    <select className="insp-control" defaultValue={value}>
      {all.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

function Num({ value, step }: { value: number; step?: number }) {
  return <input className="insp-control num" type="number" step={step ?? 1}
                defaultValue={value} />
}

function Toggle({ on }: { on: boolean }) {
  return <input className="insp-toggle" type="checkbox" defaultChecked={on} />
}

function Val({ children }: { children: ReactNode }) {
  return <span className="insp-value">{children}</span>
}

function Out({ children }: { children: ReactNode }) {
  return <div className="insp-out">{children}</div>
}

export default function Inspector({ stage, walk, onOpenReview }: {
  stage: string
  walk: WalkDetail
  onOpenReview: () => void
}) {
  const { stages: s, pipeline: p, meta } = walk

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
          <Row label="Stitcher"><Select value="Insta360 MediaSDK"
                                        options={['Insta360 MediaSDK', 'ffmpeg v360']} /></Row>
          <Row label="Blend"><Select value="optflow" options={['optflow', 'template', 'dynamic']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Flowstate levelling"><Toggle on /></Row>
          <Row label="Output width"><Select value={String(p.extract.pano_width)}
                                            options={['3840', '5760']} /></Row>
          <Row label="Sample rate"><Num value={p.extract.fps} step={0.5} /></Row>
          <Row label="Container"><Val>{p.extract.sdk_image}</Val></Row>
        </Group>
        <Out>{s.panos} panos at {p.extract.fps}fps</Out>
      </>
    )
  }

  if (stage === 'gate') {
    return (
      <>
        <Group title="component">
          <Row label="Score"><Select value="variance of Laplacian"
                                     options={['variance of Laplacian', 'Tenengrad']} /></Row>
          <Row label="Rule"><Select value="sharpest per window"
                                    options={['sharpest per window', 'absolute threshold', 'none']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Window"><Num value={p.gate.window} /></Row>
          <Row label="Band top"><Num value={p.gate.band[0]} step={0.05} /></Row>
          <Row label="Band bottom"><Num value={p.gate.band[1]} step={0.05} /></Row>
        </Group>
        <Out>{s.sharp} sharp of {s.panos} panos</Out>
      </>
    )
  }

  if (stage === 'faces') {
    return (
      <>
        <Group title="component">
          <Row label="Projection"><Select value="gnomonic"
                                          options={['gnomonic', 'cubemap', 'none (equirect)']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Field of view"><Num value={p.faces.fov_deg} /></Row>
          <Row label="Face size"><Num value={p.faces.size} step={128} /></Row>
          <Row label="Yaws"><Val>{p.faces.yaws.join(', ')}</Val></Row>
        </Group>
        <Out>{s.sharp} × {p.faces.yaws.length} = {s.faces} faces</Out>
      </>
    )
  }

  if (stage === 'embed') {
    return (
      <>
        <Group title="component">
          <Row label="Backbone"><Select value="DINOv3 ViT-B/16"
                                        options={['DINOv3 ViT-B/16', 'DINOv2 ViT-B/14',
                                                  'RADIO', 'CLIP ViT-B/16']} /></Row>
          <Row label="Token"><Select value="CLS" options={['CLS', 'patch mean']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="Input size"><Num value={p.embed.img_size} step={32} /></Row>
          <Row label="Batch size"><Num value={p.embed.batch_size} /></Row>
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
          <Row label="Rule"><Select value="greedy cosine"
                                    options={['greedy cosine', 'agglomerative', 'coverage']} /></Row>
          <Row label="Threshold"><Select value="fixed" options={['fixed', 'per video (auto)']} /></Row>
        </Group>
        <Group title="settings">
          <Row label="τ"><Num value={p.dedup.tau} step={0.005} /></Row>
          <Row label="Scope"><Select value="per walk" options={['per walk', 'across walks']} /></Row>
        </Group>
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
