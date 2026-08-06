export type Face = {
  idx: number
  panoIdx: number
  tSec: number
  yaw: number
  url: string
  kept: boolean
  anchor: number | null
  cosine: number | null
  sharpness: number      // variance of Laplacian, scored when the face was cut
}

export type Group = {
  anchor: Face
  members: Face[]
  auto: number // idx the pipeline chose: the sharpest member of the group
  pick: number // idx of the effective pick (auto unless a human overrode it)
  dropped: boolean
}

export type WalkMeta = {
  walkId: string
  videoFile: string
  site: string
  building: string
  floor: string
  stage: string
  operator: string
  mountHeightCm: number | null
  shotDate: string
  shotTime: string
  camera: string          // read out of the .insv, not typed
} | null

// a run that broke, recorded next to whatever it managed to produce
export type Failure = {
  stage: string
  message: string
  detail: string
  at: string
}

export type WalkSummary = {
  id: string
  faces: number
  kept: number
  meta: WalkMeta
  ready: boolean         // got as far as a manifest
  error: Failure | null
}

export type Stages = {
  frames: number
  seconds: number
  source: string         // the raw frame size, e.g. "3840x1920"
  lenses: number         // fisheye circles the stitch actually had
  panos: number
  sharp: number
  faces: number
  pairs: number          // identical-frame pairs the health check measured
  reference: number      // what those pairs scored, median
  farPairs: number       // far-apart pairs the threshold was placed against
  calibTau: number       // the threshold that resolves to
  segmented: number      // kept frames that carry a mask
  classMix: { name: string; share: number }[]
  anchors: number
  absorbed: number
  dropped: number
  overridden: number     // groups where a human swapped the auto pick
  kept: number
}

// the stage settings a walk was built with, as the server reports them
export type PipelineSpec = {
  extract: { fps: number; pano_width: number; sdk_image: string }
  faces: { fov_deg: number; size: number; yaws: number[] }
  gate: { window: number; floor: number; dead: number; band: [number, number] }
  embed: { model_name: string; img_size: number; batch_size: number }
  calib: { samples: number; far_seconds: number; false_merge_pct: number }
  dedup: { tau: number; rule: string }
  segment: {
    enabled: boolean
    model_name: string
    classes: string[]        // open-vocabulary models only
    threshold: number
  }
  embed_model_used: string
  backbones: string[]        // the models this walk could be re-embedded with
  segmenters: Record<string, string>   // model -> 'semantic' | 'open'
  site_classes: string[]     // the default vocabulary for an open model
}

// what one configuration produced, appended every time a run finishes
export type Run = {
  at: string
  from: string           // the stage that run entered at
  backbone: string
  tau: number
  rule: string
  reference: number      // what identical frames scored under that backbone
  pairs: number
  faces: number
  anchors: number
  absorbed: number
}

export type WalkDetail = {
  id: string
  faces: number
  groups: Group[]
  meta: WalkMeta
  stages: Stages
  pipeline: PipelineSpec
  runs: Run[]            // newest first
  error: Failure | null
}

export type StageState = 'queued' | 'running' | 'done' | 'error'

export type Job = {
  id: number
  walkId: string
  status: StageState
  stage: string                            // the stage running right now
  stages: Record<string, StageState>
  stats: Record<string, number>            // counts each stage emitted
  error: string
  first: string                            // '' for an ingest, else where a re-run entered
  params: Partial<PipelineSpec> | null     // the settings this run is using
}

export type Decision = {
  anchorIdx: number
  action: 'pick' | 'drop' | 'restore'
  pickIdx?: number
}

/** FastAPI puts the reason in `detail`, as a string or a validation list.
    Without it every failure reads the same and says nothing actionable. */
export function serverSaid(body: string): string {
  try {
    const detail = JSON.parse(body)?.detail
    return (Array.isArray(detail) ? detail[0]?.msg : detail) || ''
  } catch { return '' }
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    throw new Error(serverSaid(await res.text().catch(() => ''))
      || `${res.status} ${res.statusText} for ${url}`)
  }
  return res.json()
}

/** The same face at thumbnail width.
 *
 * The grids draw these around 150 px and the file is 1024 square at roughly
 * 280 KB, so the full one is about twelve times the bytes and fifty times the
 * pixels a tile needs. Only the expanded views ask for the original. */
export const thumb = (url: string, w = 256) => `${url}?w=${w}`

export const fetchWalks = () => req<WalkSummary[]>('/api/walks')
export const fetchWalk = (id: string) =>
  req<WalkDetail>(`/api/walks/${encodeURIComponent(id)}`)
export const postDecision = (walkId: string, d: Decision) =>
  req<{ ok: boolean }>(`/api/walks/${encodeURIComponent(walkId)}/decisions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(d),
  })
// how a walk's identical-content reference was measured
export type CalibPair = {
  pano: number
  yaw: number
  tSec: number
  face: string
  neighbour: string
  cosine: number
}

export type Calibration = {
  tau: number | null              // null when the walk is too short to place one
  falseMergePct: number
  farSeconds: number
  samples: number
  pairs: CalibPair[]
  // what elsewhere scores, which is what tau is placed against
  far: { n: number; median: number; p95: number; p99: number; max: number }
  // the identical-content ceiling, kept as a health check on the stitch
  reference: { median: number; p05: number; min: number; n: number }
  healthy: boolean
  referenceError: string
  gapSeconds: number
}

export const fetchCalibration = (walkId: string) =>
  req<Calibration>(`/api/walks/${encodeURIComponent(walkId)}/calibration`)

// the segmenter's output, per kept frame: the mask and what it is made of
export type SegFrame = {
  face: string
  tSec: number
  yaw: number
  url: string
  mask: string
  classes: Record<string, number>
}

export type Segmentation = {
  model: string
  labels: Record<string, string>   // mask index -> class name
  frames: SegFrame[]
  classMix: { name: string; share: number }[]
}

export const fetchSegmentation = (walkId: string) =>
  req<Segmentation>(`/api/walks/${encodeURIComponent(walkId)}/segmentation`)

export const fetchJobs = () => req<Job[]>('/api/jobs')

// Settings edited but not yet applied, by the section of the pipeline params
// they belong to. The server folds them in and re-runs from the earliest stage
// they invalidate, so the shape here is the shape it validates against.
// what the walk was shot on and where. Nothing derived depends on it, so it
// saves in place rather than re-running a stage.
export type MetaEdit = {
  site?: string
  building?: string
  floor?: string
  stage?: string
  operator?: string
  mountHeightCm?: number
  shotDate?: string
  shotTime?: string
  camera?: string
}

export type Pending = {
  meta?: MetaEdit
  gate?: { window?: number; floor?: number; dead?: number; band?: [number, number] }
  faces?: { fov_deg?: number; size?: number; yaws?: number[] }
  embed?: { model_name?: string; batch_size?: number }
  calib?: { samples?: number; far_seconds?: number; false_merge_pct?: number }
  dedup?: { tau?: number; rule?: 'fixed' | 'calibrated' }
  segment?: {
    enabled?: boolean
    model_name?: string
    classes?: string[]
    threshold?: number
  }
}

export type Section = keyof Pending

// the stages a machine re-runs, in order; Video is the upload and Review is
// people. Must match jobs.pipeline.STAGES on the server.
export const STAGES = ['stitch', 'gate', 'faces', 'embed', 'calibrate',
                       'select', 'segment']

// which stage owns each section: editing it re-runs that stage and the rest.
// `meta` is deliberately absent: it re-runs nothing.
export const STAGE_OF: Record<string, string> = {
  gate: 'gate', faces: 'faces', embed: 'embed', calib: 'calibrate',
  dedup: 'select', segment: 'segment',
}

/** A re-select answers with the new counts; anything heavier answers with the
    job now running, and the canvas follows it from there. */
export type RerunResult = {
  changed: boolean
  anchors?: number
  absorbed?: number
  id?: number
}

export const postRerun = (walkId: string, p: Pending) =>
  req<RerunResult>(`/api/walks/${encodeURIComponent(walkId)}/rerun`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(p),
  })

export const retryWalk = (walkId: string) =>
  req<Job>(`/api/walks/${encodeURIComponent(walkId)}/retry`, { method: 'POST' })

export const deleteWalk = (walkId: string) =>
  req<{ deleted: string; videos: string[] }>(
    `/api/walks/${encodeURIComponent(walkId)}`, { method: 'DELETE' })

export const patchMeta = (walkId: string, m: MetaEdit) =>
  req<WalkMeta>(`/api/walks/${encodeURIComponent(walkId)}/meta`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(m),
  })

// XHR instead of fetch: multi-GB .insv uploads need progress events
export const postIngest = (form: FormData, onProgress?: (frac: number) => void) =>
  new Promise<Job>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/ingest')
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText))
        return
      }
      // the same `detail` the fetch path reads. A refused upload is refused
      // for a reason the operator can act on ("upload the _10_ file too"),
      // and a bare status code throws that away.
      reject(new Error(serverSaid(xhr.responseText)
        || `${xhr.status} ${xhr.statusText} for /api/ingest`))
    }
    xhr.onerror = () => reject(new Error('network error during upload'))
    xhr.send(form)
  })
