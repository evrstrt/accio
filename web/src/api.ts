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
  auto: number // idx the pipeline chose
  pick: number // idx exported: auto unless a human overrode it
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
  camera: string          // read from the .insv
} | null

// a failed run, recorded next to whatever it produced
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
  lenses: number         // fisheye circles the stitch had
  panos: number
  sharp: number
  faces: number
  pairs: number          // identical-frame pairs in the health check
  reference: number      // median cosine of those pairs
  farPairs: number       // far-apart pairs the threshold was placed against
  calibTau: number       // the calibrated threshold
  segmented: number      // kept frames that carry a mask
  classMix: { name: string; share: number }[]
  anchors: number
  absorbed: number
  solo: number          // groups of one dropped for blur
  dropped: number
  overridden: number     // groups where a human swapped the auto pick
  kept: number
}

// the stage settings a walk was built with
export type PipelineSpec = {
  extract: { fps: number; pano_width: number; sdk_image: string }
  faces: { fov_deg: number; size: number; yaws: number[] }
  gate: { dead: number; band: [number, number] }
  embed: { model_name: string; img_size: number; batch_size: number }
  calib: { samples: number; far_seconds: number; false_merge_pct: number }
  dedup: { tau: number; rule: string; solo_floor: number; solo_span: number }
  segment: {
    enabled: boolean
    model_name: string
    classes: string[]        // open-vocabulary models only
    threshold: number
  }
  embed_model_used: string
  backbones: string[]        // models available for re-embedding
  segmenters: Record<string, string>   // model -> 'semantic' | 'open'
  site_classes: string[]     // default vocabulary for an open model
}

// one finished run's configuration and counts
export type Run = {
  at: string
  from: string           // the stage that run entered at
  backbone: string
  tau: number
  rule: string
  reference: number      // median cosine of identical frames under that backbone
  pairs: number
  faces: number
  anchors: number
  absorbed: number
  solo: number          // groups of one dropped for blur
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
  stage: string                            // the stage running now
  stages: Record<string, StageState>
  stats: Record<string, number>            // counts each stage emitted
  error: string
  first: string                            // '' for an ingest, else the stage a re-run entered at
  params: Partial<PipelineSpec> | null     // the settings this run uses
}

export type Decision = {
  anchorIdx: number
  action: 'pick' | 'drop' | 'restore'
  pickIdx?: number
}

/** FastAPI puts the reason in `detail`, as a string or a validation list. */
function serverSaid(body: string): string {
  try {
    const detail = JSON.parse(body)?.detail
    return (Array.isArray(detail) ? detail[0]?.msg : detail) || ''
  } catch { return '' }
}

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

/** True for a 404, which several routes answer until a stage has run. */
export const isMissing = (e: unknown) => e instanceof ApiError && e.status === 404

async function ok(res: Response, url: string): Promise<Response> {
  if (res.ok) return res
  throw new ApiError(serverSaid(await res.text().catch(() => ''))
    || `${res.status} ${res.statusText} for ${url}`, res.status)
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  return (await ok(await fetch(url, init), url)).json()
}

/** The same image at 256 px, the one thumbnail width the server serves; the 1024 original is about 280 KB. */
export const thumb = (url: string) => `${url}?w=256`

export const fetchWalks = () => req<WalkSummary[]>('/api/walks')
export const fetchWalk = (id: string) =>
  req<WalkDetail>(`/api/walks/${encodeURIComponent(id)}`)
export const postDecision = (walkId: string, d: Decision) =>
  req<{ ok: boolean }>(`/api/walks/${encodeURIComponent(walkId)}/decisions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(d),
  })
// one identical-content pair of the health check
type CalibPair = {
  pano: number
  yaw: number
  tSec: number
  face: string
  neighbour: string
  cosine: number
}

export type Calibration = {
  tau: number | null              // null when there are too few far pairs
  falseMergePct: number
  farSeconds: number
  samples: number
  pairs: CalibPair[]
  // far-apart pairs, which tau is placed against
  far: { n: number; median: number; p95: number; p99: number; max: number }
  // identical-content pairs, a health check on the stitch
  reference: { median: number; p05: number; min: number; n: number }
  healthy: boolean
  referenceError: string
  gapSeconds: number
}

export const fetchCalibration = (walkId: string) =>
  req<Calibration>(`/api/walks/${encodeURIComponent(walkId)}/calibration`)

// the segmenter's output for one kept frame
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

// saved in place; nothing derived depends on it
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

// edits not yet applied, by pipeline section; the server re-runs from the
// earliest stage they invalidate
export type Pending = {
  meta?: MetaEdit
  gate?: { dead?: number; band?: [number, number] }
  faces?: { fov_deg?: number; size?: number; yaws?: number[] }
  embed?: { model_name?: string; batch_size?: number }
  calib?: { samples?: number; far_seconds?: number; false_merge_pct?: number }
  dedup?: { tau?: number; rule?: 'fixed' | 'calibrated'; solo_floor?: number }
  segment?: {
    enabled?: boolean
    model_name?: string
    classes?: string[]
    threshold?: number
  }
}

export type Section = keyof Pending

// must match jobs.pipeline.STAGES on the server
export const STAGES = ['stitch', 'gate', 'faces', 'embed', 'calibrate',
                       'select', 'segment']

// the stage an edit to each section re-runs from; `meta` re-runs nothing
export const STAGE_OF: Record<string, string> = {
  gate: 'gate', faces: 'faces', embed: 'embed', calib: 'calibrate',
  dedup: 'select', segment: 'segment',
}

/** New counts for a re-select; the job id for anything heavier. */
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

export const exportZip = async (walkId: string): Promise<Blob> => {
  const url = `/api/walks/${encodeURIComponent(walkId)}/export`
  return (await ok(await fetch(url), url)).blob()
}

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
      reject(new Error(serverSaid(xhr.responseText)
        || `${xhr.status} ${xhr.statusText} for /api/ingest`))
    }
    xhr.onerror = () => reject(new Error('network error during upload'))
    xhr.send(form)
  })
