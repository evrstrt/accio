export type Face = {
  idx: number
  panoIdx: number
  tSec: number
  yaw: number
  url: string
  kept: boolean
  anchor: number | null
  cosine: number | null
}

export type Group = {
  anchor: Face
  members: Face[]
  pick: number // idx of the effective pick (anchor unless overridden)
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
  pairs: number          // identical-frame pairs the calibration measured
  reference: number      // what those pairs scored, median
  calibTau: number       // the threshold that resolves to
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
  gate: { window: number; band: [number, number] }
  embed: { model_name: string; img_size: number; batch_size: number }
  calib: { samples: number; quantile: number }
  dedup: { tau: number; rule: string }
  embed_model_used: string
  backbones: string[]        // the models this walk could be re-embedded with
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
  tau: number
  quantile: number
  samples: number
  pairs: CalibPair[]
  reference: { median: number; p05: number; min: number; n: number }
  healthy: boolean
  gapSeconds: number
}

export const fetchCalibration = (walkId: string) =>
  req<Calibration>(`/api/walks/${encodeURIComponent(walkId)}/calibration`)

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
  gate?: { window?: number; band?: [number, number] }
  faces?: { fov_deg?: number; size?: number; yaws?: number[] }
  embed?: { model_name?: string; batch_size?: number }
  calib?: { samples?: number; quantile?: number }
  dedup?: { tau?: number; rule?: 'fixed' | 'calibrated' }
}

export type Section = keyof Pending

// the stages a machine re-runs, in order; Video is the upload and Review is
// people. Must match jobs.pipeline.STAGES on the server.
export const STAGES = ['stitch', 'gate', 'faces', 'embed', 'calibrate', 'select']

// which stage owns each section: editing it re-runs that stage and the rest.
// `meta` is deliberately absent: it re-runs nothing.
export const STAGE_OF: Record<string, string> = {
  gate: 'gate', faces: 'faces', embed: 'embed', calib: 'calibrate',
  dedup: 'select',
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
