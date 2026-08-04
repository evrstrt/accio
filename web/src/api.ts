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
  stage: string
  operator: string
  mountHeightCm: number | null
  shotDate: string
} | null

export type WalkSummary = {
  id: string
  faces: number
  kept: number
  meta: WalkMeta
}

export type Stages = {
  frames: number
  seconds: number
  panos: number
  sharp: number
  faces: number
  anchors: number
  absorbed: number
  dropped: number
  kept: number
}

// the stage settings a walk was built with, as the server reports them
export type PipelineSpec = {
  extract: { fps: number; pano_width: number; sdk_image: string }
  faces: { fov_deg: number; size: number; yaws: number[] }
  gate: { window: number; band: [number, number] }
  embed: { model_name: string; img_size: number; batch_size: number }
  dedup: { tau: number; rule: string }
  embed_model_used: string
}

export type WalkDetail = {
  id: string
  faces: number
  groups: Group[]
  meta: WalkMeta
  stages: Stages
  pipeline: PipelineSpec
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
}

export type Decision = {
  anchorIdx: number
  action: 'pick' | 'drop' | 'restore'
  pickIdx?: number
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`)
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
export const fetchJobs = () => req<Job[]>('/api/jobs')

// staged settings, applied to an existing walk (Select re-runs from the
// cached embeddings, so this returns in seconds)
export type Pending = { tau?: number; rule?: 'fixed' | 'auto' }

export const postRerun = (walkId: string, p: Pending) =>
  req<{ faces: number; anchors: number; absorbed: number }>(
    `/api/walks/${encodeURIComponent(walkId)}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
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
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText))
      else reject(new Error(`${xhr.status} ${xhr.statusText} for /api/ingest`))
    }
    xhr.onerror = () => reject(new Error('network error during upload'))
    xhr.send(form)
  })
