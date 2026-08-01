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

export type WalkDetail = {
  id: string
  faces: number
  groups: Group[]
  meta: WalkMeta
}

export type Job = {
  id: number
  walkId: string
  status: 'queued' | 'running' | 'done' | 'error'
  stage: string
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
