import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchCalibration, fetchJobs, fetchSegmentation, fetchWalk, fetchWalks,
         isMissing, postDecision } from './api'
import type { Calibration, Decision, Job, Segmentation, WalkDetail,
              WalkSummary } from './api'

export const inFlight = (j: Job) => j.status === 'queued' || j.status === 'running'

export const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

// a 404 is the absence of a record; anything else is a failure
const orNone = (e: unknown): null => {
  if (isMissing(e)) return null
  throw e
}

/** The walk with one decision applied locally, without a refetch. */
function applied(walk: WalkDetail, d: Decision): WalkDetail {
  const groups = walk.groups.map((g) => {
    if (g.anchor.idx !== d.anchorIdx) return g
    if (d.action === 'drop') return { ...g, dropped: true }
    if (d.action === 'restore') return { ...g, dropped: false }
    return { ...g, pick: d.pickIdx ?? g.auto }
  })
  const dropped = groups.filter((g) => g.dropped).length
  return {
    ...walk,
    groups,
    stages: {
      ...walk.stages,
      dropped,
      overridden: groups.filter((g) => g.pick !== g.auto).length,
      kept: groups.length - dropped,
    },
  }
}

/** The job list, polled every 2 s while a run is in flight and on window focus.
    `onLanded` fires when the selected walk's own run leaves flight, `onIdle`
    after a poll that found nothing running; a rejected `onIdle` counts as offline. */
export function useJobs(selected: string | null, onLanded: (id: string) => void,
                        onIdle: () => Promise<unknown>) {
  const [jobs, setJobs] = useState<Job[]>([])
  // the server stopped answering; clears itself on the next answer
  const [offline, setOffline] = useState(false)
  const watching = useRef<number | null>(null)

  // polling only while something runs: /api/walks parses every manifest of every walk
  const poll = () => fetchJobs().then((js) => {
    setJobs(js)
    setOffline(false)
    const mine = js.find((j) => j.walkId === selected)
    // our job leaving flight means the frames on disk changed
    if (mine && selected && inFlight(mine)) watching.current = mine.id
    else if (mine && !inFlight(mine) && selected
             && watching.current === mine.id) {
      watching.current = null
      onLanded(selected)
      return
    }
    if (!js.some(inFlight)) {
      onIdle().then(() => setOffline(false), () => setOffline(true))
    }
  }).catch(() => setOffline(true))

  // the interval and the focus listener read the current closure through the ref,
  // so a walk click does not restart the timer
  const pollRef = useRef(poll)
  useEffect(() => { pollRef.current = poll })

  // a run started in another tab or before a reload still shows in the rail
  useEffect(() => { fetchJobs().then(setJobs).catch(() => {}) }, [])

  const active = jobs.some(inFlight)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => pollRef.current(), 2000)
    return () => clearInterval(t)
  }, [active])

  useEffect(() => {
    const onFocus = () => { pollRef.current() }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [])

  return { jobs, setJobs, offline }
}

/** The selected walk with its calibration and segmentation. A slow response
    for a previously selected walk is dropped rather than landing under the
    current one: group indices are per-walk integers. `onError` must be stable. */
export function useWalk(selected: string | null,
                        setWalks: (ws: WalkSummary[]) => void,
                        onError: (msg: string) => void) {
  const [walk, setWalk] = useState<WalkDetail | null>(null)
  const [calib, setCalib] = useState<Calibration | null>(null)
  const [seg, setSeg] = useState<Segmentation | null>(null)
  const [unreachable, setUnreachable] = useState<string | null>(null)
  const selectedRef = useRef<string | null>(null)

  useEffect(() => {
    selectedRef.current = selected
    if (!selected) return
    const id = selected
    setWalk(null)
    setCalib(null)
    setSeg(null)
    setUnreachable(null)
    // a queued or running walk has no manifest yet, so a 404 here is expected
    fetchWalk(id)
      .then((w) => {
        if (selectedRef.current !== id) return
        setWalk(w)
        setUnreachable(null)
      })
      .catch((e) => {
        if (selectedRef.current !== id) return
        setWalk(null)
        setUnreachable(message(e))
      })
    // measured on every run, so only walks made before it exists lack a record
    fetchCalibration(id).catch(orNone)
      .then((c) => { if (selectedRef.current === id) setCalib(c) })
      .catch((e) => { if (selectedRef.current === id) onError(message(e)) })
    // 404s until the Segment stage has been turned on for this walk
    fetchSegmentation(id).catch(orNone)
      .then((s) => { if (selectedRef.current === id) setSeg(s) })
      .catch((e) => { if (selectedRef.current === id) onError(message(e)) })
  }, [selected, onError])

  const reload = useCallback((id: string) =>
    Promise.all([
      fetchWalk(id),
      fetchWalks(),
      fetchCalibration(id).catch(orNone),
      fetchSegmentation(id).catch(orNone),
    ]).then(([w, ws, c, s]) => {
      setWalks(ws)
      if (selectedRef.current !== id) return
      setWalk(w)
      setCalib(c)
      setSeg(s)
    }), [setWalks])

  /** Applied locally at once; on refusal the walk is refetched so the grid
      matches the server, and the error is rethrown. */
  const decide = useCallback((d: Decision): Promise<void> => {
    if (!selected) return Promise.resolve()
    setWalk((w) => (w ? applied(w, d) : w))
    return postDecision(selected, d).then(() => {}, (e: unknown) => {
      fetchWalk(selected)
        .then((w) => { if (selectedRef.current === selected) setWalk(w) })
        .catch(() => {})
      throw e
    })
  }, [selected])

  return { walk, setWalk, calib, seg, unreachable, reload, decide }
}
