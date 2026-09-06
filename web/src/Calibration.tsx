import { useEffect, useRef } from 'react'
import { thumb } from './api'
import type { Calibration } from './api'

// how tau was placed. It comes from the far pairs; the reference pairs shown
// below are a health check of the stitch and backbone, not the threshold.

export default function CalibrationModal({ walkId, calib, onClose }: {
  walkId: string
  calib: Calibration
  onClose: () => void
}) {
  const url = (p: string) => `/api/walks/${encodeURIComponent(walkId)}/${p}`
  const ms = Math.round(calib.gapSeconds * 1000)
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => { closeRef.current?.focus() }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <>
      <div className="z-scrim open" onClick={onClose} />
      <div className="z-dialog calib" role="dialog" aria-label="Calibration">
        <div className="calib-head">
          <h2>Calibration</h2>
          <button className="icon-btn" ref={closeRef} onClick={onClose} aria-label="Close">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <p className="calib-note">
          τ is placed on the {calib.far.n.toLocaleString()} pairs of faces at
          one heading more than {calib.farSeconds}s apart. Those are different
          places, and {calib.falseMergePct}% of them are allowed to merge, so τ
          sits at the {(100 - calib.falseMergePct).toFixed(calib.falseMergePct % 1 ? 1 : 0)}th
          {' '}percentile of what elsewhere scores here. Every face stays on
          disk, so a lower budget brings merged ones back.
        </p>

        <div className="calib-stats">
          <div><span className="calib-n">{calib.far.median}</span> far median</div>
          <div><span className="calib-n">{calib.far.p95}</span> p95</div>
          <div><span className="calib-n">{calib.far.p99}</span> p99</div>
          <div><span className="calib-n">{calib.far.max}</span> highest</div>
          <div className="calib-tau">
            τ <span className="calib-n">{calib.tau ?? 'none'}</span>
          </div>
        </div>

        {calib.tau === null && (
          <div className="calib-warn">
            This walk has only {calib.far.n} pairs more than {calib.farSeconds}s
            apart, too few to place a threshold on. Use the fixed rule, or a
            longer walk.
          </div>
        )}

        <p className="calib-note">
          The {calib.reference.n} pairs below are the health check: a kept frame
          against the raw frame {ms} ms later, over {calib.samples} panoramas
          spread across the walk. Identical content, so this is the ceiling for
          this stitch and backbone.
        </p>

        <div className="calib-stats">
          <div><span className="calib-n">{calib.reference.median}</span> median</div>
          <div><span className="calib-n">{calib.reference.p05}</span> p05</div>
          <div><span className="calib-n">{calib.reference.min}</span> lowest</div>
          <div><span className="calib-n">{calib.reference.n}</span> pairs</div>
        </div>

        {calib.referenceError && (
          <div className="calib-warn">
            The health check did not run: {calib.referenceError}. τ is
            unaffected, it comes from the far pairs, but nothing here vouches
            for the stitch on this walk.
          </div>
        )}

        {!calib.healthy && !calib.referenceError && (
          <div className="calib-warn">
            Identical frames score lower here than they should. The stitch, the
            exposure or the backbone is likely misbehaving on this walk.
          </div>
        )}

        <div className="calib-grid">
          {calib.pairs.map((p) => (
            <div className="calib-pair" key={`${p.pano}-${p.yaw}`}>
              <div className="calib-imgs">
                <img src={thumb(url(`faces/${p.face}`))}
                     alt={`kept frame at ${p.tSec}s`} loading="lazy" />
                <img src={thumb(url(`calib/${p.neighbour}`))}
                     alt={`${ms} ms later`} loading="lazy" />
              </div>
              <div className="calib-meta">
                <span>t={p.tSec}s y{p.yaw}</span>
                <span className={calib.tau !== null && p.cosine <= calib.tau
                                 ? 'calib-low' : ''}>
                  {p.cosine.toFixed(3)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
