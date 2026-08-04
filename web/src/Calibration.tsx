import type { Calibration } from './api'

// How tau was arrived at, with the frames it was measured on. Each pair is a
// kept frame and the raw frame a fraction of a second later: the same scene,
// so whatever they score is this walk's ceiling for "identical".

export default function CalibrationModal({ walkId, calib, onClose }: {
  walkId: string
  calib: Calibration
  onClose: () => void
}) {
  const url = (p: string) => `/api/walks/${encodeURIComponent(walkId)}/${p}`
  const ms = Math.round(calib.gapSeconds * 1000)

  return (
    <>
      <div className="z-scrim open" onClick={onClose} />
      <div className="z-dialog calib" role="dialog" aria-label="Calibration">
        <div className="calib-head">
          <h2>Calibration</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <p className="calib-note">
          Every kept frame here is paired with the raw frame {ms} ms later: the
          same scene, so what they score is what <em>identical</em> means for
          this walk. τ is set at the {calib.quantile}th percentile of those
          scores, so a merge needs two frames about as alike as a pair taken
          {' '}{ms} ms apart.
        </p>

        <div className="calib-stats">
          <div><span className="calib-n">{calib.reference.median}</span> median</div>
          <div><span className="calib-n">{calib.reference.p05}</span> p05</div>
          <div><span className="calib-n">{calib.reference.min}</span> lowest</div>
          <div><span className="calib-n">{calib.reference.n}</span> pairs</div>
          <div className="calib-tau">
            τ <span className="calib-n">{calib.tau}</span>
          </div>
        </div>

        {!calib.healthy && (
          <div className="calib-warn">
            The reference is lower than identical frames should score. The
            stitch, the exposure or the backbone is likely misbehaving on this
            walk, and a threshold above the reference would merge nothing.
          </div>
        )}

        <div className="calib-grid">
          {calib.pairs.map((p) => (
            <div className="calib-pair" key={`${p.pano}-${p.yaw}`}>
              <div className="calib-imgs">
                <img src={url(`faces/${p.face}`)} alt={`kept frame at ${p.tSec}s`}
                     loading="lazy" />
                <img src={url(`calib/${p.neighbour}`)} alt={`${ms} ms later`}
                     loading="lazy" />
              </div>
              <div className="calib-meta">
                <span>t={p.tSec}s y{p.yaw}</span>
                <span className={p.cosine <= calib.tau ? 'calib-low' : ''}>
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
