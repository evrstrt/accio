import { useEffect, useMemo, useRef, useState } from 'react'
import { thumb } from './api'
import type { Segmentation, SegFrame } from './api'
import { classColour } from './labels'

// masks drawn over their frames. The mask on disk is class indices; colours
// are chosen here.

const css = (c: [number, number, number], a = 1) =>
  `rgba(${c[0]},${c[1]},${c[2]},${a})`

/** Draw the frame, then the class boundaries (pixels with a differently labelled neighbour). */
function paint(canvas: HTMLCanvasElement, frame: HTMLImageElement,
               mask: HTMLImageElement, labels: string[], size: number,
               fill: boolean) {
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(frame, 0, 0, size, size)

  // nearest neighbour: smoothing a label image averages class indices into classes never predicted
  const off = document.createElement('canvas')
  off.width = off.height = size
  const octx = off.getContext('2d', { willReadFrequently: true })!
  octx.imageSmoothingEnabled = false
  octx.drawImage(mask, 0, 0, size, size)
  const m = octx.getImageData(0, 0, size, size).data

  const out = ctx.getImageData(0, 0, size, size)
  const px = out.data
  const at = (x: number, y: number) => m[(y * size + x) * 4]
  // classColour hashes a string; do not call it per pixel
  const palette: [number, number, number][] = []
  const colourOf = (c: number) =>
    palette[c] ?? (palette[c] = classColour(labels[c] ?? String(c)))
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const c = at(x, y)
      const edge = (x > 0 && at(x - 1, y) !== c) || (y > 0 && at(x, y - 1) !== c)
        || (x < size - 1 && at(x + 1, y) !== c)
        || (y < size - 1 && at(x, y + 1) !== c)
      if (!edge && !fill) continue
      const colour = colourOf(c)
      const i = (y * size + x) * 4
      const a = edge ? 1 : 0.22
      px[i] = px[i] * (1 - a) + colour[0] * a
      px[i + 1] = px[i + 1] * (1 - a) + colour[1] * a
      px[i + 2] = px[i + 2] * (1 - a) + colour[2] * a
    }
  }
  ctx.putImageData(out, 0, 0)
}

function load(src: string): Promise<HTMLImageElement> {
  return new Promise((done, fail) => {
    const img = new Image()
    img.onload = () => done(img)
    img.onerror = () => fail(new Error(src))
    img.src = src
  })
}

/** Paints when scrolled near, not on mount: a few hundred tiles painting in one tick locks the tab. */
function Overlay({ frame, labels, size, fill }: {
  frame: SegFrame
  labels: string[]
  size: number
  fill: boolean
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [near, setNear] = useState(false)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el || near) return
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { setNear(true); io.disconnect() }
    }, { rootMargin: '400px' })
    io.observe(el)
    return () => io.disconnect()
  }, [near])

  useEffect(() => {
    // wait for the labels, or the colours disagree with the legend
    if (!near || !labels.length) return
    let live = true
    // the 1024 original is about fifty times the pixels of a 220 px tile
    const small = size <= 256
    Promise.all([load(small ? thumb(frame.url) : frame.url),
                 load(small ? thumb(frame.mask) : frame.mask)])
      .then(([img, mask]) => {
        if (live && ref.current) paint(ref.current, img, mask, labels, size, fill)
      })
      .catch(() => { if (live) setFailed(true) })
    return () => { live = false }
  }, [near, frame.url, frame.mask, labels, size, fill])

  return (
    <span className="overlay">
      <canvas ref={ref} width={size} height={size} />
      {/* a failed load would otherwise look like a tile not yet painted */}
      {failed && <span className="overlay-bad" title="this frame or its mask
                       could not be loaded" />}
    </span>
  )
}

/** Class name per mask index, from the table the run records. */
function labelTable(labels: Segmentation['labels']): string[] {
  const out: string[] = []
  for (const [i, name] of Object.entries(labels ?? {})) out[Number(i)] = name
  return out
}

export default function Segments({ seg }: { seg: Segmentation }) {
  const [open, setOpen] = useState<string | null>(null)
  const [fill, setFill] = useState(false)
  // a dep of every Overlay's paint effect, so it must keep its identity across renders
  const labels = useMemo(() => labelTable(seg.labels), [seg.labels])
  const shown = seg.frames.find((f) => f.face === open)

  return (
    <>
      <header className="walk-header">
        <div className="header-row">
          <div className="funnel">
            <span><span className="n">{seg.frames.length}</span>
              {' '}<span className="label">frames masked</span></span>
          </div>
          <div className="run-line">
            {seg.model.split('/').pop()}
            <span className="sep">·</span>
            <label className="seg-toggle">
              <input type="checkbox" checked={fill}
                     onChange={(e) => setFill(e.target.checked)} />
              shade the areas
            </label>
          </div>
        </div>
        <div className="meta-line">
          {seg.classMix.map((c) => (
            <span className="chip" key={c.name}>
              <i style={{ background: css(classColour(c.name)) }} />
              {c.name} {Math.round(c.share * 100)}%
            </span>
          ))}
        </div>
      </header>

      {shown && (
        <div className="seg-open">
          <Overlay frame={shown} labels={labels} size={512} fill={fill} />
          <div className="seg-open-side">
            <div className="seg-open-head">
              t={shown.tSec.toFixed(1)}s y{shown.yaw}
              <button className="pill-discard" onClick={() => setOpen(null)}>
                Close
              </button>
            </div>
            <div className="runs">
              {Object.entries(shown.classes).map(([name, share]) => (
                <div className="mix" key={name}>
                  <span className="mix-name">
                    <i className="dot" style={{ background: css(classColour(name)) }} />
                    {name}
                  </span>
                  <span className="mix-bar">
                    <span style={{ width: `${Math.round(share * 100)}%`,
                                   background: css(classColour(name), 0.75) }} />
                  </span>
                  <span className="mix-pct">{Math.round(share * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="grid">
        {seg.frames.map((f) => (
          <button
            key={f.face}
            className={`tile${open === f.face ? ' open' : ''}`}
            onClick={() => setOpen(open === f.face ? null : f.face)}
            title={Object.keys(f.classes).slice(0, 4).join(', ')}
          >
            <Overlay frame={f} labels={labels} size={220} fill={fill} />
            <span className="meta">
              <span>t={f.tSec.toFixed(1)}s</span>
              <span>y{f.yaw}</span>
            </span>
          </button>
        ))}
      </div>
    </>
  )
}
