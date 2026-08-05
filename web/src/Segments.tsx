import { useEffect, useRef, useState } from 'react'
import type { Segmentation, SegFrame } from './api'

// What the segmenter saw, drawn over the frames it saw it in.
//
// The mask on disk is class indices, so the colour is chosen here rather than
// baked in: the file stays the label. Outlines rather than a filled overlay,
// because judging a mask means looking at whether its boundary follows the
// thing, and a wash over the whole frame hides exactly that.

/** A stable colour per class name. Deterministic so a class keeps its colour
    between frames and between walks, which is what makes the grid readable. */
export function classColour(name: string): [number, number, number] {
  let h = 0
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0
  const hue = Math.abs(h) % 360
  // fixed saturation and lightness: the hue carries the identity, and equal
  // weight keeps one class from shouting over another
  const f = (n: number) => {
    const k = (n + hue / 30) % 12
    const c = 0.58 * Math.min(1, Math.max(-1, Math.min(k - 3, 9 - k)))
    return Math.round(255 * (0.58 - c * 0.42))
  }
  return [f(0), f(8), f(4)]
}

const css = (c: [number, number, number], a = 1) =>
  `rgba(${c[0]},${c[1]},${c[2]},${a})`

/** Draw the frame, then the class boundaries over it.
    A pixel is a boundary when a neighbour belongs to another class, which on a
    label image is all a contour is. */
function paint(canvas: HTMLCanvasElement, frame: HTMLImageElement,
               mask: HTMLImageElement, labels: string[], size: number,
               fill: boolean) {
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(frame, 0, 0, size, size)

  // nearest neighbour for the mask: smoothing a label image averages class
  // indices together and invents classes that were never predicted
  const off = document.createElement('canvas')
  off.width = off.height = size
  const octx = off.getContext('2d', { willReadFrequently: true })!
  octx.imageSmoothingEnabled = false
  octx.drawImage(mask, 0, 0, size, size)
  const m = octx.getImageData(0, 0, size, size).data

  const out = ctx.getImageData(0, 0, size, size)
  const px = out.data
  const at = (x: number, y: number) => m[(y * size + x) * 4]
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const c = at(x, y)
      const edge = (x > 0 && at(x - 1, y) !== c) || (y > 0 && at(x, y - 1) !== c)
        || (x < size - 1 && at(x + 1, y) !== c)
        || (y < size - 1 && at(x, y + 1) !== c)
      if (!edge && !fill) continue
      const colour = classColour(labels[c] ?? String(c))
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

function Overlay({ frame, labels, size, fill }: {
  frame: SegFrame
  labels: string[]
  size: number
  fill: boolean
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    let live = true
    Promise.all([load(frame.url), load(frame.mask)])
      .then(([img, mask]) => {
        if (live && ref.current) paint(ref.current, img, mask, labels, size, fill)
      })
      .catch(() => {})
    return () => { live = false }
  }, [frame.url, frame.mask, labels, size, fill])
  return <canvas ref={ref} width={size} height={size} />
}

/** Which mask index is which class. The run records it, because the mask is
    indices and only the model knows what they mean; walks segmented before
    that fall back to matching each frame's classes against its pixel counts,
    which is right whenever no two classes cover exactly the same area. */
function useLabels(seg: Segmentation): string[] {
  const [labels, setLabels] = useState<string[]>([])
  useEffect(() => {
    if (Object.keys(seg.labels ?? {}).length) {
      const out: string[] = []
      for (const [i, name] of Object.entries(seg.labels)) out[Number(i)] = name
      setLabels(out)
      return
    }
    const first = seg.frames[0]
    if (!first) return
    let live = true
    load(first.mask).then((mask) => {
      const off = document.createElement('canvas')
      off.width = mask.width
      off.height = mask.height
      const ctx = off.getContext('2d', { willReadFrequently: true })!
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(mask, 0, 0)
      const data = ctx.getImageData(0, 0, off.width, off.height).data
      const counts = new Map<number, number>()
      for (let i = 0; i < data.length; i += 4) {
        counts.set(data[i], (counts.get(data[i]) ?? 0) + 1)
      }
      const byShare = [...counts.entries()].sort((a, b) => b[1] - a[1])
      const names = Object.keys(first.classes)
      const out: string[] = []
      byShare.slice(0, names.length).forEach(([idx], i) => { out[idx] = names[i] })
      if (live) setLabels(out)
    }).catch(() => {})
    return () => { live = false }
  }, [seg])
  return labels
}

export default function Segments({ seg }: { seg: Segmentation }) {
  const [open, setOpen] = useState<string | null>(null)
  const [fill, setFill] = useState(false)
  const labels = useLabels(seg)
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
