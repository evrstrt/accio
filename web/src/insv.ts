// Reading an .insv well enough to refuse it before it is uploaded.
//
// An .insv is an MP4 with Insta360's own trailer bolted on, so the frame size
// is in a `tkhd` box: width and height are its last eight bytes, 16.16 fixed
// point. Parsing those beats handing the file to a <video> element, which
// needs the media pipeline and does not run in a background tab.
//
// The shape is what says whether a file is a whole recording. Two fisheye
// circles side by side make a 2:1 frame and stand alone; one circle per file
// makes a square one, and those always come as a _00_/_10_ pair. Uploading
// half a pair stitches the front hemisphere over the back, so it is worth
// catching here rather than after a few hundred megabytes have gone up.

const SCAN = 8 << 20   // moov sits at one end or the other; read both

export type FrameSize = { w: number; h: number }

/** The largest track's frame size, or null if no `tkhd` turned up. */
export function frameSize(buf: ArrayBuffer): FrameSize | null {
  const b = new Uint8Array(buf)
  const dv = new DataView(buf)
  let best: FrameSize | null = null
  for (let i = 4; i + 4 < b.length; i++) {
    // 'tkhd'
    if (b[i] !== 0x74 || b[i + 1] !== 0x6b || b[i + 2] !== 0x68 || b[i + 3] !== 0x64) {
      continue
    }
    const size = dv.getUint32(i - 4)
    const end = i - 4 + size
    if (size < 84 || end > b.length) continue
    const w = dv.getUint32(end - 8) >> 16
    const h = dv.getUint32(end - 4) >> 16
    // audio tracks carry a tkhd too, sized 0x0; the video track is the big one
    if (w > 0 && h > 0 && (!best || w * h > best.w * best.h)) best = { w, h }
  }
  return best
}

export async function readFrameSize(file: File): Promise<FrameSize | null> {
  for (const part of [file.slice(0, SCAN), file.slice(Math.max(0, file.size - SCAN))]) {
    const found = frameSize(await part.arrayBuffer())
    if (found) return found
  }
  return null
}

export const lensesInFrame = (s: FrameSize) => (s.w >= s.h * 1.5 ? 2 : 1)

/** Why this selection cannot be processed, or "" if it can. Mirrors the
    server's guards so the answer arrives before the upload, not after it. */
export function whyNotReady(files: File[], sizes: (FrameSize | null)[]): string {
  if (!files.length) return 'Drop the .insv here to start'
  const names = files.map((f) => f.name)
  const front = names.find((n) => n.includes('_00_'))
  if (names.some((n) => n.includes('_10_')) && !front) {
    return 'That is the back lens. Add the _00_ file alongside it.'
  }
  const main = sizes[Math.max(0, names.findIndex((n) => n === front))] ?? sizes[0]
  if (main && lensesInFrame(main) === 1 && files.length < 2) {
    const want = front ? front.replace('_00_', '_10_') : 'the second'
    return `${main.w}×${main.h} is one lens of a pair. Add ${want} too.`
  }
  return ''
}
