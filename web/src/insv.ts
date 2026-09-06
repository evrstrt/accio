// reads enough of an .insv to refuse it before upload. It is an MP4 with an
// Insta360 trailer; width and height are the last eight bytes of `tkhd`, 16.16
// fixed point. A 2:1 frame holds both lenses; a square frame is one lens of a
// _00_/_10_ pair, and half a pair stitches the front hemisphere over the back.

const SCAN = 8 << 20   // moov sits at one end or the other; read both

export type FrameSize = { w: number; h: number }

/** The largest track's frame size, or null if no `tkhd` turned up. */
function frameSize(buf: ArrayBuffer): FrameSize | null {
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

const lensesInFrame = (s: FrameSize) => (s.w >= s.h * 1.5 ? 2 : 1)

// the Insta360 trailer ends in this marker and holds the camera model in a
// protobuf-shaped record
const MAGIC = '8db42d694ccc418790edff439fe026bf'
const TRAILER = 2 << 20

export async function readCamera(file: File): Promise<string> {
  const tail = new Uint8Array(
    await file.slice(Math.max(0, file.size - TRAILER)).arrayBuffer())
  const text = new TextDecoder('latin1').decode(tail)
  if (!text.endsWith(MAGIC)) return ''
  // field 2, length-delimited: \x12 <len> "Insta360 ..."
  // oxlint-disable-next-line no-control-regex
  const at = /\x12([\x01-\x40])(Insta360 [ -~]{1,24})/.exec(text)
  return at ? at[2].slice(0, at[1].charCodeAt(0)) : ''
}

// VID_20260728_114811_..., the camera's local clock; the container's creation_time is UTC
const STAMP = /VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/

export function stampFromName(name: string): { date: string; time: string } {
  const m = STAMP.exec(name)
  return m
    ? { date: `${m[1]}-${m[2]}-${m[3]}`, time: `${m[4]}:${m[5]}` }
    : { date: '', time: '' }
}

/** Why this selection cannot be processed, or "" if it can. Mirrors the server's guards. */
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
