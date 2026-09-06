import { STAGES } from './api'

/** "vit_base_patch16_dinov3.lvd1689m" -> "DINOv3 ViT-B/16" */
export function backboneLabel(name: string): string {
  const m = /vit_(base|small|large)_patch(\d+)_(\w+?)[._]/.exec(name)
  if (!m) return name
  const size = { base: 'B', small: 'S', large: 'L' }[m[1]] ?? m[1]
  const family = { dinov3: 'DINOv3', dinov2: 'DINOv2', clip: 'CLIP' }[m[3]] ?? m[3]
  return `${family} ViT-${size}/${m[2]}`
}

/** "nvidia/segformer-b4-finetuned-ade-512-512" -> "SegFormer-B4 · ADE20K" */
export function segmenterLabel(name: string): string {
  const seg = /segformer-(b\d)-finetuned-ade/.exec(name)
  if (seg) return `SegFormer-${seg[1].toUpperCase()} · ADE20K`
  if (name.includes('mask2former')) return 'Mask2Former-L · ADE20K'
  if (name.includes('oneformer')) return 'OneFormer-L · ADE20K'
  if (name.includes('grounding-dino')) return 'Grounding DINO + SAM · your classes'
  return name
}

// block titles on the pipeline canvas, by stage id
export const TITLES: Record<string, string> = {
  video: 'Video', stitch: 'Stitch', gate: 'Gate', faces: 'Faces', embed: 'Embed',
  calibrate: 'Calibrate', select: 'Select', segment: 'Segment', review: 'Review',
}

/** "Faces, Embed, Select": the stages an edit at `from` re-runs. */
export function rerunLabel(from: string): string {
  const i = STAGES.indexOf(from)
  if (i < 0) return ''
  return STAGES.slice(i).map((id) => TITLES[id] ?? id).join(', ')
}

/** A deterministic colour per class name, stable across frames and walks. */
export function classColour(name: string): [number, number, number] {
  let h = 0
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0
  const hue = Math.abs(h) % 360
  // fixed saturation and lightness; only the hue varies
  const f = (n: number) => {
    const k = (n + hue / 30) % 12
    const c = 0.58 * Math.min(1, Math.max(-1, Math.min(k - 3, 9 - k)))
    return Math.round(255 * (0.58 - c * 0.42))
  }
  return [f(0), f(8), f(4)]
}
