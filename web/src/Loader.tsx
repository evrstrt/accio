// The loading mark, the same one the ciq-demo free-roam viewer uses so the
// two tools read as one product: the Technosoft logo drawing itself on, a
// fixed extrusion whose contours draw in order (back face, walls, front face,
// then the four cap lines that make the T read) on a 3.4 s loop.
//
// The geometry is baked. In the study it is generated from the octagon's
// vertices, the counter ellipse and a depth vector; the depth never changes
// here, so the one frame that generator produces is checked in instead. The
// class names are load-bearing: `.bk` back face, `.w` walls, `.frfill` the
// occluder, `.tbg` the T tint, `.fr` front contours, `.cap` the four lines.
// Timing lives in index.css under `.tsx-loader`.

const MARK = `
  <svg class="wf" xmlns="http://www.w3.org/2000/svg" viewBox="-5 0 842 850">
  <path class="bk" fill-rule="evenodd" d="M298.6 -24.5L638.8 -24.4L880.0 216.4L880.0 556.1L638.7 796.6L299.1 796.5L58.4 556.4L58.3 215.9ZM794.5 386.1A325.6 319.8 0 1 0 143.3 386.1A325.6 319.8 0 1 0 794.5 386.1Z"/>
  <path class="bk" d="M408.1 71.9A325.6 319.8 0 0 1 529.9 72.0L529.9 700.2A325.6 319.8 0 0 1 408.1 700.3Z"/>
  <path class="w" d="M250.6 14.5L590.8 14.6L638.8 -24.4L298.6 -24.5Z"/>
  <path class="w" d="M590.8 14.6L832.0 255.4L880.0 216.4L638.8 -24.4Z"/>
  <path class="w" d="M832.0 255.4L832.0 595.1L880.0 556.1L880.0 216.4Z"/>
  <path class="w" d="M832.0 595.1L590.7 835.6L638.7 796.6L880.0 556.1Z"/>
  <path class="w" d="M481.9 111.0L481.9 739.2L529.9 700.2L529.9 72.0Z"/>
  <path class="w" d="M626.2 673.3A325.6 319.8 0 0 0 215.6 176.9L263.6 137.9A325.6 319.8 0 0 1 674.2 634.3Z"/>
  <path class="frfill" fill-rule="evenodd" d="M250.6 14.5L590.8 14.6L832.0 255.4L832.0 595.1L590.7 835.6L251.1 835.5L10.4 595.4L10.3 254.9ZM746.5 425.1A325.6 319.8 0 1 0 95.3 425.1A325.6 319.8 0 1 0 746.5 425.1Z"/>
  <path class="frfill" d="M360.1 110.9A325.6 319.8 0 0 1 481.9 111.0L481.9 739.2A325.6 319.8 0 0 1 360.1 739.3Z"/>
  <path class="tbg" fill-rule="evenodd" d="M250.6 14.5L590.8 14.6L832.0 255.4L832.0 595.1L590.7 835.6L251.1 835.5L10.4 595.4L10.3 254.9ZM746.5 425.1A325.6 319.8 0 1 0 95.3 425.1A325.6 319.8 0 1 0 746.5 425.1ZM832.0 255.4L832.0 595.1L590.7 835.6L590.7 698.0A325.6 319.8 0 0 0 721.9 303.1ZM10.3 254.9L10.4 595.4L251.1 835.5L251.1 698.0A325.6 319.8 0 0 1 119.9 303.1Z"/>
  <path class="tbg" d="M360.1 110.9A325.6 319.8 0 0 1 481.9 111.0L481.9 739.2A325.6 319.8 0 0 1 360.1 739.3Z"/>
  <path class="fr" d="M250.6 14.5L590.8 14.6L832.0 255.4L832.0 595.1L590.7 835.6L251.1 835.5L10.4 595.4L10.3 254.9Z"/>
  <path class="fr" d="M481.9 111.0A325.6 319.8 0 0 1 481.9 739.2Z"/>
  <path class="fr" d="M360.1 110.9A325.6 319.8 0 0 0 360.1 739.3Z"/>
  <path class="cap" d="M721.9 303.1L832.0 255.4"/>
  <path class="cap" d="M590.7 698.0L590.7 835.6"/>
  <path class="cap" d="M119.9 303.1L10.3 254.9"/>
  <path class="cap" d="M251.1 698.0L251.1 835.5"/></svg>
`

/** The mark is the progress indicator; the lines under it say what is coming
    and, if it stops coming, why. No bar: there is no byte count to put in one. */
export default function Loader({ label, sub, error }: {
  label: string
  sub?: string
  error?: string | null
}) {
  return (
    <div className="loading">
      <div className="tsx-loader" dangerouslySetInnerHTML={{ __html: MARK }} />
      <div className="z-mono">{error ? label : `${label}…`}</div>
      {sub && <div className="z-mono">{sub}</div>}
      {error && <div className="z-mono load-error">{error}</div>}
    </div>
  )
}
