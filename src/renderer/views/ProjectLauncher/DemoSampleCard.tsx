import { studio } from '@/lib/studio'
import characterCover from '../../assets/h3-character.jpg'

/** A published workflow that builds a portable .char from a handful of references. */
const DEMO_URL =
  'https://inlinestudio.art/workflows/minimax-h3-consistent-characters-with-references-with-char-model'

/** The "Try a demo" card: opens the consistent-character workflow in the browser. */
export function DemoSampleCard(): React.JSX.Element {
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <h2 className="mb-3 text-sm font-medium text-zinc-300">Try a demo</h2>
      <button
        onClick={() => void studio().shell.openExternal(DEMO_URL)}
        className="group block w-full overflow-hidden rounded-lg border border-border bg-panel text-left transition-colors hover:border-accent"
      >
        <div className="aspect-video w-full overflow-hidden bg-panel">
          <img
            src={characterCover}
            alt="Consistent Character with Minimax H3"
            className="h-full w-full object-cover transition-transform group-hover:scale-[1.02]"
          />
        </div>
        <div className="flex items-center gap-2 p-3">
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-semibold text-zinc-100">
              Consistent Character with Minimax H3
            </span>
            <span className="block text-xs text-zinc-400">
              Drop 2-5 reference images, describe your character, and export a portable .char to use
              across videos.
            </span>
          </span>
          <ExternalIcon />
        </div>
      </button>
    </section>
  )
}

function ExternalIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="h-4 w-4 shrink-0 text-zinc-500 group-hover:text-accent"
    >
      <path d="M14 3h7v7" />
      <path d="M10 14 21 3" />
      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
    </svg>
  )
}
