/** Asset imports resolve to their bundled URL (Vite). */
declare module '*.svg' {
  const src: string
  export default src
}

declare module '*.png' {
  const src: string
  export default src
}

declare module '*.jpg' {
  const src: string
  export default src
}

/** Side-effect CSS imports (e.g. '@xyflow/react/dist/style.css'); Vite bundles them. */
declare module '*.css'
