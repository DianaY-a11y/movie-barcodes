import { useEffect, useRef, useState } from 'react'
import './App.css'
import barcodes from './data/barcodes.json'

const WAVE_DECAY    = 0.90   // activation × this each frame (~1 s to decay at 60 fps)
const WAVE_INJECT   = 0.10   // activation added per frame at cursor (gaussian peak)
const WAVE_SIGMA    = 5      // gaussian spread of injection (stripe units)
const WAVE_MAX_EXT  = 0.35   // max height extension above/below image (fraction of img height)
const WAVE_CENTER_W = 5      // max width multiplier at full activation

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

const GRAIN_COUNT = 60000

function GrainCanvas() {
  const canvasRef = useRef(null)
  const stateRef  = useRef(null)  // { particles, colors, W, H }
  const rafRef    = useRef(null)
  const mouseRef  = useRef(null)  // { x, y } canvas-relative, or null

  useEffect(() => {
    fetch('/barcodes/ChungkingExpress_vivid.json')
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(colors => {
        const canvas = canvasRef.current
        if (!canvas) return
        const W = canvas.offsetWidth  || canvas.width
        const H = canvas.offsetHeight || canvas.height
        canvas.width  = W
        canvas.height = H

        const particles = Array.from({ length: GRAIN_COUNT }, () => {
          const si    = Math.floor(Math.random() * colors.length)
          const hex   = colors[si].hex
          const r     = parseInt(hex.slice(1, 3), 16)
          const g     = parseInt(hex.slice(3, 5), 16)
          const b     = parseInt(hex.slice(5, 7), 16)
          return {
            homeX: (si / colors.length) * W,
            y:     Math.random() * H,
            vy:    0.2 + Math.random() * 0.45,
            phase: Math.random() * Math.PI * 2,
            amp:   3 + Math.random() * 10,
            color: `${r},${g},${b}`,
            alpha: 0.45 + Math.random() * 0.55,
            sz:    1 + Math.random() * 1.5,
          }
        })

        stateRef.current = { particles, colors, W, H }
      })
      .catch(() => {})

    const tick = (time) => {
      const canvas = canvasRef.current
      const state  = stateRef.current
      if (!canvas || !state) { rafRef.current = requestAnimationFrame(tick); return }

      const { particles, W, H } = state

      // Sync canvas size to CSS layout dimensions
      const cW = canvas.offsetWidth
      const cH = canvas.offsetHeight
      if (cW !== canvas.width || cH !== canvas.height) {
        canvas.width  = cW
        canvas.height = cH
        // Rescale homeX to new width
        const colors = state.colors
        particles.forEach(p => {
          // Recover stripe index from old homeX then recompute
          const si = Math.round((p.homeX / W) * (colors.length - 1))
          p.homeX = (si / colors.length) * cW
        })
        state.W = cW
        state.H = cH
      }

      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const mouse = mouseRef.current
      const RADIUS = 110

      for (const p of particles) {
        p.y += p.vy
        if (p.y > state.H) p.y -= state.H

        let x     = p.homeX + Math.sin(time * 0.0007 + p.phase) * p.amp
        let alpha = p.alpha

        if (mouse) {
          const dx   = x - mouse.x
          const dy   = p.y - mouse.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < RADIUS && dist > 0) {
            const t     = 1 - dist / RADIUS
            const force = t * 30
            x     += (dx / dist) * force
            alpha  = Math.min(1, alpha + t * 0.5)
          }
        }

        ctx.fillStyle = `rgba(${p.color},${alpha})`
        ctx.fillRect(x, p.y, p.sz, p.sz)
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  const handleMouseMove = (e) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  return (
    <div className="grain-outer">
      <div className="grain-wrap" onMouseMove={handleMouseMove} onMouseLeave={() => { mouseRef.current = null }}>
        <canvas ref={canvasRef} className="grain-canvas" />
      </div>
      <p className="grain-label">Chungking Express &mdash; Wong Kar-wai</p>
    </div>
  )
}

function Intro() {
  return (
    <section className="intro snap-section">
      <div className="intro-content">
        <h1>Film<br />Barcode</h1>
        <p className="intro-desc">
          A movie barcode compresses an entire film into a single image.
          Each vertical stripe is the dominant color of one scene —
          a film's complete visual palette, collapsed into one frame.
        </p>
        <p className="intro-desc">
          We collect two types of colors: the dominant, or natural, color of each scene, setting the overall tone of the scene, and the most vivid color, which reveals the scene's most striking hue.
        </p>
        <p className="intro-cta">
          Generate your own with the{' '}
          <a href="https://github.com/DianaY-a11y/movie-barcode-cli" target="_blank" rel="noopener noreferrer">
            movie-barcode CLI
          </a>
          .
        </p>
        <div className="scroll-hint">↓</div>
      </div>
      <GrainCanvas />
    </section>
  )
}

function BarcodeSection({ barcode, containerRef }) {
  const sectionRef  = useRef(null)
  const wrapperRef  = useRef(null)
  const canvasRef   = useRef(null)
  const actsRef     = useRef(new Float32Array(0))   // activation per stripe [0-1]
  const rgbsRef     = useRef([])                    // precomputed 'rgb(r,g,b)' strings
  const hiRef       = useRef(-1)                    // hovered stripe index (-1 = none)
  const rafRef      = useRef(null)
  const fetchedRef  = useRef(false)                 // ensure sidecar fetched only once

  const [t, setT]           = useState(1)
  const [active, setActive] = useState(false)
  const [colors, setColors] = useState([])
  const [hovered, setHovered] = useState(null)   // { x, hex, t } — tooltip only

  // ── Fetch sidecar color JSON — deferred until the section is first active ──
  useEffect(() => {
    if (!active || fetchedRef.current) return
    fetchedRef.current = true
    const url = `/barcodes/${barcode.filename.replace(/\.\w+$/, '.json')}`
    fetch(url)
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(setColors)
      .catch(() => {})
  }, [active, barcode.filename])

  // ── Precompute RGB strings when colors load ────────────────────────────────
  useEffect(() => {
    if (colors.length === 0) return
    actsRef.current = new Float32Array(colors.length)
    rgbsRef.current = colors.map(({ hex }) => {
      const r = parseInt(hex.slice(1, 3), 16)
      const g = parseInt(hex.slice(3, 5), 16)
      const b = parseInt(hex.slice(5, 7), 16)
      return `rgb(${r},${g},${b})`
    })
  }, [colors])

  // ── Scroll → blur / active ─────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current
    const section   = sectionRef.current
    if (!container || !section) return
    const update = () => {
      const rect = section.getBoundingClientRect()
      const vh   = window.innerHeight
      const mid  = rect.top + rect.height / 2
      const tVal = Math.min(Math.abs(mid - vh / 2) / (vh * 0.55), 1)
      setT(tVal)
      setActive(tVal < 0.12)
    }
    container.addEventListener('scroll', update, { passive: true })
    update()
    return () => container.removeEventListener('scroll', update)
  }, [containerRef])

  // ── RAF wave animation ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!active || colors.length === 0) {
      cancelAnimationFrame(rafRef.current)
      actsRef.current.fill(0)
      const cvs = canvasRef.current
      if (cvs) cvs.getContext('2d')?.clearRect(0, 0, cvs.width, cvs.height)
      return
    }

    const tick = () => {
      const canvas  = canvasRef.current
      const wrapper = wrapperRef.current
      const acts    = actsRef.current
      const rgbs    = rgbsRef.current
      const n       = acts.length
      if (!canvas || !wrapper || n === 0) { rafRef.current = requestAnimationFrame(tick); return }

      // Inject gaussian pulse at cursor position
      const hi = hiRef.current
      if (hi >= 0 && hi < n) {
        const iw = Math.ceil(WAVE_SIGMA * 3)
        for (let i = Math.max(0, hi - iw); i <= Math.min(n - 1, hi + iw); i++) {
          const d = i - hi
          acts[i] = Math.min(1, acts[i] + Math.exp(-(d * d) / (2 * WAVE_SIGMA * WAVE_SIGMA)) * WAVE_INJECT)
        }
      }

      // Decay all activations toward 0
      for (let i = 0; i < n; i++) {
        acts[i] *= WAVE_DECAY
        if (acts[i] < 0.003) acts[i] = 0
      }

      // Sync canvas pixel buffer to current CSS dimensions
      const imgH = wrapper.offsetHeight
      const cW   = wrapper.offsetWidth
      const cH   = Math.round(imgH * (1 + 2 * WAVE_MAX_EXT))
      if (canvas.width !== cW)  canvas.width  = cW
      if (canvas.height !== cH) canvas.height = cH

      // Draw activated stripes
      const ctx    = canvas.getContext('2d')
      const topPad = imgH * WAVE_MAX_EXT
      const sw     = cW / n
      ctx.clearRect(0, 0, cW, cH)

      for (let i = 0; i < n; i++) {
        const act = acts[i]
        if (act < 0.003) continue
        const extPx  = act * topPad
        const wMult  = 1 + (WAVE_CENTER_W - 1) * act
        const sw2    = sw * wMult
        ctx.fillStyle = rgbs[i]
        ctx.fillRect(i * sw - (sw2 - sw) / 2, topPad - extPx, sw2, imgH + extPx * 2)
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [active, colors])

  // ── Mouse handlers ─────────────────────────────────────────────────────────
  const handleHover = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x    = e.clientX - rect.left
    const idx  = Math.min(Math.floor((x / rect.width) * colors.length), colors.length - 1)
    hiRef.current = idx
    setHovered({ x, hex: colors[idx].hex, t: colors[idx].t })
  }

  const handleMouseLeave = () => {
    hiRef.current = -1
    setHovered(null)
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  const blurPx    = t * 10
  const innerStop = Math.max(15, Math.round((1 - t) * 65))
  const outerStop = Math.min(innerStop + 32, 100)
  const maskGrad  = `radial-gradient(ellipse at center, black ${innerStop}%, transparent ${outerStop}%)`
  const src       = `/barcodes/${barcode.filename}`

  return (
    <section
      ref={sectionRef}
      className={`barcode-section snap-section${active ? ' active' : ''}`}
    >
      <div className="barcode-wrapper" ref={wrapperRef}>
        <img className="barcode-img-blurred" src={src} alt="" aria-hidden="true"
          loading="lazy" style={{ filter: `blur(${blurPx}px)` }} />
        <img className="barcode-img-sharp" src={src} alt={barcode.title}
          loading="lazy" style={{ maskImage: maskGrad, WebkitMaskImage: maskGrad }} />

        {active && colors.length > 0 && (
          <div className="barcode-hover-overlay" onMouseMove={handleHover} onMouseLeave={handleMouseLeave}>
            <canvas ref={canvasRef} className="barcode-wave-canvas" />
            {hovered && (
              <div className="barcode-tooltip"
                style={{ left: Math.max(60, Math.min(hovered.x, (wrapperRef.current?.offsetWidth ?? 9999) - 60)) }}>
                <span className="barcode-tooltip-swatch" style={{ background: hovered.hex }} />
                <span className="barcode-tooltip-hex">{hovered.hex}</span>
                <span className="barcode-tooltip-time">{formatTime(hovered.t)}</span>
              </div>
            )}
          </div>
        )}
        <div className="barcode-overlay" />
        <div className="barcode-title">
          {barcode.title}
          <span className="barcode-mode">{barcode.mode}</span>
        </div>
      </div>
    </section>
  )
}

function FilterBar({ directors, selected, onSelect, visible }) {
  return (
    <div className={`filter-bar${visible ? ' filter-bar--visible' : ''}`}>
      <button
        className={`filter-btn${selected === null ? ' active' : ''}`}
        onClick={() => onSelect(null)}
      >
        All
      </button>
      {directors.map(d => (
        <button
          key={d}
          className={`filter-btn${selected === d ? ' active' : ''}`}
          onClick={() => onSelect(d)}
        >
          {d}
        </button>
      ))}
    </div>
  )
}

const ALL_DIRECTORS = [...new Set(barcodes.map(b => b.director))]

export default function App() {
  const containerRef  = useRef(null)
  const [selectedDirector, setSelectedDirector] = useState(null)
  const [showFilter, setShowFilter]             = useState(false)

  // Show filter bar once the intro has scrolled out of view
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const update = () => setShowFilter(container.scrollTop > window.innerHeight * 0.9)
    container.addEventListener('scroll', update, { passive: true })
    return () => container.removeEventListener('scroll', update)
  }, [])

  const filtered = selectedDirector
    ? barcodes.filter(b => b.director === selectedDirector)
    : barcodes

  const handleFilter = (director) => {
    setSelectedDirector(director)
    const top = director === null ? 0 : window.innerHeight
    containerRef.current?.scrollTo({ top, behavior: 'smooth' })
  }

  return (
    <>
      <FilterBar directors={ALL_DIRECTORS} selected={selectedDirector} onSelect={handleFilter} visible={showFilter} />
      <div className="scroll-container" ref={containerRef}>
        <Intro />
        {filtered.map((b) => (
          <BarcodeSection key={b.filename} barcode={b} containerRef={containerRef} />
        ))}
      </div>
    </>
  )
}
