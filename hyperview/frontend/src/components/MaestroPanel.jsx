import React, { useRef, useEffect, useState, useCallback } from 'react'

const TUBE_CONFIG = {
  fe:         { label: 'FE',   color: '#ff6633', barColor: '#ff4422', glowColor: 'rgba(255,68,34,0.3)', maxVal: 1 },
  dopamine:   { label: 'DOP',  color: '#00ddff', barColor: '#00bbdd', glowColor: 'rgba(0,189,221,0.3)', maxVal: 1 },
  noradrenaline: { label: 'NOR', color: '#ff3366', barColor: '#dd2255', glowColor: 'rgba(221,34,85,0.3)', maxVal: 1 },
  acetylcholine: { label: 'ACH', color: '#ffaa00', barColor: '#dd8800', glowColor: 'rgba(221,136,0,0.3)', maxVal: 1 },
}

function Bubble({ id, delay, left }) {
  const style = {
    position: 'absolute',
    bottom: '-8px',
    left: `${left}%`,
    width: `${4 + Math.random() * 6}px`,
    height: `${4 + Math.random() * 6}px`,
    borderRadius: '50%',
    background: 'rgba(200, 230, 245, 0.3)',
    border: '1px solid rgba(200, 230, 245, 0.15)',
    animation: `bubble-rise ${3 + Math.random() * 4}s ${delay}s infinite ease-in`,
    pointerEvents: 'none',
  }
  return <div style={style} />
}

function LiquidTube({ id, value, config }) {
  const pct = Math.min(100, Math.max(0, (value / config.maxVal) * 100))
  const [bubbles] = useState(() => {
    return Array.from({ length: 6 }, (_, i) => ({
      id: `${id}-${i}`,
      delay: i * 0.7 + Math.random() * 0.5,
      left: 15 + Math.random() * 70,
    }))
  })
  const waveRef = useRef(null)

  useEffect(() => {
    if (!waveRef.current) return
    waveRef.current.style.height = `${pct}%`
  }, [pct])

  return (
    <div className="maestro-tube">
      <span className="maestro-label">{config.label}</span>
      <div className="maestro-bar-bg">
        <div className="maestro-bar-liquid-container" ref={waveRef}>
          <div
            className="maestro-bar-fill maestro-bar-liquid"
            style={{
              background: `linear-gradient(180deg, ${config.barColor}, ${config.color})`,
              boxShadow: value > 0.3 ? `0 0 12px ${config.glowColor}, inset 0 0 20px ${config.glowColor}` : 'none',
            }}
          >
            <div className="maestro-bar-wave" />
          </div>
          {bubbles.map(b => (
            <Bubble key={b.id} id={b.id} delay={b.delay} left={b.left} />
          ))}
          {value > 0.6 && (
            <div
              className="maestro-tube-glow"
              style={{
                background: `radial-gradient(ellipse at 50% 50%, ${config.glowColor}, transparent 70%)`,
              }}
            />
          )}
        </div>
      </div>
      <span className="maestro-value">{value.toFixed(2)}</span>
    </div>
  )
}

function MiniFEChart({ history }) {
  const canvasRef = useRef(null)
  const animRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !history || history.length < 2) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height

    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      const gradient = ctx.createLinearGradient(0, 0, w, 0)
      gradient.addColorStop(0, 'rgba(255, 102, 51, 0.1)')
      gradient.addColorStop(0.5, 'rgba(255, 102, 51, 0.3)')
      gradient.addColorStop(1, 'rgba(255, 102, 51, 0.1)')

      ctx.beginPath()
      ctx.strokeStyle = '#ff6633'
      ctx.lineWidth = 1.5
      ctx.shadowColor = 'rgba(255, 102, 51, 0.4)'
      ctx.shadowBlur = 4

      const step = w / Math.max(history.length - 1, 1)
      history.forEach((val, i) => {
        const x = i * step
        const y = h - val * h
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.stroke()
      ctx.shadowBlur = 0

      ctx.lineTo(w, h)
      ctx.lineTo(0, h)
      ctx.closePath()
      ctx.fillStyle = 'rgba(255, 102, 51, 0.06)'
      ctx.fill()

      ctx.fillStyle = 'rgba(255, 102, 51, 0.8)'
      ctx.beginPath()
      ctx.arc(w - 2, h - history[history.length - 1] * h, 2, 0, Math.PI * 2)
      ctx.fill()
    }

    draw()
  }, [history])

  return <canvas ref={canvasRef} className="fe-chart" width={200} height={50} />
}

export default function MaestroPanel({ fe, dopamine, noradrenaline, acetylcholine }) {
  const feHistory = useRef([])
  const [glowPulse, setGlowPulse] = useState(0)

  useEffect(() => {
    if (fe !== undefined) {
      feHistory.current.push(fe)
      if (feHistory.current.length > 80) feHistory.current.shift()
    }
  }, [fe])

  useEffect(() => {
    let frame
    const animate = () => {
      setGlowPulse(prev => (prev + 0.02) % (Math.PI * 2))
      frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [])

  const tubes = [
    { id: 'fe', value: fe, config: TUBE_CONFIG.fe },
    { id: 'dop', value: dopamine, config: TUBE_CONFIG.dopamine },
    { id: 'nor', value: noradrenaline, config: TUBE_CONFIG.noradrenaline },
    { id: 'ach', value: acetylcholine, config: TUBE_CONFIG.acetylcholine },
  ]

  return (
    <div className="maestro-content">
      {tubes.map(t => (
        <LiquidTube key={t.id} id={t.id} value={t.value} config={t.config} />
      ))}
      <MiniFEChart history={feHistory.current} />
      <div className="maestro-status-bar">
        <div className="maestro-status-dot" style={{ opacity: 0.5 + 0.5 * Math.sin(glowPulse) }} />
        <span className="maestro-status-text">Sistema Límbico Ativo</span>
      </div>
    </div>
  )
}
