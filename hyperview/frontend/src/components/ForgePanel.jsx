import React, { useRef, useEffect, useState, useMemo } from 'react'

function SparkParticle({ id, delay }) {
  const style = {
    position: 'absolute',
    width: '2px',
    height: '2px',
    borderRadius: '50%',
    background: '#ff8844',
    boxShadow: '0 0 4px #ff6622',
    animation: `spark-burst 0.8s ${delay}s ease-out forwards`,
    pointerEvents: 'none',
  }
  return <div style={style} />
}

function LightningBolt({ active }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!active || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    let frame

    const drawBolt = () => {
      ctx.clearRect(0, 0, w, h)
      const segments = 8 + Math.floor(Math.random() * 6)
      const startX = w * 0.3 + Math.random() * w * 0.4
      const startY = 0
      const endX = w * 0.3 + Math.random() * w * 0.4
      const endY = h

      ctx.beginPath()
      ctx.moveTo(startX, startY)

      let x = startX
      let y = startY
      for (let i = 1; i < segments; i++) {
        const t = i / segments
        const tx = startX + (endX - startX) * t + (Math.random() - 0.5) * w * 0.3
        const ty = startY + (endY - startY) * t
        ctx.lineTo(tx, ty)
        x = tx
        y = ty
      }
      ctx.lineTo(endX, endY)

      ctx.strokeStyle = 'rgba(0, 200, 255, 0.6)'
      ctx.lineWidth = 1.5
      ctx.shadowColor = 'rgba(0, 200, 255, 0.8)'
      ctx.shadowBlur = 8
      ctx.stroke()

      ctx.shadowBlur = 0
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
      ctx.lineWidth = 0.5
      ctx.stroke()

      frame = setTimeout(drawBolt, 100 + Math.random() * 200)
    }

    drawBolt()
    return () => clearTimeout(frame)
  }, [active])

  if (!active) return null

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        opacity: 0.5,
      }}
    />
  )
}

function ShockwaveRing({ active }) {
  if (!active) return null
  return (
    <div className="forge-shockwave">
      <div className="forge-shockwave-ring" />
    </div>
  )
}

function AnvilCrystal({ active, residual }) {
  const [sparks, setSparks] = useState([])
  const sparkId = useRef(0)

  useEffect(() => {
    if (!active) {
      setSparks([])
      return
    }
    const interval = setInterval(() => {
      const id = sparkId.current++
      setSparks(prev => [...prev.slice(-15), { id, delay: 0 }])
    }, 150 + Math.random() * 200)
    return () => clearInterval(interval)
  }, [active])

  return (
    <div className={`forge-anvil-glow ${active ? 'active' : ''}`}>
      <div className="forge-anvil-inner">
        <svg viewBox="0 0 64 64" className="forge-anvil-svg">
          <polygon
            points="16,48 48,48 56,16 8,16"
            fill={active ? 'rgba(0, 200, 255, 0.15)' : 'rgba(0, 150, 200, 0.08)'}
            stroke={active ? '#00ddff' : 'rgba(0, 180, 220, 0.3)'}
            strokeWidth="1"
          />
          <rect
            x="12" y="48" width="40" height="4" rx="2"
            fill={active ? 'rgba(0, 200, 255, 0.2)' : 'rgba(0, 150, 200, 0.08)'}
            stroke={active ? '#00ddff' : 'rgba(0, 180, 220, 0.3)'}
            strokeWidth="1"
          />
          {active && (
            <>
              <line x1="20" y1="20" x2="44" y2="20"
                stroke="#ff8844" strokeWidth="2" strokeLinecap="round"
                style={{ animation: 'forge-glow-pulse 0.5s infinite alternate' }} />
              <line x1="24" y1="28" x2="40" y2="28"
                stroke="#ffaa44" strokeWidth="1.5" strokeLinecap="round"
                style={{ animation: 'forge-glow-pulse 0.7s 0.2s infinite alternate' }} />
            </>
          )}
        </svg>
      </div>
      {sparks.map(s => (
        <SparkParticle key={s.id} id={s.id} delay={s.delay} />
      ))}
      {active && (
        <div className="forge-heat-ripple" />
      )}
    </div>
  )
}

export default function ForgePanel({ active, residual, lastBind, output }) {
  const [showShockwave, setShowShockwave] = useState(false)
  const prevActive = useRef(false)

  useEffect(() => {
    if (active && !prevActive.current) {
      setShowShockwave(true)
      setTimeout(() => setShowShockwave(false), 1000)
    }
    prevActive.current = active
  }, [active])

  const recentBinds = useMemo(() => {
    if (!lastBind) return []
    return [
      { a: lastBind.concept_a?.substring(0, 18) || '...', b: lastBind.concept_b?.substring(0, 18) || '...' }
    ]
  }, [lastBind])

  return (
    <div className="forge-content">
      <div className="forge-canvas-area">
        <AnvilCrystal active={active} residual={residual} />
        <LightningBolt active={active} />
        <ShockwaveRing active={showShockwave} />
      </div>

      <div className="forge-metrics">
        <div className="forge-metric">
          <span className="forge-metric-label">Energia</span>
          <span className="forge-metric-value" style={{ color: residual < 0.5 ? '#ff6633' : '#00ddff' }}>
            {residual.toFixed(3)}
          </span>
        </div>
        <div className="forge-metric">
          <span className="forge-metric-label">Estado</span>
          <span className="forge-metric-value">
            {active ? 'FORJANDO' : 'OCIOSA'}
          </span>
        </div>
      </div>

      {recentBinds.length > 0 && (
        <div className="forge-binds">
          {recentBinds.map((b, i) => (
            <div key={i} className="forge-bind-item">
              <span className="forge-bind-tag">{b.a}</span>
              <span className="forge-bind-arrow">→</span>
              <span className="forge-bind-tag">{b.b}</span>
            </div>
          ))}
        </div>
      )}

      {output && (
        <div className="forge-output-area" title={output}>
          <div className="forge-output-label">Última saída</div>
          <div className="forge-output-text">
            {output.substring(0, 80)}{output.length > 80 ? '...' : ''}
          </div>
        </div>
      )}
    </div>
  )
}
