import React, { useRef, useEffect, useState, useMemo } from 'react'

function TokenStream({ text, step }) {
  const [visibleTokens, setVisibleTokens] = useState([])
  const [tokenIndex, setTokenIndex] = useState(0)

  const tokens = useMemo(() => {
    if (!text) return []
    const words = text.split(/(\s+)/)
    const result = []
    for (const w of words) {
      if (w.trim()) {
        if (w.match(/[\(\)\[\]\{\}:,;.=+*/%\-]/)) {
          for (const ch of w) {
            result.push({ text: ch, type: 'symbol' })
          }
        } else if (w.match(/^[A-Z]/)) {
          result.push({ text: w, type: 'class' })
        } else if (w.match(/^def|class|return|if|else|for|while|import|from$/)) {
          result.push({ text: w, type: 'keyword' })
        } else if (w.match(/^\d+$/)) {
          result.push({ text: w, type: 'number' })
        } else {
          result.push({ text: w, type: 'text' })
        }
      } else {
        result.push({ text: w, type: 'space' })
      }
    }
    return result
  }, [text])

  useEffect(() => {
    setTokenIndex(0)
    setVisibleTokens([])
  }, [text])

  useEffect(() => {
    if (tokenIndex >= tokens.length) return
    const timer = setTimeout(() => {
      setTokenIndex(prev => prev + 1)
      setVisibleTokens(tokens.slice(0, tokenIndex + 1))
    }, 30 + Math.random() * 20)
    return () => clearTimeout(timer)
  }, [tokenIndex, tokens])

  return (
    <div className="projection-token-stream">
      {visibleTokens.map((t, i) => (
        <span
          key={i}
          className={`projection-token projection-token-${t.type}`}
          style={{ animationDelay: `${i * 0.02}s` }}
        >
          {t.text}
        </span>
      ))}
      {tokenIndex < tokens.length && (
        <span className="projection-cursor">▊</span>
      )}
    </div>
  )
}

function ConceptFlow({ workingHDV }) {
  const items = useMemo(() => {
    if (!workingHDV || workingHDV.length === 0) return []
    return workingHDV.slice(-12).reverse()
  }, [workingHDV])

  if (items.length === 0) return null

  return (
    <div className="projection-concept-flow">
      <div className="projection-flow-label">Working Memory</div>
      <div className="projection-flow-tags">
        {items.map((item, i) => (
          <span
            key={i}
            className="projection-flow-tag"
            style={{
              animation: `tag-in 0.3s ${i * 0.05}s ease-out both`,
              borderColor: `hsla(${190 + (i % 5) * 30}, 80%, 60%, 0.4)`,
            }}
            title={item}
          >
            {item.length > 22 ? item.substring(0, 20) + '..' : item}
          </span>
        ))}
      </div>
    </div>
  )
}

function NetworkGraph({ chunks }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !chunks || chunks.length < 2) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height

    ctx.clearRect(0, 0, w, h)

    const nodes = chunks.slice(-8).map((chunk, i) => ({
      x: 20 + (w - 40) * (i / Math.max(chunks.slice(-8).length - 1, 1)),
      y: h / 2 + Math.sin(i * 1.2) * (h * 0.15),
      r: 3 + Math.sin(i) * 1.5,
      label: chunk.substring(0, 12),
    }))

    for (let i = 1; i < nodes.length; i++) {
      const gradient = ctx.createLinearGradient(nodes[i - 1].x, nodes[i - 1].y, nodes[i].x, nodes[i].y)
      gradient.addColorStop(0, 'rgba(0, 180, 220, 0.1)')
      gradient.addColorStop(0.5, 'rgba(0, 200, 255, 0.3)')
      gradient.addColorStop(1, 'rgba(0, 180, 220, 0.1)')
      ctx.beginPath()
      ctx.moveTo(nodes[i - 1].x, nodes[i - 1].y)
      ctx.lineTo(nodes[i].x, nodes[i].y)
      ctx.strokeStyle = gradient
      ctx.lineWidth = 1
      ctx.stroke()
    }

    nodes.forEach((node, i) => {
      ctx.beginPath()
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(0, ${180 + i * 10}, 255, ${0.3 + i * 0.08})`
      ctx.fill()
      ctx.strokeStyle = 'rgba(0, 200, 255, 0.5)'
      ctx.lineWidth = 0.5
      ctx.stroke()

      if (i === nodes.length - 1) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.r + 3, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(0, 220, 255, 0.6)'
        ctx.lineWidth = 1
        ctx.setLineDash([2, 2])
        ctx.stroke()
        ctx.setLineDash([])
      }
    })
  }, [chunks])

  if (!chunks || chunks.length < 1) return null

  return (
    <canvas
      ref={canvasRef}
      className="projection-network"
      width={260}
      height={70}
    />
  )
}

function ResonanceBar({ residual }) {
  const bars = 24

  return (
    <div className="projection-resonance">
      {Array.from({ length: bars }, (_, i) => {
        const center = (i / bars) * 2 - 1
        const height = Math.max(2, Math.abs(Math.sin(center * Math.PI * 0.5 + residual * Math.PI)) * 16 * residual)
        return (
          <div
            key={i}
            className="projection-resonance-bar"
            style={{
              height: `${height}px`,
              opacity: 0.2 + residual * 0.6,
              background: residual > 0.5 ? '#00ddff' : '#ff6633',
              boxShadow: residual > 0.5 ? '0 0 4px #00ddff' : '0 0 4px #ff6633',
            }}
          />
        )
      })}
    </div>
  )
}

export default function MentalProjection({ intention, lastChunk, chunks, workingHDV }) {
  const recentChunks = useMemo(() => (chunks || []).slice(-8), [chunks])
  const lastText = lastChunk?.text || ''
  const lastStep = lastChunk?.step || 0
  const residual = lastChunk?.residual_energy ?? 1.0

  return (
    <div className="projection-content">
      <div className="projection-intention" title={intention}>
        <span className="projection-intention-label">Intenção</span>
        <span className="projection-intention-text">
          {intention || 'Aguardando comando...'}
        </span>
      </div>

      <ResonanceBar residual={residual} />

      <div className="projection-stream-container">
        <div className="projection-stream-label">
          <span>Fluxo de tokens</span>
          {lastStep > 0 && <span className="projection-step">passo {lastStep}</span>}
        </div>
        <div className="projection-stream-box">
          {lastText ? (
            <TokenStream key={`${lastStep}-${lastText.substring(0, 20)}`} text={lastText} step={lastStep} />
          ) : (
            <div className="projection-empty">[...]</div>
          )}
        </div>
      </div>

      <NetworkGraph chunks={recentChunks} />
      <ConceptFlow workingHDV={workingHDV} />
    </div>
  )
}
