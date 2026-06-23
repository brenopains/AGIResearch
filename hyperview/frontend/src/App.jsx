/**
 * HyperView — Painel Principal
 * Layout: 4 paineis + chat input
 * 
 *   ┌─────────────────────────────────────┐
 *   │           CHAT INPUT                │
 *   ├──────────────────────┬──────────────┤
 *   │                      │  MAESTRO     │
 *   │    OCEAN HDC (3D)    │  PANEL       │
 *   │                      │              │
 *   ├──────────────────────┴──────────────┤
 *   │  FORJA        │  PROJECAO MENTAL   │
 *   └───────────────┴────────────────────┘
 */

import React, { useState, useMemo } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import OceanHDC from './components/OceanHDC'
import MaestroPanel from './components/MaestroPanel'
import ForgePanel from './components/ForgePanel'
import MentalProjection from './components/MentalProjection'
import ChatInput from './components/ChatInput'

export default function App() {
  const { connected, events, status, sendQuery, sendCompositional } = useWebSocket()
  const [queryText, setQueryText] = useState('')
  const [selectedConcept, setSelectedConcept] = useState(null)
  const [compositionalMode, setCompositionalMode] = useState(true)

  const latestEvents = useMemo(() => {
    const map = {}
    for (const e of events) {
      map[e.type] = e
    }
    return map
  }, [events])

  const lastChunk = latestEvents['chunk']
  const lastFE = latestEvents['fe_update']
  const lastBind = latestEvents['bind']
  const lastForge = latestEvents['forge_activate']
  const pcaUpdate = latestEvents['pca_update']
  const pcaPoints = pcaUpdate?.points || []
  const pcaEdges = pcaUpdate?.edges || []
  const memoryActivations = latestEvents['memory_activate']?.concepts || []
  const workingHDV = latestEvents['working_hdv']?.items || []

  const handleQuery = (text) => {
    setQueryText(text)
    setSelectedConcept(null)
    if (compositionalMode) {
      sendCompositional(text)
    } else {
      sendQuery(text)
    }
  }

  const handleToggleMode = () => {
    setCompositionalMode(prev => !prev)
  }

  const handleConceptClick = (point) => {
    setSelectedConcept(point)
  }

  return (
    <div className="app">
      <div className={`connection-dot ${connected ? 'connected' : ''}`} />

      <ChatInput onSend={handleQuery} connected={connected} compositional={compositionalMode} onToggleMode={handleToggleMode} />

      <div className="panels">
        <div className="panel panel-ocean">
          <div className="panel-header">
            <span className="panel-title">OCEANO HDC</span>
            <span className="panel-subtitle">{status.memory_items} conceitos</span>
          </div>
          <OceanHDC
            points={pcaPoints}
            edges={pcaEdges}
            activeConcepts={memoryActivations.map(c => c.label)}
            selectedConcept={selectedConcept}
            onConceptClick={handleConceptClick}
          />
        </div>

        <div className="panel panel-maestro">
          <div className="panel-header">
            <span className="panel-title">MAESTRO LIMBICO</span>
          </div>
          <MaestroPanel
            fe={lastFE?.fe ?? 1.0}
            dopamine={lastFE?.dopamine ?? 0}
            noradrenaline={lastFE?.noradrenaline ?? 0}
            acetylcholine={lastFE?.acetylcholine ?? 0}
          />
        </div>

        <div className="panel panel-forge">
          <div className="panel-header">
            <span className="panel-title">FORJA</span>
          </div>
          <ForgePanel
            active={!!lastForge}
            residual={lastForge?.residual ?? lastFE?.fe ?? 1.0}
            lastBind={lastBind}
            output={status.output}
          />
        </div>

        <div className="panel panel-projection">
          <div className="panel-header">
            <span className="panel-title">PROJECAO MENTAL</span>
          </div>
          <MentalProjection
            intention={status.intention}
            lastChunk={lastChunk}
            chunks={events.filter(e => e.type === 'chunk').map(e => e.text)}
            workingHDV={workingHDV}
          />
        </div>
      </div>
    </div>
  )
}
