/**
 * ChatInput — Barra de prompt no topo da tela
 */
import React, { useState } from 'react'

export default function ChatInput({ onSend, connected, compositional, onToggleMode }) {
  const [text, setText] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim() && connected) {
      onSend(text.trim())
      setText('')
    }
  }

  return (
    <form className="chat-container" onSubmit={handleSubmit}>
      <input
        className="chat-input"
        type="text"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder={compositional ? "Query composicional (CognitiveAnswerer)..." : "Digite uma intencao para a NovaMind..."}
        disabled={!connected}
      />
      <button
        type="button"
        className="chat-mode-toggle"
        onClick={onToggleMode}
        title={compositional ? "Modo composicional" : "Modo generativo"}
        style={{
          background: compositional ? 'rgba(0, 255, 200, 0.15)' : 'rgba(255, 102, 51, 0.15)',
          border: compositional ? '1px solid rgba(0, 255, 200, 0.4)' : '1px solid rgba(255, 102, 51, 0.4)',
          color: compositional ? '#00ffc8' : '#ff6633',
        }}
      >
        {compositional ? 'CMP' : 'GEN'}
      </button>
      <button className="chat-send" type="submit" disabled={!connected || !text.trim()}>
        {connected ? (compositional ? 'Compor' : 'Enviar') : '...'}
      </button>
    </form>
  )
}
