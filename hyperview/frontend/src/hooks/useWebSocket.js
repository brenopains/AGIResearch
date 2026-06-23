/**
 * useWebSocket — Hook para conexao com o backend HyperView.
 * Gerencia reconexao automatica e fila de eventos.
 */
import { useState, useEffect, useRef, useCallback } from 'react'

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState({
    intention: '',
    output: '',
    memory_items: 0,
    working_hdv_size: 0,
    pca_points: 0,
  })
  
  const ws = useRef(null)
  const reconnectTimer = useRef(null)
  const reconnectAttempts = useRef(0)
  const maxEvents = 200

  const connect = useCallback(() => {
    // Evita múltiplas conexões simultâneas
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/ws/stream`
    
    const socket = new WebSocket(url)
    
    socket.onopen = () => {
      setConnected(true)
      reconnectAttempts.current = 0
    }
    
    socket.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data)
        if (event.type === 'status') {
          setStatus(prev => ({ ...prev, ...event }))
        }
        setEvents(prev => {
          const next = [...prev, event]
          if (next.length > maxEvents) next.shift()
          return next
        })
      } catch (e) {
        // ignore parse errors
      }
    }
    
    socket.onclose = () => {
      setConnected(false)
      ws.current = null
      reconnectAttempts.current += 1
      // Backoff exponencial: 1s, 2s, 4s, 8s... max 30s
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current - 1), 30000)
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      reconnectTimer.current = setTimeout(connect, delay)
    }
    
    socket.onerror = () => {
      socket.close()
    }
    
    ws.current = socket
  }, [])

  useEffect(() => {
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (ws.current) ws.current.close()
    }
  }, [connect])

  const sendQuery = useCallback((text, maxChunks = 12) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: 'query',
        text,
        max_chunks: maxChunks
      }))
    }
  }, [])

  const sendCompositional = useCallback((text) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: 'compositional',
        text,
      }))
    }
  }, [])

  return { connected, events, status, sendQuery, sendCompositional }
}
