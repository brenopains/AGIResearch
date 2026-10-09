import React, { useState, useEffect, useRef } from 'react';
import HUD from './components/HUD';
import CognitiveCanvas from './components/CognitiveCanvas';
import './index.css';

function App() {
  const [state, setState] = useState({});
  const [chatHistory, setChatHistory] = useState([]);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket('ws://localhost:8000/ws');
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "chat_response") {
            setChatHistory(prev => [...prev, { sender: 'agi', text: data.text, vocal_audio_url: data.vocal_audio_url }]);
          } else {
            setState(data);
          }
        } catch (e) {
          console.error("Error parsing websocket data", e);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket closed. Reconnecting in 5s...");
        // DO NOT reload the page. Just reconnect silently.
        reconnectTimer.current = setTimeout(connect, 5000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const sendMessage = (msg) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  };

  return (
    <>
      <HUD state={state} sendMessage={sendMessage} chatHistory={chatHistory} setChatHistory={setChatHistory} />
      <div className="canvas-container">
        <CognitiveCanvas state={state} />
      </div>
    </>
  );
}

export default App;
