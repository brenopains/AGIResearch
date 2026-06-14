import React, { useState, useEffect } from 'react';
import HUD from './components/HUD';
import CognitiveCanvas from './components/CognitiveCanvas';
import './index.css';

function App() {
  const [state, setState] = useState({
    sensory_recent: [],
    episodic_uncompressed: [],
    world_model: { entities: {}, causal_links: [], beliefs: {} },
    representations: {},
    curiosity: { active_questions: [], attention: {} }
  });

  useEffect(() => {
    // Attempt connection to the Cognitive OS backend
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setState(data);
      } catch (e) {
        console.error("Error parsing websocket data", e);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket connection closed.");
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <>
      <HUD state={state} />
      <div className="canvas-container">
        <CognitiveCanvas state={state} />
      </div>
    </>
  );
}

export default App;
