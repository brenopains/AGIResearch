import React from 'react';
import { Activity, Brain, Eye, HelpCircle } from 'lucide-react';

const HUD = ({ state }) => {
  const formatTime = (ts) => {
    const d = new Date(ts * 1000);
    return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`;
  };

  return (
    <div className="hud-container">
      {/* Left Panel: Sensory Flow & Episodic Memory */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="panel">
          <div className="panel-title"><Eye size={16} /> Sensory Flow</div>
          <div>
            {state.sensory_recent.map((evt, idx) => (
              <div key={idx} className="event-item">
                <span className="event-time">{formatTime(evt.timestamp)}</span>
                <span style={{ color: evt.type === 'text' ? '#60a5fa' : '#94a3b8' }}>
                  {evt.content}
                </span>
              </div>
            ))}
            {state.sensory_recent.length === 0 && <div className="event-item" style={{color: '#64748b'}}>Aguardando estímulo...</div>}
          </div>
        </div>

        <div className="panel">
          <div className="panel-title"><Activity size={16} /> Timeline (Episódica)</div>
          <div>
            {state.episodic_uncompressed.map((ep, idx) => (
              <div key={idx} className="event-item">
                <span className="event-time">{formatTime(ep.recorded_at)}</span>
                <span>{ep.event.content}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Middle Spacer for 3D Canvas */}
      <div style={{ pointerEvents: 'none' }}></div>

      {/* Right Panel: World Model & Curiosity */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="panel">
          <div className="panel-title"><HelpCircle size={16} /> Curiosidade</div>
          <div>
            {state.curiosity.active_questions && state.curiosity.active_questions.map((q, idx) => (
              <div key={idx} className="event-item" style={{ color: '#fcd34d' }}>
                ? {q}
              </div>
            ))}
            {(!state.curiosity.active_questions || state.curiosity.active_questions.length === 0) && (
              <div className="event-item" style={{color: '#64748b'}}>Nenhuma pergunta ativa</div>
            )}
          </div>
        </div>
        
        <div className="panel" style={{ flexGrow: 1 }}>
          <div className="panel-title"><Brain size={16} /> Crenças Ativas</div>
          <div>
            {Object.keys(state.world_model.beliefs || {}).map((belief, idx) => (
              <div key={idx} className="event-item" style={{ color: '#34d399' }}>
                {belief} (Confirmado)
              </div>
            ))}
            {Object.keys(state.world_model.beliefs || {}).length === 0 && (
              <div className="event-item" style={{color: '#64748b'}}>Nenhuma crença forte formada</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HUD;
