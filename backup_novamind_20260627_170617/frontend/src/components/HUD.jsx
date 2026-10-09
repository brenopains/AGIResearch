import React, { useRef, useEffect, useState } from 'react';
import { Activity, Brain, Eye, Ear, Globe, BookOpen, Target, Network, Layers, Zap, Volume2, Send, Mic, Paperclip, ShieldAlert } from 'lucide-react';

const AudioSyncPlayer = ({ src, timestamp }) => {
  const audioRef = useRef(null);
  const lastTimeRef = useRef(-1);

  useEffect(() => {
    if (audioRef.current && timestamp !== null && timestamp !== lastTimeRef.current) {
      lastTimeRef.current = timestamp;
      audioRef.current.currentTime = timestamp;
      
      // Play a short burst of audio (1.5 seconds) matching the AGI's current temporal focus
      audioRef.current.play().catch(err => console.log("Audio playback not started yet:", err));
      
      const timer = setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.pause();
        }
      }, 1600);
      
      return () => clearTimeout(timer);
    }
  }, [timestamp, src]);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: '#94a3b8', background: 'rgba(255,255,255,0.05)', padding: '6px 12px', borderRadius: '6px', width: '100%', marginTop: '5px' }}>
      <Ear size={14} color="#a78bfa" style={{ animation: 'pulse 1s infinite' }} />
      <span style={{ fontWeight: '500' }}>Sincronia de Áudio:</span>
      <audio 
        ref={audioRef} 
        src={src} 
        preload="auto" 
        style={{ display: 'none' }}
      />
      <span style={{ color: '#a78bfa', fontFamily: 'monospace' }}>
        t = {timestamp}s (Ouvindo...)
      </span>
    </div>
  );
};

const VocalPlayer = ({ src }) => {
  const audioRef = useRef(null);
  const lastSrcRef = useRef("");

  useEffect(() => {
    if (src && src !== lastSrcRef.current) {
      lastSrcRef.current = src;
      if (audioRef.current) {
        audioRef.current.load();
        audioRef.current.play().catch(err => console.log("Vocal playback:", err));
      }
    }
  }, [src]);

  if (!src) return null;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: '#fcd34d', background: 'rgba(252, 211, 77, 0.08)', padding: '6px 12px', borderRadius: '6px', width: '100%', marginTop: '5px' }}>
      <Volume2 size={14} color="#fcd34d" style={{ animation: 'pulse 1s infinite' }} />
      <span style={{ fontWeight: '500' }}>Vocalização AGI:</span>
      <audio ref={audioRef} preload="auto" style={{ display: 'none' }}>
        <source src={src} type="audio/wav" />
      </audio>
      <span style={{ color: '#fcd34d', fontFamily: 'monospace' }}>
        Reproduzindo...
      </span>
    </div>
  );
};

const HUD = ({ state, sendMessage, chatHistory = [], setChatHistory }) => {
  if (!state || Object.keys(state).length === 0) {
    return <div className="hud-container"><div className="panel">Conectando ao NovaMind...</div></div>;
  }

  const learning = state.autonomous_learning || {};
  const streams = state.streams || {};
  const media = state.media || {};
  const visCortex = state.visual_cortex || {};
  const audCortex = state.auditory_cortex || {};
  const pendingPerms = state.pending_permissions || [];

  const [inputText, setInputText] = useState("");
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);
  const chatHistoryEndRef = useRef(null);

  // Auto-scroll chat history
  useEffect(() => {
    if (chatHistoryEndRef.current) {
      chatHistoryEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  const handleSendText = () => {
    if (!inputText.trim()) return;
    sendMessage({ type: 'user_chat', text: inputText });
    setChatHistory(prev => [...prev, { sender: 'user', text: inputText }]);
    setInputText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSendText();
    }
  };

  // Audio Recording (Microphone)
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        uploadAudioFile(audioBlob, "recorded_mic.webm");
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      // Stop all tracks to release mic
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setRecording(false);
    }
  };

  const uploadAudioFile = async (blob, filename) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", blob, filename);

    try {
      const resp = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();
      
      setChatHistory(prev => [
        ...prev, 
        { sender: 'user', text: `[Áudio de Voz Enviado: ${filename}]` },
        { sender: 'agi', text: data.text, vocal_audio_url: data.vocal_audio_url }
      ]);
      
      if (data.vocal_audio_url) {
        const audioObj = new Audio(data.vocal_audio_url);
        audioObj.play().catch(e => console.log("Audio play err:", e));
      }
    } catch (err) {
      console.error("Audio upload failed:", err);
    } finally {
      setUploading(false);
    }
  };

  // File Upload (Drag & Drop or paperclip)
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadAttachment(file);
    }
  };

  const uploadAttachment = async (file) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();
      
      setChatHistory(prev => [
        ...prev, 
        { sender: 'user', text: `[Anexo Enviado: ${file.name}]` },
        { sender: 'agi', text: data.text, vocal_audio_url: data.vocal_audio_url }
      ]);

      if (data.vocal_audio_url) {
        const audioObj = new Audio(data.vocal_audio_url);
        audioObj.play().catch(e => console.log("Audio play err:", e));
      }
    } catch (err) {
      console.error("File upload failed:", err);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      uploadAttachment(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <>
      {/* AGENTIAL PERMISSIONS OVERLAY PANEL */}
      {pendingPerms.length > 0 && (
        <div className="perm-panel">
          {pendingPerms.map((perm) => (
            <div key={perm.id} className="perm-card">
              <div className="perm-header">
                <ShieldAlert size={18} />
                <span>AÇÃO AGÊNTICA REQUERIDA</span>
              </div>
              <div style={{ fontSize: '0.85rem', fontWeight: '500', color: '#f8fafc' }}>
                {perm.type === 'execute_code' 
                  ? "O Maestro quer executar o código abaixo no terminal:" 
                  : perm.details}
              </div>
              {perm.type === 'execute_code' && (
                <div className="perm-details">
                  {perm.details}
                </div>
              )}
              <div className="perm-actions">
                <button 
                  className="btn-perm btn-reject" 
                  onClick={() => sendMessage({ type: 'permission_response', id: perm.id, approved: false })}
                >
                  Bloquear
                </button>
                <button 
                  className="btn-perm btn-approve" 
                  onClick={() => sendMessage({ type: 'permission_response', id: perm.id, approved: true })}
                >
                  Aprovar Ação
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="hud-container">
        {/* Left Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

          {/* LEARNING STATUS — What the AGI is doing RIGHT NOW */}
          <div className="panel" style={{ borderColor: '#38bdf8' }}>
            <div className="panel-title"><Zap size={16} color="#38bdf8" /> APRENDIZADO AUTÔNOMO</div>
            <div className="event-item" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>Ciclo Ativo:</span>
              <button 
                onClick={() => sendMessage({ type: 'toggle_learning', enabled: !learning.enabled })}
                style={{
                  background: learning.enabled ? '#10b981' : '#ef4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '4px 10px',
                  fontSize: '0.75rem',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                {learning.enabled ? 'ATIVADO' : 'DESATIVADO'}
              </button>
            </div>
            <div className="event-item" style={{ color: '#fcd34d', fontWeight: 'bold' }}>
              Status: {learning.status || 'booting'}
            </div>
            <div className="event-item" style={{ color: '#60a5fa' }}>
              Tópico: {learning.current_topic || '—'}
            </div>
            <div className="event-item" style={{ color: '#94a3b8', fontSize: '0.85em' }}>
              Fonte: {learning.current_source || '—'}
            </div>
            <div className="event-item" style={{ color: '#94a3b8', fontSize: '0.85em' }}>
              Progresso: {learning.sentences_processed || 0}/{learning.total_sentences || 0} sentenças
            </div>
            <div className="event-item" style={{ color: '#34d399' }}>
              Tópicos concluídos: {learning.topics_completed || 0}
            </div>
          </div>

          {/* SENSORY STREAMS — What the AGI is seeing/hearing/reading */}
          <div className="panel">
            <div className="panel-title"><Eye size={16} /> FLUXO SENSORIAL</div>
            <div className="event-item">
              <Eye size={12} style={{ marginRight: '6px', color: '#38bdf8' }} />
              <span style={{ color: streams.vision !== '—' ? '#38bdf8' : '#475569' }}>
                {streams.vision || 'Standby'}
              </span>
            </div>
            <div className="event-item">
              <Ear size={12} style={{ marginRight: '6px', color: '#a78bfa' }} />
              <span style={{ color: streams.audio !== '—' ? '#a78bfa' : '#475569' }}>
                {streams.audio || 'Standby'}
              </span>
            </div>
            <div className="event-item">
              <BookOpen size={12} style={{ marginRight: '6px', color: '#34d399' }} />
              <span style={{ color: streams.reading !== '—' ? '#34d399' : '#475569' }}>
                {streams.reading || 'Standby'}
              </span>
            </div>
          </div>

          {/* SENSORY CORTEX — AGI's own vision and hearing */}
          <div className="panel">
            <div className="panel-title"><Eye size={16} /> CÓRTICES SENSORIAIS (RAW)</div>
            <div className="event-item" style={{ color: '#38bdf8', fontSize: '0.8em', lineHeight: '1.3' }}>
              <span style={{ fontWeight: '600', color: '#60a5fa' }}>Visão:</span> {visCortex.last_description || '—'}
            </div>
            <div className="event-item" style={{ color: '#a78bfa', fontSize: '0.8em', lineHeight: '1.3' }}>
              <span style={{ fontWeight: '600', color: '#c084fc' }}>Audição:</span> {audCortex.last_description || '—'}
            </div>
            <div className="event-item" style={{ color: '#475569', fontSize: '0.7em' }}>
              frames: {visCortex.total_frames || 0} | segmentos áudio: {audCortex.total_segments || 0}
            </div>
          </div>

          {/* INTERACTIVE CHAT & EMULATION CARD */}
          <div className="panel" style={{ height: '360px' }}>
            <div className="panel-title"><Brain size={16} /> INTERFACE COGNITIVA</div>
            
            {/* Live autonomous sentence processing (top banner inside panel) */}
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', color: '#94a3b8', borderLeft: '2px solid #ef4444' }}>
              <span style={{ fontWeight: 'bold', color: '#ef4444' }}>Live CVM:</span> {learning.current_sentence || 'Aguardando fluxo...'}
            </div>

            {/* Chat History */}
            <div 
              className="chat-history"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              {chatHistory.length === 0 ? (
                <div style={{ color: '#475569', fontSize: '0.8rem', textAlign: 'center', marginTop: '20px' }}>
                  Arraste arquivos ou digite/fale para conversar com o cérebro da NovaMind
                </div>
              ) : (
                chatHistory.map((msg, idx) => (
                  <div 
                    key={idx} 
                    className={`chat-bubble ${msg.sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-agi'}`}
                  >
                    {msg.text}
                  </div>
                ))
              )}
              <div ref={chatHistoryEndRef} />
            </div>

            {/* Chat Input row */}
            <div className="chat-input-row">
              <input 
                type="file" 
                ref={fileInputRef} 
                style={{ display: 'none' }}
                onChange={handleFileChange}
                accept=".pdf,.mp3,.wav,.webm,.mp4"
              />
              <button 
                className="icon-btn" 
                title="Anexar arquivo (PDF, MP3, MP4)"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                <Paperclip size={16} />
              </button>
              
              <input 
                type="text" 
                className="chat-input-text" 
                placeholder={uploading ? "Carregando..." : "Digite para a NovaMind..."}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={uploading}
              />
              
              <button 
                className={`icon-btn ${recording ? 'icon-btn-recording' : ''}`}
                title={recording ? "Parar Gravação" : "Falar por Áudio"}
                onClick={recording ? stopRecording : startRecording}
                disabled={uploading}
              >
                <Mic size={16} />
              </button>

              <button 
                className="icon-btn"
                onClick={handleSendText}
                disabled={uploading || !inputText.trim()}
              >
                <Send size={16} />
              </button>
            </div>
          </div>

        </div>

        {/* Middle Spacer with Multimodal Observation Chamber at the bottom */}
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center', pointerEvents: 'none', paddingBottom: '10px' }}>
          {media && media.type === "video" && (
            <div className="panel observation-chamber" style={{ pointerEvents: 'auto', width: '460px', border: '1px solid #3b82f6', boxShadow: '0 8px 32px rgba(59, 130, 246, 0.25)', gap: '10px', padding: '15px' }}>
              <div className="panel-title" style={{ borderColor: '#3b82f6', color: '#60a5fa' }}>
                <Eye size={16} color="#3b82f6" /> MONITOR MULTIMODAL (LIVE)
              </div>
              
              {/* The live frame image */}
              <div style={{ position: 'relative', width: '100%', height: '220px', backgroundColor: '#020617', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                {media.frame_url ? (
                  <img 
                    src={`${media.frame_url}?cb=${media.timestamp}`} 
                    alt="AGI Sensory Vision Frame" 
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    onError={(e) => { 
                      e.target.src = 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=500&auto=format&fit=crop&q=60'; 
                    }}
                  />
                ) : (
                  <div style={{ color: '#475569', fontSize: '0.85em' }}>Aguardando frame de vídeo...</div>
                )}
                
                {/* Overlay with timestamp */}
                {media.timestamp !== null && (
                  <div style={{ position: 'absolute', bottom: '10px', right: '10px', background: 'rgba(0,0,0,0.85)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.75rem', fontFamily: 'monospace', color: '#60a5fa' }}>
                    t = {media.timestamp}s
                  </div>
                )}

                {/* Overlay with focus ring representing AGI's focal attention */}
                <div style={{ position: 'absolute', width: '50px', height: '50px', border: '2px dashed #ef4444', borderRadius: '50%', animation: 'pulse 1.5s infinite', pointerEvents: 'none' }}></div>
              </div>

              {/* Subtitle / Caption text */}
              <div style={{ minHeight: '50px', background: 'rgba(0,0,0,0.4)', borderRadius: '6px', padding: '10px', fontSize: '0.85rem', color: '#f8fafc', borderLeft: '3px solid #38bdf8', lineHeight: '1.4' }}>
                {media.caption || 'Estudando canal do YouTube...'}
              </div>

              {/* Live Audio playback synchronization */}
              {media.audio_url && media.timestamp !== null && (
                <AudioSyncPlayer src={media.audio_url} timestamp={media.timestamp} />
              )}

              {/* Vocalização real da AGI */}
              {media.vocal_audio_url && (
                <VocalPlayer src={media.vocal_audio_url} />
              )}
            </div>
          )}

          {/* Vocal player também aparece independente do tipo de mídia */}
          {(!media || media.type !== "video") && media?.vocal_audio_url && (
            <VocalPlayer src={media.vocal_audio_url} />
          )}
        </div>

        {/* Right Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* WORLD MODEL */}
          <div className="panel">
            <div className="panel-title"><Network size={16} /> Modelo de Mundo</div>
            <div className="event-item" style={{color: '#94a3b8'}}>
              Entidades: {state.world_model?.nodes?.length || 0} | Causas: {state.world_model?.edges?.length || 0}
            </div>
            {state.world_model?.causal_chains?.slice(0, 3).map((chain, idx) => (
              <div key={idx} className="event-item" style={{ color: '#34d399', fontSize: '0.85em' }}>
                {Array.isArray(chain) ? chain.join(' → ') : String(chain)}
              </div>
            ))}
          </div>

          {/* METACOGNITION */}
          <div className="panel">
            <div className="panel-title"><Layers size={16} /> Metacognição</div>
            <div className="event-item" style={{ color: '#fcd34d' }}>
              Confiança: {((state.metacognition?.confidence?.overall || 0) * 100).toFixed(1)}%
            </div>
            <div className="event-item" style={{ color: '#94a3b8' }}>
              Confusão: {((state.metacognition?.emotional_state?.confusion || 0) * 100).toFixed(1)}%
            </div>
          </div>

          {/* EPISODIC MEMORY */}
          <div className="panel">
            <div className="panel-title"><Activity size={16} /> Memória Episódica ({state.memory?.stats?.total_episodes || 0})</div>
            {state.memory?.contents?.episodic?.slice(-4).map((ep, idx) => (
              <div key={idx} className="event-item" style={{ fontSize: '0.85em' }}>
                <span style={{ color: '#94a3b8' }}>{ep.content?.input?.substring(0, 60) || '...'}</span>
              </div>
            ))}
          </div>

          {/* GOALS & CURIOSITY */}
          <div className="panel">
            <div className="panel-title"><Target size={16} /> Curiosidade</div>
            {state.goals?.active_goals?.slice(0, 3).map((g, idx) => (
              <div key={idx} className="event-item" style={{ color: '#fb923c', fontSize: '0.85em' }}>
                ▶ {g.description}
              </div>
            ))}
            {(!state.goals?.active_goals || state.goals.active_goals.length === 0) && (
              <div className="event-item" style={{color: '#475569'}}>Seguindo currículo</div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default HUD;
