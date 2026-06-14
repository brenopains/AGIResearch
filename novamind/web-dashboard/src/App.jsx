import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Activity, BrainCircuit, Terminal, Cpu, MessageSquare, Moon, Sun, Eye, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import ForceGraph2D from 'react-force-graph-2d';

const API = 'http://localhost:5000';

const NovaDashboard = () => {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'NovaCrush Engine Online. Awaiting sensory input.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDreaming, setIsDreaming] = useState(false);
  const [systemState, setSystemState] = useState(null);
  const [tab, setTab] = useState('chat'); // 'chat' | 'vision'
  
  const messagesEndRef = useRef(null);
  const graphRef = useRef(null);
  const brocaRef = useRef(null);
  // Keep a stable reference for graph data so ForceGraph doesn't flicker
  const stableGraphRef = useRef({ nodes: [], links: [] });
  
  // Auto-scroll chat & broca
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  useEffect(() => {
    if (brocaRef.current) brocaRef.current.scrollTop = brocaRef.current.scrollHeight;
  }, [systemState?.thought_history]);

  // Fetch state periodically (slower to not waste CPU)
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch(`${API}/api/state`);
        if (res.ok) setSystemState(await res.json());
      } catch (e) { /* server down */ }
    };
    fetchState();
    const interval = setInterval(fetchState, 5000);
    return () => clearInterval(interval);
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setIsLoading(true);
    try {
      const res = await fetch(`${API}/api/think`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: userMsg })
      });
      const data = await res.json();
      if (data.error) {
        setMessages(prev => [...prev, { role: 'bot', error: true, text: `Erro: ${data.error}` }]);
      } else {
        const conf = data.response?.confidence ? Math.round(data.response.confidence * 100) : 0;
        const text = data.response?.text || 'Sem resposta.';
        // Parse the structured response
        const lines = text.split('\n');
        const native = lines.find(l => l.includes('NovaCrush Native Response:'))?.replace('**NovaCrush Native Response:**', '').trim() || text;
        const topo = lines.find(l => l.includes('Topological Prediction:'))?.replace('*(Topological Prediction:', '').replace(')*', '').trim() || '';
        setMessages(prev => [...prev, { 
          role: 'bot', text: native,
          meta: `Ciclo #${data.cycle} | Confianca: ${conf}%`,
          topo
        }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: 'bot', error: true, text: 'Falha na conexao com o motor AGI.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDream = async () => {
    try {
      const res = await fetch(`${API}/api/dream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable: !isDreaming })
      });
      const data = await res.json();
      setIsDreaming(data.status === 'dreaming');
      setMessages(prev => [...prev, { role: 'bot', text: data.message }]);
    } catch (e) { console.error(e); }
  };

  // Build stable graph data (only update when node count changes to avoid flickering)
  const graphData = useMemo(() => {
    if (!systemState?.world_model) return stableGraphRef.current;
    const wm = systemState.world_model;
    const nodes = (wm.nodes || []).slice(0, 200).map(n => ({ id: n.id, label: n.label }));
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = (wm.edges || [])
      .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map(e => ({ source: e.source, target: e.target }));
    // Only update ref if count changed (prevents ForceGraph re-simulation flicker)
    if (nodes.length !== stableGraphRef.current.nodes.length) {
      stableGraphRef.current = { nodes, links };
    }
    return stableGraphRef.current;
  }, [systemState?.world_model?.nodes?.length]);

  // Concept list for "Vision" tab
  const conceptList = useMemo(() => {
    if (!systemState?.perception?.all_concepts) return [];
    return systemState.perception.all_concepts.slice(0, 100);
  }, [systemState?.perception?.all_concepts]);

  // Novacrush stats
  const crush = systemState?.novacrush;
  const cycles = systemState?.cycle_count || 0;
  const totalConcepts = systemState?.perception?.total_concepts || 0;
  const uptime = systemState?.uptime_seconds || 0;
  const uptimeStr = uptime < 60 ? `${Math.floor(uptime)}s` : `${Math.floor(uptime/60)}m ${Math.floor(uptime%60)}s`;

  // Build training curve from thought_history (real data)
  const trainingCurve = useMemo(() => {
    if (!systemState?.thought_history) return [];
    return systemState.thought_history.map((t, i) => ({
      cycle: t.cycle,
      confidence: Math.round((t.confidence || 0) * 100),
      time: Math.round(t.cycle_time_ms || 0),
    }));
  }, [systemState?.thought_history]);

  const nodeCanvasObject = useCallback((node, ctx) => {
    ctx.beginPath();
    ctx.arc(node.x, node.y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = '#8b5cf6';
    ctx.fill();
    ctx.font = '3px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText(node.label || '', node.x + 4, node.y + 1);
  }, []);

  return (
    <div className="flex h-screen p-3 gap-3 bg-background text-slate-200 font-sans">
      
      {/* ═══ LEFT: World Model Graph + Stats ═══ */}
      <div className="flex flex-col w-[28%] gap-3">
        
        {/* World Model Graph */}
        <div className="flex-1 bg-surface border border-border rounded-xl p-3 flex flex-col relative overflow-hidden">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-neonPurple font-semibold text-sm">
              <BrainCircuit size={16} />
              <span>World Model (Grafo Causal)</span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">{graphData.nodes.length} nos | {graphData.links.length} arestas</span>
          </div>
          
          <div className="flex-1 relative rounded-lg overflow-hidden bg-black/30">
            {graphData.nodes.length > 0 ? (
              <ForceGraph2D
                ref={graphRef}
                graphData={graphData}
                nodeCanvasObject={nodeCanvasObject}
                linkColor={() => 'rgba(139, 92, 246, 0.15)'}
                linkWidth={0.5}
                backgroundColor="transparent"
                cooldownTicks={50}
                warmupTicks={20}
                enableZoomInteraction={true}
                enablePanInteraction={true}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
                Envie uma mensagem para construir o World Model...
              </div>
            )}
          </div>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-surface border border-border rounded-lg p-3 text-center">
            <div className="text-xl font-bold text-white">{totalConcepts}</div>
            <div className="text-[10px] text-slate-400 uppercase tracking-wider">Conceitos</div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3 text-center">
            <div className="text-xl font-bold text-white">{cycles}</div>
            <div className="text-[10px] text-slate-400 uppercase tracking-wider">Ciclos</div>
          </div>
        </div>

        {/* Training Curve */}
        <div className="h-40 bg-surface border border-border rounded-xl p-3 flex flex-col">
          <div className="flex items-center gap-2 mb-2 text-accent font-semibold text-sm">
            <Activity size={14} />
            <span>Evolucao (Confianca %)</span>
          </div>
          <div className="flex-1 -ml-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingCurve}>
                <XAxis dataKey="cycle" stroke="#475569" fontSize={9} />
                <YAxis stroke="#475569" fontSize={9} domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', fontSize: 11 }} />
                <Line type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ═══ CENTER: Chat + Vision Tabs ═══ */}
      <div className="flex-1 bg-surface border border-border rounded-xl flex flex-col backdrop-blur-md shadow-2xl">
        
        {/* Header with tabs */}
        <div className="p-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isDreaming ? 'bg-indigo-400 animate-pulse' : 'bg-green-500 animate-pulse'}`}></div>
            <div className="flex bg-black/30 rounded-lg p-0.5">
              <button onClick={() => setTab('chat')} className={`px-3 py-1 rounded-md text-xs transition-colors ${tab === 'chat' ? 'bg-accent/20 text-accent' : 'text-slate-400 hover:text-white'}`}>
                <MessageSquare size={12} className="inline mr-1" /> Chat
              </button>
              <button onClick={() => setTab('vision')} className={`px-3 py-1 rounded-md text-xs transition-colors ${tab === 'vision' ? 'bg-neonPurple/20 text-neonPurple' : 'text-slate-400 hover:text-white'}`}>
                <Eye size={12} className="inline mr-1" /> Visao Interna
              </button>
            </div>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
            <button onClick={toggleDream}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md border transition-colors ${isDreaming ? 'bg-indigo-600/20 border-indigo-500/30 text-indigo-300' : 'border-white/10 hover:bg-white/5'}`}>
              {isDreaming ? <Moon size={12} className="animate-pulse" /> : <Sun size={12} />}
              {isDreaming ? 'SONHANDO' : 'ACORDADO'}
            </button>
            <div className="flex items-center gap-1"><Cpu size={12}/> RTX 3050</div>
            <div><Zap size={12} className="inline" /> {uptimeStr}</div>
          </div>
        </div>

        {/* CHAT TAB */}
        {tab === 'chat' && (
          <>
            <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4">
              <AnimatePresence>
                {messages.map((msg, idx) => (
                  <motion.div key={idx} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                    className={`max-w-[85%] flex flex-col ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                    <div className={`px-4 py-2.5 rounded-2xl text-sm ${
                      msg.role === 'user' ? 'bg-accent/20 border border-accent/30 text-white rounded-br-none' 
                      : msg.error ? 'bg-red-500/10 border border-red-500/30 text-red-200 rounded-bl-none'
                      : 'bg-white/5 border border-white/10 text-slate-100 rounded-bl-none'
                    }`}>
                      {msg.text}
                    </div>
                    {msg.topo && (
                      <div className="text-[9px] text-neonPurple/60 mt-1 px-2 font-mono italic">
                        Topologia: {msg.topo.substring(0, 120)}...
                      </div>
                    )}
                    {msg.meta && (
                      <div className="text-[10px] text-slate-500 mt-0.5 font-mono px-2">{msg.meta}</div>
                    )}
                  </motion.div>
                ))}
                {isLoading && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="self-start text-xs text-accent animate-pulse font-mono flex items-center gap-2">
                    <Activity size={12} /> Processando trajetoria geometrica...
                  </motion.div>
                )}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>
            <div className="p-3 border-t border-border bg-black/20">
              <div className="flex gap-2">
                <input type="text" value={input} onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && sendMessage()}
                  placeholder="Fale com o NovaMind..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-accent transition-colors text-white placeholder-slate-500"
                  disabled={isLoading} />
                <button onClick={sendMessage} disabled={isLoading || !input.trim()}
                  className="bg-accent hover:bg-accent/80 text-white px-5 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                  <MessageSquare size={16} />
                </button>
              </div>
            </div>
          </>
        )}

        {/* VISION TAB: What the AGI "sees" internally */}
        {tab === 'vision' && (
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            <div className="text-xs text-slate-400 mb-2">
              Este painel mostra exatamente o que a AGI percebeu, como ela organizou os conceitos,
              e as cadeias causais que ela construiu no World Model. Nada e hardcoded.
            </div>

            {/* Last processed input */}
            {systemState?.thought_history?.length > 0 && (() => {
              const last = systemState.thought_history[systemState.thought_history.length - 1];
              return (
                <div className="bg-black/30 border border-neonPurple/20 rounded-lg p-4">
                  <div className="text-neonPurple text-xs font-semibold mb-2">Ultimo Pensamento (Ciclo {last.cycle})</div>
                  <div className="text-sm text-white mb-1">Entrada: <span className="text-slate-300">{last.input}</span></div>
                  <div className="text-sm text-emerald-300">Saida: {last.response_preview?.substring(0, 200)}</div>
                  <div className="text-[10px] text-slate-500 mt-2">
                    Confianca: {Math.round((last.confidence || 0) * 100)}% | Tempo: {Math.round(last.cycle_time_ms || 0)}ms
                  </div>
                </div>
              );
            })()}

            {/* Concept Cloud */}
            <div className="bg-black/30 border border-white/10 rounded-lg p-4">
              <div className="text-accent text-xs font-semibold mb-3">Conceitos Percebidos ({totalConcepts} total)</div>
              <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                {conceptList.map((c, i) => (
                  <span key={i} className="px-2 py-0.5 bg-neonPurple/10 border border-neonPurple/20 rounded-full text-[10px] text-neonPurple/80">
                    {c.label}
                  </span>
                ))}
              </div>
            </div>

            {/* Causal chains from World Model */}
            <div className="bg-black/30 border border-white/10 rounded-lg p-4">
              <div className="text-emerald-400 text-xs font-semibold mb-3">Cadeias Causais (World Model)</div>
              {systemState?.world_model?.edges?.slice(0, 30).map((edge, i) => (
                <div key={i} className="text-[10px] text-slate-300 font-mono py-0.5 border-b border-white/5">
                  <span className="text-accent">{edge.source_label || edge.source}</span>
                  <span className="text-slate-500 mx-1">--[{edge.relation || 'causa'}]--&gt;</span>
                  <span className="text-emerald-300">{edge.target_label || edge.target}</span>
                  {edge.weight && <span className="text-slate-600 ml-2">({(edge.weight * 100).toFixed(0)}%)</span>}
                </div>
              )) || <div className="text-slate-500 text-[10px]">Nenhuma cadeia causal ainda. Envie mensagens ou ative o modo Sonho.</div>}
            </div>

            {/* NovaCrush Compression Stats */}
            {crush && (
              <div className="bg-black/30 border border-white/10 rounded-lg p-4">
                <div className="text-amber-400 text-xs font-semibold mb-2">NovaCrush Compression</div>
                <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                  <div>Compressao: <span className="text-white">{crush.compression?.overall_ratio || '?'}x</span></div>
                  <div>Ternario: <span className="text-white">{crush.compression?.ternary_bytes || '?'} bytes</span></div>
                  <div>FF treinado: <span className="text-white">{crush.forward_forward?.trained ? 'Sim' : 'Nao'}</span></div>
                  <div>Genoma: <span className="text-white">{crush.last_genome_bytes || 'N/A'} bytes</span></div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ═══ RIGHT: Broca's Area Terminal ═══ */}
      <div className="w-[22%] bg-black/40 border border-border rounded-xl p-3 flex flex-col font-mono text-[10px]">
        <div className="flex items-center gap-2 mb-3 text-emerald-400 font-semibold border-b border-border pb-2 text-xs">
          <Terminal size={14} />
          <span>Area de Broca (Log)</span>
        </div>
        
        <div ref={brocaRef} className="flex-1 overflow-y-auto text-slate-400 space-y-1">
          <div className="text-emerald-500/40">&gt;&gt; Inicializacao completa</div>
          <div className="text-emerald-500/40">&gt;&gt; Conceitos destilados carregados</div>
          <div className="text-emerald-500/40">&gt;&gt; Matriz de transicao ternaria ativa</div>
          {systemState?.thought_history?.map((thought, i) => (
            <div key={i} className="pt-1 border-t border-white/5">
              <span className="text-slate-600">C{thought.cycle}</span>{' '}
              <span className="text-accent">[IN]</span>{' '}
              <span className="text-slate-400">{(thought.input || '').substring(0, 40)}</span>
              <br />
              <span className="text-emerald-400">[OUT]</span>{' '}
              <span className="text-emerald-200">{(thought.response_preview || '').substring(0, 80)}</span>
              <br />
              <span className="text-slate-600">{Math.round(thought.cycle_time_ms || 0)}ms | {Math.round((thought.confidence || 0) * 100)}%</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};

export default NovaDashboard;
