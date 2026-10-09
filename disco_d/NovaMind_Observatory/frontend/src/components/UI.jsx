import React from 'react';
export const fmt=n=>typeof n==='number'&&Number.isFinite(n)?new Intl.NumberFormat('pt-BR',{maximumFractionDigits:2}).format(n):'—';
export const short=(s,n=90)=>String(s??'').length>n?`${String(s).slice(0,n)}…`:String(s??'');
export const moduleLabel=id=>String(id||'').replace(/\\/g,'/').split('/').pop().replace(/\.py$/,'');
export const seconds=ms=>typeof ms==='number'?`${(ms/1000).toFixed(ms<1000?3:2)} s`:'ordem registrada';
export const eventTone=e=>/failed|reject|error/.test(e?.kind)?'rose':/accept|complete|result/.test(e?.kind)?'mint':/memory/.test(e?.kind)?'violet':'blue';
export function Icon({name,size=20,...props}){
 const p={orbit:<><circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(-35 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(65 12 12)"/></>,timeline:<><path d="M6 3v18M10 5h10M10 12h7M10 19h10"/><circle cx="6" cy="5" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="6" cy="19" r="2"/></>,spark:<path d="m12 2 2.7 6.9L22 12l-7.3 3.1L12 22l-2.7-6.9L2 12l7.3-3.1Z"/>,memory:<><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/></>,cube:<path d="m12 2 9 5v10l-9 5-9-5V7Z M3 7l9 5 9-5M12 12v10"/>,layers:<path d="m12 3 10 5-10 5L2 8Z M2 12l10 5 10-5M2 16l10 5 10-5"/>,arrow:<path d="M5 12h14m-6-6 6 6-6 6"/>,search:<><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></>,chevron:<path d="m9 5 7 7-7 7"/>,down:<path d="m6 9 6 6 6-6"/>,play:<path d="m8 4 12 8-12 8Z"/>,pause:<path d="M8 4v16M16 4v16"/>,step:<path d="m5 5 10 7-10 7ZM19 5v14"/>,back:<path d="m17 5-10 7 10 7ZM3 5v14"/>,close:<path d="m5 5 14 14M5 19 19 5"/>,code:<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>,download:<path d="M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6"/>,plus:<path d="M12 4v16M4 12h16"/>,info:<><circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/></>,pulse:<path d="M2 12h5l3-8 4 16 3-8h5"/>,history:<><path d="M3 4v5h5M3 9a9 9 0 1 1 0 6M12 7v5l3 2"/></>,bolt:<path d="m13 2-9 12h7l-1 8 10-13h-7Z"/>,clock:<><circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/></>};
 return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{p[name]||p.orbit}</svg>;
}
export function Logo(){return <svg className="brand-mark" width="32" height="37" viewBox="0 0 40 44" fill="none" aria-hidden="true"><path d="M20 2 38 36 20 27 2 36Z" fill="currentColor"/><path d="m20 8 0 16-12 6Z" fill="#d7e4ff"/><path d="m20 27 0-19 12 22Z" fill="#7292ff"/></svg>}
export function Empty({icon='orbit',title,children}){return <div className="empty"><span className="empty-symbol"><Icon name={icon} size={28}/></span><h3>{title}</h3><p>{children}</p></div>}
export function Pill({children,tone='blue',dot=false}){return <span className={`pill ${tone}`}>{dot&&<i/>}{children}</span>}
export function SectionHead({eyebrow,title,children}){return <div className="section-head"><div>{eyebrow&&<span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{children}</div>}
export function EventTime({event}){return <span className="mono event-time">{event?.time_basis==='ordinal'||event?.elapsed_ms==null?`#${event?.seq??0}`:`+${seconds(event.elapsed_ms)}`}</span>}
export const JSONText=({value})=><pre className="json">{JSON.stringify(value,null,2)}</pre>;
export function Sparkline({values}){
 const nums=values.filter(v=>typeof v==='number'&&Number.isFinite(v));if(!nums.length)return null;const min=Math.min(...nums),max=Math.max(...nums),span=max-min||1;
 const points=nums.map((v,i)=>`${i/(nums.length-1||1)*400},${64-(v-min)/span*52}`).join(' ');
 return <div className="series"><svg viewBox="0 0 400 74" role="img" aria-label={`${nums.length} valores numéricos, mínimo ${min}, máximo ${max}`}><path d="M0 65h400" stroke="#dce3ec"/><polyline points={points} fill="none" stroke="#4a72ed" strokeWidth="2"/></svg><div><span>mín. {fmt(min)}</span><span>{nums.length} valores</span><span>máx. {fmt(max)}</span></div></div>;
}
export function Representation({value,depth=0}){
 if(value==null)return <span className="muted">Não registrado</span>;
 if(typeof value!=='object')return <span className={typeof value==='number'?'mono':''}>{String(value)}</span>;
 if(Array.isArray(value)){
  if(!value.length)return <span className="muted">Nenhum item registrado</span>;
  if(value.every(v=>typeof v==='number'))return <Sparkline values={value}/>;
  if(value.every(v=>Array.isArray(v)&&v.every(n=>typeof n==='number'))&&value.length<=40&&value[0]?.length<=40){const vals=value.flat(),min=Math.min(...vals),max=Math.max(...vals),span=max-min||1;return <div className="matrix-wrap"><div className="matrix" style={{gridTemplateColumns:`repeat(${value[0].length},1fr)`}}>{vals.map((v,i)=><span key={i} style={{background:`hsl(${218+(v-min)/span*70} 62% ${94-(v-min)/span*48}%)`}} title={String(v)}>{vals.length<=100?fmt(v):''}</span>)}</div><small>Matriz numérica registrada · {value.length} × {value[0].length}</small></div>}
  if(value.every(v=>typeof v==='string'))return <div className="token-list">{value.slice(0,100).map((v,i)=><span key={i}>{v}</span>)}{value.length>100&&<span>+{value.length-100} itens</span>}</div>;
  if(value.every(v=>v&&typeof v==='object'&&!Array.isArray(v))&&depth<3){const columns=[...new Set(value.flatMap(Object.keys))].slice(0,6);return <div className="data-table-wrap"><table className="data-table"><thead><tr>{columns.map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{value.slice(0,80).map((row,i)=><tr key={i}>{columns.map(k=><td key={k}>{typeof row[k]==='object'?short(JSON.stringify(row[k]),80):String(row[k]??'—')}</td>)}</tr>)}</tbody></table>{value.length>80&&<small>80 de {value.length} itens. A exportação contém todos os registros.</small>}</div>}
  return <JSONText value={value}/>;
 }
 if(depth>2)return <JSONText value={value}/>;
 return <div className="properties">{Object.entries(value).map(([key,val])=><div className={`property ${typeof val==='object'?'wide':''}`} key={key}><span className="property-name">{key.replaceAll('_',' ')}</span><Representation value={val} depth={depth+1}/></div>)}</div>;
}
export function SourceInfo({source,onOpen}){return source?<div className="source-info"><Icon name="code" size={15}/><div><button className="text-button" onClick={onOpen} title={source.path}>{short(source.path?.split(/[\\/]/).pop()||'Fonte registrada',32)}{source.line?`:${source.line}`:''}</button><small>{source.mode==='imported'?'Registro importado':source.mode==='external'?'Produtor externo · não verificado':source.mode==='observatory'?'Evento do observatório':source.mode==='observed'||source.mode==='runtime'?'Instrumentação real':'Origem declarada'}{source.sha256&&` · SHA ${source.sha256.slice(0,10)}`}</small></div></div>:<p className="muted small-text">A origem deste item não foi registrada.</p>}

