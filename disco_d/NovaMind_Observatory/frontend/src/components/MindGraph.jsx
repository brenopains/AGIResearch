import React, {useMemo, useRef, useState, useEffect, Component} from 'react'
import {Canvas, useThree} from '@react-three/fiber'
import {OrbitControls, Html} from '@react-three/drei'
import * as THREE from 'three'

const idOf = v => typeof v === 'object' ? (v?.id || v?.module_id) : v
const hash = s => { let h=2166136261; for(const c of String(s)) h=Math.imul(h^c.charCodeAt(0),16777619); return h>>>0 }
const labelOf = n => n?.label || n?.name || n?.id?.split('.').pop() || ''
const PALETTE=['#3c7dff','#28a9aa','#8175df','#5a94bc','#cca76b','#79a5bb']

function layout(nodes, edges, mode) {
  const degree=new Map(nodes.map(n=>[n.id,0]))
  for(const e of edges){for(const id of [idOf(e.source),idOf(e.target)]) if(degree.has(id)) degree.set(id,degree.get(id)+1)}
  const sorted=[...nodes].sort((a,b)=>(degree.get(b.id)-degree.get(a.id)) || a.id.localeCompare(b.id))
  const out=new Map()
  sorted.forEach((n,i)=>{
    const h=hash(n.id), angle=i*2.399963229728653
    let x,y,z
    if(mode==='layers'){
      const layer=n.kind==='class'?1:n.kind==='function'?2:Math.min(4,Math.floor(Math.log2((degree.get(n.id)||0)+1)))
      const index=sorted.slice(0,i).filter(v=>Math.min(4,Math.floor(Math.log2((degree.get(v.id)||0)+1)))===layer).length
      const r=.45+Math.sqrt(index)*.3
      x=Math.cos(angle)*r;z=Math.sin(angle)*r;y=(layer-2)*.9
    }else{
      const phi=Math.acos(1-2*(i+.5)/Math.max(sorted.length,1))
      const r=i<7 ? .65+(i*.13) : 1.7+(h%80)/100
      x=Math.cos(angle)*Math.sin(phi)*r*1.16
      y=Math.cos(phi)*r*.91+.15
      z=Math.sin(angle)*Math.sin(phi)*r*.85
    }
    out.set(n.id,{position:[x,y,z],degree:degree.get(n.id)||0,color:PALETTE[h%PALETTE.length],node:n})
  })
  return out
}

function makeLines(edges, locations) {
  const values=[]
  for(const e of edges){
    const a=locations.get(idOf(e.source)), b=locations.get(idOf(e.target))
    if(!a || !b || a===b)continue
    values.push(...a.position,...b.position)
  }
  const geometry=new THREE.BufferGeometry()
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(values,3))
  return geometry
}

function Points({nodes,locations,selectedId,activeIds,onSelect,onHover}) {
  const ref=useRef()
  const {invalidate}=useThree()
  useEffect(()=>{
    if(!ref.current)return
    const obj=new THREE.Object3D(), color=new THREE.Color()
    nodes.forEach((node,i)=>{
      const loc=locations.get(node.id)
      const selected=node.id===selectedId, active=activeIds.has(node.id)
      const radius=selected ? .13 : active ? .105 : .032+Math.min(loc.degree,30)*.0019
      obj.position.set(...loc.position);obj.scale.setScalar(radius);obj.updateMatrix()
      ref.current.setMatrixAt(i,obj.matrix)
      color.set(selected?'#175fff':active?'#2abca8':loc.color)
      ref.current.setColorAt(i,color)
    })
    ref.current.instanceMatrix.needsUpdate=true
    if(ref.current.instanceColor) ref.current.instanceColor.needsUpdate=true
    ref.current.computeBoundingSphere()
    invalidate()
  },[nodes,locations,selectedId,activeIds,invalidate])
  return <instancedMesh ref={ref} args={[null,null,nodes.length]} onClick={e=>{e.stopPropagation();onSelect?.(nodes[e.instanceId]?.id)}} onPointerOver={e=>{e.stopPropagation();onHover(nodes[e.instanceId]?.id);document.body.style.cursor='pointer'}} onPointerOut={()=>{onHover(null);document.body.style.cursor=''}}>
    <sphereGeometry args={[1,12,12]}/>
    <meshStandardMaterial roughness={.22} metalness={.22} emissive="#476485" emissiveIntensity={.25}/>
  </instancedMesh>
}

function CameraReset({reset}) {
  const {camera}=useThree()
  useEffect(()=>{camera.position.set(6.2,3.4,7.2);camera.lookAt(0,0,0)},[reset,camera])
  return null
}

function Scene({nodes,edges,events,selectedId,onSelect,mode,reset}) {
  const [hover,setHover]=useState(null)
  const locations=useMemo(()=>layout(nodes,edges,mode),[nodes,edges,mode])
  const connections=useMemo(()=>makeLines(edges,locations),[edges,locations])
  const activeIds=useMemo(()=>new Set(events.map(e=>e.module_id).filter(Boolean)),[events])
  const observedEdges=useMemo(()=>{
    const ids=new Map(events.map(e=>[e.id,e]))
    const set=new Set(),result=[]
    for(const e of events){
      const parent=ids.get(e.parent_id)
      if(parent?.module_id && e.module_id && parent.module_id!==e.module_id){
        const key=parent.module_id+'|'+e.module_id
        if(!set.has(key)){set.add(key);result.push({source:parent.module_id,target:e.module_id})}
      }
      if(e.kind==='observed_call' && e.data?.caller && e.data?.callee){
        result.push({source:e.data.caller,target:e.data.callee})
      }
    }return result
  },[events])
  const activity=useMemo(()=>makeLines(observedEdges,locations),[observedEdges,locations])
  const related=useMemo(()=>makeLines(edges.filter(e=>idOf(e.source)===selectedId||idOf(e.target)===selectedId),locations),[edges,selectedId,locations])
  useEffect(()=>()=>connections.dispose(),[connections])
  useEffect(()=>()=>activity.dispose(),[activity])
  useEffect(()=>()=>related.dispose(),[related])
  const important=useMemo(()=>[...nodes].sort((a,b)=>locations.get(b.id).degree-locations.get(a.id).degree).slice(0,5),[nodes,locations])
  const labels=[...new Set([...important.map(n=>n.id),selectedId,hover])].filter(id=>locations.has(id))
  return <>
    <ambientLight intensity={1.7}/>
    <directionalLight position={[5,6,5]} intensity={2.4} color="#e8f3ff"/>
    <pointLight position={[-4,1,-2]} intensity={4} color="#8ab2ff"/>
    <CameraReset reset={reset}/>
    <OrbitControls key={reset} makeDefault enablePan enableDamping dampingFactor={.08} minDistance={3.3} maxDistance={15} maxPolarAngle={Math.PI*.88}/>
    <group position={[0,.25,0]}>
      <lineSegments geometry={connections}><lineBasicMaterial color="#87acd0" transparent opacity={.17} depthWrite={false}/></lineSegments>
      <lineSegments geometry={related}><lineBasicMaterial color="#458aff" transparent opacity={.67} depthWrite={false}/></lineSegments>
      <lineSegments geometry={activity}><lineBasicMaterial color="#15aa93" transparent opacity={.95}/></lineSegments>
      <Points nodes={nodes} locations={locations} selectedId={selectedId} activeIds={activeIds} onSelect={onSelect} onHover={setHover}/>
      {labels.map(id=>{
        const loc=locations.get(id),focus=id===selectedId||id===hover
        return <Html key={id} position={loc.position} center distanceFactor={9} zIndexRange={[20,0]} style={{pointerEvents:'none',transform:'translateY(-21px)'}}>
          <div style={{whiteSpace:'nowrap',font:'500 11px "Segoe UI",sans-serif',color:focus?'#164c9a':'#567088',padding:focus?'7px 11px':'3px 7px',border:focus?'1px solid rgba(108,157,225,.5)':'1px solid rgba(208,221,233,.6)',borderRadius:focus?9:6,background:focus?'rgba(250,253,255,.94)':'rgba(249,252,255,.76)',boxShadow:focus?'0 5px 24px rgba(70,122,183,.13)':'none'}}>
            <span style={{display:'inline-block',width:5,height:5,background:activeIds.has(id)?'#13ac95':loc.color,borderRadius:10,marginRight:6}}/>{labelOf(loc.node)}
            {focus&&<span style={{display:'block',fontSize:9,color:'#8092a5',marginTop:3}}>{loc.degree} relações no código{activeIds.has(id)?' · observado':''}</span>}
          </div>
        </Html>
      })}
      {mode==='layers'&&[0,1,2,3,4].map(i=><group key={i} position={[0,(i-2)*.9,0]}>
        <mesh rotation={[-Math.PI/2,0,0]}><planeGeometry args={[6,5]}/><meshBasicMaterial color="#7eabdb" transparent opacity={.035} side={THREE.DoubleSide} depthWrite={false}/></mesh>
        <mesh><boxGeometry args={[6,.012,5]}/><meshBasicMaterial wireframe color="#86b0da" transparent opacity={.17}/></mesh>
      </group>)}
    </group>
    <group position={[0,-2.7,0]}>
      <mesh rotation={[-Math.PI/2,0,0]}><circleGeometry args={[3.3,96]}/><meshBasicMaterial color="#d5e4f2" transparent opacity={.15} side={THREE.DoubleSide}/></mesh>
      {[2.3,2.65,3.05,3.3].map((r,i)=><mesh key={r} rotation={[-Math.PI/2,0,0]} position={[0,i*.008,0]}><ringGeometry args={[r,r+.012,100]}/><meshBasicMaterial color="#8bb2db" transparent opacity={.23} side={THREE.DoubleSide}/></mesh>)}
    </group>
  </>
}

function Map2D({nodes,edges,events,selectedId,onSelect}) {
  const loc=useMemo(()=>layout(nodes,edges,'graph'),[nodes,edges])
  const active=new Set(events.map(e=>e.module_id))
  const point=id=>{const p=loc.get(id)?.position||[0,0,0];return [400+p[0]*100,260+p[1]*92]}
  return <svg viewBox="0 0 800 530" role="img" aria-label="Mapa de módulos e dependências" style={{width:'100%',height:'100%',minHeight:400}}>
    {edges.map((e,i)=>{const a=point(idOf(e.source)),b=point(idOf(e.target));return <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#a4bbd4" strokeOpacity=".27"/>})}
    {nodes.map(n=>{const [x,y]=point(n.id),sel=n.id===selectedId;return <g key={n.id} onClick={()=>onSelect?.(n.id)} onKeyDown={e=>e.key==='Enter'&&onSelect?.(n.id)} tabIndex="0" role="button" aria-label={labelOf(n)} style={{cursor:'pointer'}}>
      <circle cx={x} cy={y} r={sel?8:active.has(n.id)?6:3.5} fill={active.has(n.id)?'#14ac98':sel?'#276dff':loc.get(n.id).color}/>
      <title>{labelOf(n)}</title>{sel&&<text x={x+12} y={y+4} fill="#204973" fontSize="13">{labelOf(n)}</text>}
    </g>})}
  </svg>
}

class GraphBoundary extends Component {
  state={failed:false}
  static getDerivedStateFromError(){return {failed:true}}
  render(){return this.state.failed?this.props.fallback:this.props.children}
}

export default function MindGraph({nodes=[],edges=[],events=[],selectedId,onSelect,viewMode='graph'}) {
  const [reset,setReset]=useState(0),[flat,setFlat]=useState(false)
  const safeNodes=useMemo(()=>nodes.filter(n=>n.id),[nodes])
  const safeEdges=useMemo(()=>{const ids=new Set(safeNodes.map(n=>n.id));return edges.filter(e=>ids.has(idOf(e.source))&&ids.has(idOf(e.target)))},[safeNodes,edges])
  const fallback=<Map2D nodes={safeNodes} edges={safeEdges} events={events} selectedId={selectedId} onSelect={onSelect}/>
  useEffect(()=>()=>{document.body.style.cursor=''},[])
  return <div className="mind-graph" style={{height:'100%',minHeight:420,position:'relative',isolation:'isolate'}}>
    {safeNodes.length ? flat||viewMode==='2d'? fallback :
      <GraphBoundary fallback={fallback}>
        <Canvas dpr={[1,1.6]} camera={{position:[6.2,3.4,7.2],fov:42}} gl={{antialias:true,alpha:true,powerPreference:'low-power'}} style={{background:'transparent'}} onPointerMissed={()=>{}} frameloop="demand">
          <Scene nodes={safeNodes} edges={safeEdges} events={events} selectedId={selectedId} onSelect={onSelect} mode={viewMode} reset={reset}/>
        </Canvas>
      </GraphBoundary>
      : <div style={{height:420,display:'grid',placeItems:'center',color:'#8795a5',fontSize:13}}>O mapa aparece quando o catálogo de código estiver conectado.</div>}
    <div style={{position:'absolute',bottom:17,right:17,display:'flex',gap:6,zIndex:25}}>
      <button type="button" onClick={()=>setFlat(v=>!v)} title="Alternar mapa 2D e 3D" aria-label="Alternar mapa 2D e 3D" style={smallButton}>{flat?'3D':'2D'}</button>
      <button type="button" onClick={()=>setReset(v=>v+1)} title="Recentrar câmera" aria-label="Recentrar câmera" style={smallButton}>⌖</button>
    </div>
    <div style={{position:'absolute',bottom:18,left:20,color:'#8a9aaa',fontSize:10,letterSpacing:'.025em',maxWidth:'68%',pointerEvents:'none'}}>Arraste para explorar · role para aproximar · clique para inspecionar</div>
  </div>
}
const smallButton={width:31,height:29,border:'1px solid rgba(176,198,219,.6)',borderRadius:8,background:'rgba(255,255,255,.8)',color:'#506a87',cursor:'pointer',fontSize:11}

