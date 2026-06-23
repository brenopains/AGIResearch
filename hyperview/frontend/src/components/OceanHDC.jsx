import React, { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

function PointsField({ points, activeLabels }) {
  const ref = useRef()
  const activeSet = useMemo(() => new Set(activeLabels || []), [activeLabels])
  const sizeAttenuation = true

  const geo = useMemo(() => {
    if (!points || points.length === 0) return null
    const n = points.length
    const cx = points.reduce((s, p) => s + p.x, 0) / n
    const cy = points.reduce((s, p) => s + p.y, 0) / n
    const cz = points.reduce((s, p) => s + p.z, 0) / n
    const scl = 0.006
    const pos = new Float32Array(n * 3)
    const col = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const p = points[i]
      pos[i*3] = (p.x-cx)*scl; pos[i*3+1] = (p.y-cy)*scl; pos[i*3+2] = (p.z-cz)*scl
      const c = p.color || [0.1, 0.7, 1.0]
      col[i*3]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2]
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
    return g
  }, [points])

  useFrame(({ clock }) => {
    if (!ref.current || !geo) return
    const t = clock.elapsedTime
    const ca = geo.attributes.color
    if (!ca) return
    for (let i = 0; i < points.length; i++) {
      const lbl = points[i]?.label || ''
      const isActive = activeSet.has(lbl)
      if (isActive) {
        const pulse = 0.5 + 0.5 * Math.sin(t*3 + i*0.3)
        ca.array[i*3] = Math.min(1, 0.3 + pulse*0.7)
        ca.array[i*3+1] = Math.min(1, 0.8 + pulse*0.2)
        ca.array[i*3+2] = 1.0
      }
    }
    ca.needsUpdate = true
  })

  if (!geo) return null

  return (
    <points ref={ref} geometry={geo} frustumCulled={false}>
      <pointsMaterial size={0.5} vertexColors transparent opacity={0.8} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  )
}

export default function OceanHDC({ points, edges, activeConcepts }) {
  return (
    <div className="ocean-container">
      <Canvas camera={{position:[0,3,15],fov:55}} style={{background:'#060a14'}}>
        <ambientLight intensity={0.3}/>
        <pointLight position={[10,10,10]} intensity={0.5} color="#00ddff"/>
        <fog attach="fog" args={['#060a14',18,40]}/>
        {points && points.length > 0 && <PointsField points={points} activeLabels={activeConcepts} />}
        <OrbitControls enablePan enableZoom enableRotate autoRotate={false} rotateSpeed={0.4}/>
      </Canvas>
    </div>
  )
}
