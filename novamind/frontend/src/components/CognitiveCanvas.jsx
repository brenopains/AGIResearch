import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

// Color map for different manifold modalities
const MODALITY_COLORS = {
  archetype: '#f87171', // Red
  visual: '#34d399',    // Green
  audio: '#22d3ee',     // Cyan
  text: '#f59e0b',      // Yellow/Gold
  unknown: '#94a3b8'    // Gray
};

// Render causal edges from the SCM (Structural Causal Model)
const CausalLink = ({ start, end, type }) => {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end)
  ], [start, end]);

  const lineGeometry = useMemo(() => 
    new THREE.BufferGeometry().setFromPoints(points)
  , [points]);

  const color = type === 'pytorch_latent_edge' ? '#34d399' : '#a78bfa';

  return (
    <line geometry={lineGeometry}>
      <lineBasicMaterial 
        attach="material" 
        color={color} 
        transparent 
        opacity={0.3} 
        linewidth={1.5} 
      />
    </line>
  );
};

// Poincaré Sphere containing all manifolds, causal links, and semantic rules
const PoincareSphere = ({ state }) => {
  const groupRef = useRef();
  
  const concepts = state?.perception?.all_concepts || [];
  const edges = state?.world_model?.edges || [];
  const predError = state?.autonomous_learning?.prediction_error || 0;

  // Map concepts by label for quick link lookups
  const conceptsMap = useMemo(() => {
    const m = {};
    concepts.forEach(c => {
      m[c.label] = c;
    });
    return m;
  }, [concepts]);

  // Slowly rotate the entire manifold space, speeding up during prediction errors (frustration)
  useFrame((_, delta) => {
    if (groupRef.current) {
      const speed = 0.05 + predError * 0.5;
      groupRef.current.rotation.y += speed * delta;
      groupRef.current.rotation.z += (speed * 0.2) * delta;
    }
  });

  return (
    <group ref={groupRef}>
      {/* 1. Poincaré Hyperbolic Bound Sphere (Futuristic grid overlay) */}
      <mesh>
        <sphereGeometry args={[12, 24, 24]} />
        <meshBasicMaterial 
          color="#3b82f6" 
          wireframe 
          transparent 
          opacity={0.2} 
        />
      </mesh>
      
      {/* 2. Concept Nodes (Projected 3D Manifolds) */}
      {concepts.map((c, idx) => {
        const coords = c.coords3d || [0, 0, 0];
        const color = MODALITY_COLORS[c.modality] || MODALITY_COLORS.unknown;
        const size = 0.15 + (c.activation || 0.5) * 0.15;
        
        return (
          <group key={idx} position={coords}>
            {/* Core Node Particle */}
            <mesh>
              <sphereGeometry args={[size, 16, 16]} />
              <meshBasicMaterial color={color} />
            </mesh>
            
            {/* Halo ring for high-activation manifolds */}
            {c.activation > 0.6 && (
              <mesh>
                <ringGeometry args={[size * 1.3, size * 1.5, 16]} />
                <meshBasicMaterial color={color} transparent opacity={0.6} side={THREE.DoubleSide} />
              </mesh>
            )}
            
            <Text 
              position={[0, size + 0.3, 0]} 
              fontSize={0.25} 
              color="#f1f5f9"
              anchorX="center" 
              anchorY="middle"
            >
              {c.label}
            </Text>
          </group>
        );
      })}

      {/* 3. Causal Connections (Causal Links from SCM) */}
      {edges.map((edge, idx) => {
        const srcNode = conceptsMap[edge.source];
        const tgtNode = conceptsMap[edge.target];
        if (srcNode && tgtNode && srcNode.coords3d && tgtNode.coords3d) {
          return (
            <CausalLink 
              key={idx} 
              start={srcNode.coords3d} 
              end={tgtNode.coords3d} 
              type={edge.type}
            />
          );
        }
        return null;
      })}
    </group>
  );
};

const CognitiveCanvas = ({ state }) => {
  return (
    <Canvas
      camera={{ position: [0, 5, 20], fov: 60 }}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      dpr={window.devicePixelRatio || 1}
    >
      <color attach="background" args={['#020617']} />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />
      
      <OrbitControls 
        makeDefault 
        enableDamping 
        dampingFactor={0.05} 
        maxDistance={35} 
        minDistance={3} 
      />

      <PoincareSphere state={state} />
    </Canvas>
  );
};

export default CognitiveCanvas;
