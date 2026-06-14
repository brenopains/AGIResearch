import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

// Node component for a representation or entity
const CognitiveNode = ({ position, label, color = "#3b82f6", size = 1 }) => {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
      meshRef.current.rotation.x += 0.005;
    }
  });

  return (
    <group position={position}>
      <Sphere ref={meshRef} args={[size, 16, 16]}>
        <meshStandardMaterial color={color} wireframe transparent opacity={0.8} />
      </Sphere>
      <Text
        position={[0, size + 0.5, 0]}
        fontSize={0.5}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
    </group>
  );
};

const WorldModelGraph = ({ entities, links }) => {
  // Simple layout logic for entities
  const nodes = useMemo(() => {
    const keys = Object.keys(entities);
    return keys.map((key, i) => {
      // Circular layout
      const angle = (i / keys.length) * Math.PI * 2;
      const radius = 8;
      return {
        id: key,
        position: [Math.cos(angle) * radius, Math.sin(angle) * radius, -10],
        label: key
      };
    });
  }, [entities]);

  // Create links between nodes if any
  const lines = useMemo(() => {
    return links.map((link, idx) => {
      const sourceNode = nodes.find(n => n.id === link.cause);
      const targetNode = nodes.find(n => n.id === link.effect);
      if (sourceNode && targetNode) {
        return (
          <Line
            key={idx}
            points={[sourceNode.position, targetNode.position]}
            color="#94a3b8"
            lineWidth={1}
            transparent
            opacity={0.5}
          />
        );
      }
      return null;
    });
  }, [links, nodes]);

  return (
    <group>
      {nodes.map(n => (
        <CognitiveNode key={n.id} position={n.position} label={n.label} color="#10b981" size={0.8} />
      ))}
      {lines}
    </group>
  );
};

const RepresentationTree = ({ library }) => {
  // Layout logic for DAG
  const nodes = useMemo(() => {
    const keys = Object.keys(library);
    return keys.map((key, i) => {
      return {
        id: key,
        position: [i * 3 - (keys.length * 1.5), -5, i * 2],
        label: library[key].id
      };
    });
  }, [library]);

  return (
    <group>
      {nodes.map(n => (
        <CognitiveNode key={n.id} position={n.position} label={n.label} color="#8b5cf6" size={1.2} />
      ))}
    </group>
  );
};

const Scene = ({ state }) => {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
      
      {/* Central "Nucleus" */}
      <CognitiveNode position={[0, 0, 0]} label="Cognitive Core" color="#ef4444" size={2} />

      {/* World Models around nucleus */}
      <WorldModelGraph 
        entities={state.world_model?.entities || {}} 
        links={state.world_model?.causal_links || []} 
      />

      {/* Representation Evolutionary Tree */}
      <RepresentationTree library={state.representations || {}} />
    </>
  );
};

const CognitiveCanvas = ({ state }) => {
  return (
    <Canvas camera={{ position: [0, 0, 20], fov: 60 }}>
      <color attach="background" args={['#0f172a']} />
      <Scene state={state} />
    </Canvas>
  );
};

export default CognitiveCanvas;
