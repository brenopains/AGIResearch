"""
SpatialTemporalMemory — Memoria Espaco-Temporal com Tracking de Objetos
========================================================================
Conecta a percepcao visual em tempo real ao TypedCognitiveGraph:
  - Trackeia objetos entre frames (centroid matching / optical flow)
  - Computa relacoes espaciais (above, below, left_of, right_of, moving_toward)
  - Detecta eventos temporais (apareceu, desapareceu, colidiu, mudou trajetoria)
  - Alimenta TypedCognitiveGraph com relacoes tipadas espaco-temporais

Pipeline:
  frame -> VisualCortex.features() + deteccao de objetos
       -> SpatialTemporalMemory.track()
       -> TypedCognitiveGraph.add_relation('objeto_A', 'LEFT_OF', 'objeto_B')
       -> raciocinio simbolico sobre relacoes espaciais
"""
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set
import time
import math


class TrackedObject:
    """Um objeto sendo rastreado ao longo do tempo."""
    
    def __init__(self, obj_id: str, bbox: Tuple[int, int, int, int],
                 centroid: Tuple[float, float], features: Dict,
                 first_seen: float):
        self.obj_id = obj_id
        self.bbox_history = deque(maxlen=30)  # ultimos 30 frames
        self.centroid_history = deque(maxlen=30)
        self.features_history = deque(maxlen=30)
        
        self.bbox_history.append(bbox)
        self.centroid_history.append(centroid)
        self.features_history.append(features)
        
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.frames_tracked = 1
        self.is_occluded = False
        self.velocity = (0.0, 0.0)  # (dx, dy) por frame
        self.acceleration = (0.0, 0.0)
        
    def update(self, bbox: Tuple[int, int, int, int],
               centroid: Tuple[float, float], features: Dict,
               timestamp: float):
        """Atualiza posicao e computa derivadas temporais."""
        if len(self.centroid_history) >= 2:
            prev = self.centroid_history[-1]
            self.velocity = (centroid[0] - prev[0], centroid[1] - prev[1])
        
        if len(self.centroid_history) >= 3:
            prev_v = (self.centroid_history[-1][0] - self.centroid_history[-2][0],
                      self.centroid_history[-1][1] - self.centroid_history[-2][1])
            self.acceleration = (self.velocity[0] - prev_v[0], self.velocity[1] - prev_v[1])
        
        self.bbox_history.append(bbox)
        self.centroid_history.append(centroid)
        self.features_history.append(features)
        self.last_seen = timestamp
        self.frames_tracked += 1
    
    def predict_position(self, frames_ahead: int = 5) -> Tuple[float, float]:
        """Prediz posicao futura por extrapolacao linear."""
        if len(self.centroid_history) < 2:
            return self.centroid_history[-1] if self.centroid_history else (0, 0)
        cx, cy = self.centroid_history[-1]
        vx, vy = self.velocity
        ax, ay = self.acceleration
        return (cx + vx * frames_ahead + 0.5 * ax * frames_ahead**2,
                cy + vy * frames_ahead + 0.5 * ay * frames_ahead**2)
    
    def centroid(self) -> Tuple[float, float]:
        return self.centroid_history[-1] if self.centroid_history else (0, 0)
    
    def speed(self) -> float:
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)


class SpatialTemporalMemory:
    """
    Memoria espaco-temporal: trackeia objetos, detecta relacoes espaciais,
    eventos temporais, e alimenta o grafo de conhecimento tipado.
    
    Funciona puramente com features do VisualCortex (bounding boxes
    detectadas por movimento/contorno) - sem depender de YOLO/detectron.
    """
    
    # Tipos de relacao espacial
    SPATIAL_RELATIONS = {
        'LEFT_OF', 'RIGHT_OF', 'ABOVE', 'BELOW',
        'NEAR', 'FAR', 'TOUCHING',
        'MOVING_TOWARD', 'MOVING_AWAY',
        'FASTER_THAN', 'SLOWER_THAN',
    }
    
    def __init__(self, tcg=None, iou_threshold: float = 0.3,
                 max_objects: int = 20):
        self.tcg = tcg  # TypedCognitiveGraph opcional
        self.iou_threshold = iou_threshold
        self.max_objects = max_objects
        
        self.objects: Dict[str, TrackedObject] = {}
        self._next_id = 0
        self._frame_count = 0
        self._last_frame = None
        self._background = None
        
        self._events: List[Dict] = []  # ultimos eventos detectados
        self._spatial_relations: List[Tuple[str, str, str, float]] = []
        
        print(f"[STM] SpatialTemporalMemory initialized "
              f"(max_objects={max_objects})")
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> Dict:
        """
        Processa um frame RGB (H, W, 3):
          1. Detecta regioes de movimento/mudanca
          2. Match com objetos existentes (centroid + IoU)
          3. Computa relacoes espaciais entre objetos
          4. Detecta eventos (apareceu, colidiu, etc.)
          5. Alimenta TCG se disponivel
        """
        if timestamp is None:
            timestamp = time.time()
        
        self._frame_count += 1
        if self._last_frame is None:
            self._last_frame = frame
            return {'objects': list(self.objects.keys()), 'events': []}
        
        # 1. Motion detection via frame differencing
        diff = np.abs(frame.astype(np.float32) - self._last_frame.astype(np.float32))
        motion_mag = np.mean(diff, axis=2)  # (H, W)
        
        # Threshold motion
        motion_mask = motion_mag > 25.0  # parametro de sensibilidade
        
        # 2. Extract moving regions as bounding boxes
        detected_boxes = self._extract_boxes(motion_mask)
        
        # 3. Match with existing objects
        self._match_and_update(detected_boxes, frame, timestamp)
        
        # 4. Compute spatial relations
        self._compute_spatial_relations(timestamp)
        
        # 5. Detect events
        events = self._detect_events(timestamp)
        
        # 6. Feed to TCG
        if self.tcg:
            self._feed_to_tcg()
        
        self._last_frame = frame
        
        result = {
            'frame': self._frame_count,
            'n_objects': len(self.objects),
            'objects': list(self.objects.keys()),
            'n_moving': sum(1 for o in self.objects.values() if o.speed() > 2),
            'spatial_relations': len(self._spatial_relations),
            'events': events,
        }
        return result
    
    def query_objects(self, predicate: str = None) -> List[Dict]:
        """Query objetos por predicado (ex: 'moving', 'left_of:X')."""
        results = []
        for oid, obj in self.objects.items():
            info = {
                'id': oid,
                'centroid': obj.centroid(),
                'velocity': obj.velocity,
                'speed': obj.speed(),
                'frames_tracked': obj.frames_tracked,
                'is_occluded': obj.is_occluded,
            }
            if predicate == 'moving' and obj.speed() > 2:
                results.append(info)
            elif predicate is None:
                results.append(info)
        return results
    
    def spatial_query(self, relation: str, obj_id: str) -> List[str]:
        """Encontra objetos com relacao espacial a obj_id."""
        results = []
        for frm, to, rel, conf in self._spatial_relations:
            if frm == obj_id and rel == relation:
                results.append(to)
            elif to == obj_id and rel == relation:
                results.append(frm)
        return results
    
    # ─── Metodos Internos ────────────────────────────────────────────
    
    def _extract_boxes(self, motion_mask: np.ndarray) -> List[Tuple]:
        """Extrai bounding boxes de regioes de movimento."""
        from scipy import ndimage
        labeled, n_features = ndimage.label(motion_mask)
        boxes = []
        for i in range(1, n_features + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < 20:  # min area filter
                continue
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)
            boxes.append({
                'bbox': (x1, y1, x2, y2),
                'centroid': (cx, cy),
                'area': area,
            })
        # Sort by area, largest first
        boxes.sort(key=lambda b: -b['area'])
        return boxes[:self.max_objects]
    
    def _match_and_update(self, detected_boxes: List[Dict],
                          frame: np.ndarray, timestamp: float):
        """Match detected boxes with tracked objects via centroid distance + IoU."""
        matched = set()
        used_detections = set()
        
        # Try to match existing objects to detections
        for oid, obj in list(self.objects.items()):
            best_match = None
            best_dist = float('inf')
            
            for di, det in enumerate(detected_boxes):
                if di in used_detections:
                    continue
                dc = det['centroid']
                oc = obj.centroid()
                dist = math.sqrt((dc[0] - oc[0])**2 + (dc[1] - oc[1])**2)
                if dist < best_dist and dist < 100:  # max match distance
                    best_dist = dist
                    best_match = di
            
            if best_match is not None:
                det = detected_boxes[best_match]
                used_detections.add(best_match)
                features = self._extract_features(frame, det['bbox'])
                obj.update(det['bbox'], det['centroid'], features, timestamp)
                matched.add(oid)
        
        # Objects not matched: occlude
        for oid, obj in self.objects.items():
            if oid not in matched:
                obj.is_occluded = True
                if time.time() - obj.last_seen > 2.0:  # remove after 2s
                    self._events.append({
                        'type': 'DISAPPEARED',
                        'obj_id': oid,
                        'time': timestamp,
                    })
        
        # Remove stale objects
        self.objects = {oid: obj for oid, obj in self.objects.items()
                       if time.time() - obj.last_seen < 3.0}
        
        # New detections become new objects
        for di, det in enumerate(detected_boxes):
            if di not in used_detections:
                oid = f"obj_{self._next_id}"
                self._next_id += 1
                features = self._extract_features(frame, det['bbox'])
                obj = TrackedObject(oid, det['bbox'], det['centroid'],
                                   features, timestamp)
                self.objects[oid] = obj
                self._events.append({
                    'type': 'APPEARED',
                    'obj_id': oid,
                    'centroid': det['centroid'],
                    'time': timestamp,
                })
    
    def _extract_features(self, frame: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> Dict:
        """Extrai features visuais de uma bounding box."""
        x1, y1, x2, y2 = bbox
        region = frame[max(0, y1):min(frame.shape[0], y2),
                       max(0, x1):min(frame.shape[1], x2)]
        if region.size == 0:
            return {}
        
        h, w, _ = region.shape
        gray = np.mean(region, axis=2)
        
        return {
            'mean_color': tuple(np.mean(region, axis=(0, 1)).astype(int)),
            'brightness': float(np.mean(gray) / 255.0),
            'contrast': float(np.std(gray) / 128.0),
            'area': h * w,
            'aspect_ratio': w / max(h, 1),
        }
    
    def _compute_spatial_relations(self, timestamp: float):
        """Calcula relacoes espaciais entre todos os pares de objetos."""
        self._spatial_relations = []
        obj_list = list(self.objects.items())
        
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                id_a, obj_a = obj_list[i]
                id_b, obj_b = obj_list[j]
                
                ca = obj_a.centroid()
                cb = obj_b.centroid()
                
                dx = cb[0] - ca[0]
                dy = cb[1] - ca[1]
                dist = math.sqrt(dx**2 + dy**2)
                
                # Spatial predicates
                threshold = 50.0
                
                if abs(dx) > threshold and abs(dx) > abs(dy):
                    if dx > 0:
                        self._add_spatial(id_a, id_b, 'LEFT_OF', 0.8)
                        self._add_spatial(id_b, id_a, 'RIGHT_OF', 0.8)
                    else:
                        self._add_spatial(id_a, id_b, 'RIGHT_OF', 0.8)
                        self._add_spatial(id_b, id_a, 'LEFT_OF', 0.8)
                
                elif abs(dy) > threshold and abs(dy) > abs(dx):
                    if dy > 0:
                        self._add_spatial(id_a, id_b, 'ABOVE', 0.8)
                        self._add_spatial(id_b, id_a, 'BELOW', 0.8)
                    else:
                        self._add_spatial(id_a, id_b, 'BELOW', 0.8)
                        self._add_spatial(id_b, id_a, 'ABOVE', 0.8)
                
                if dist < threshold:
                    self._add_spatial(id_a, id_b, 'NEAR', 1.0 - dist / threshold)
                    self._add_spatial(id_b, id_a, 'NEAR', 1.0 - dist / threshold)
                
                # Relative velocity
                va = obj_a.velocity
                vb = obj_b.velocity
                rel_vx = vb[0] - va[0]
                rel_vy = vb[1] - va[1]
                
                if rel_vx * dx + rel_vy * dy > 0:
                    conf = min(0.9, (rel_vx * dx + rel_vy * dy) / (dist * 10 + 1))
                    self._add_spatial(id_a, id_b, 'MOVING_AWAY', conf)
                    self._add_spatial(id_b, id_a, 'MOVING_TOWARD', conf)
                else:
                    conf = min(0.9, -(rel_vx * dx + rel_vy * dy) / (dist * 10 + 1))
                    self._add_spatial(id_a, id_b, 'MOVING_TOWARD', conf)
                    self._add_spatial(id_b, id_a, 'MOVING_AWAY', conf)
    
    def _add_spatial(self, frm: str, to: str, relation: str, confidence: float):
        """Adiciona relacao espacial a lista."""
        if confidence > 0.3:
            self._spatial_relations.append((frm, to, relation, confidence))
    
    def _detect_events(self, timestamp: float) -> List[Dict]:
        """Detecta eventos (colisoes, mudancas de velocidade)."""
        events = []
        
        # Collision detection: objects NEAR each other AND moving toward
        for frm, to, rel, conf in self._spatial_relations:
            if rel == 'MOVING_TOWARD' and conf > 0.7:
                events.append({
                    'type': 'APPROACHING',
                    'obj_a': frm, 'obj_b': to,
                    'confidence': conf,
                    'time': timestamp,
                })
        
        # Speed change events
        for oid, obj in self.objects.items():
            if abs(obj.acceleration[0]) > 5 or abs(obj.acceleration[1]) > 5:
                events.append({
                    'type': 'DIRECTION_CHANGE',
                    'obj_id': oid,
                    'acceleration': obj.acceleration,
                    'time': timestamp,
                })
        
        self._events = (self._events + events)[-50:]  # keep last 50
        return events
    
    def _feed_to_tcg(self):
        """Alimenta TypedCognitiveGraph com relacoes espaciais."""
        if self.tcg is None:
            return
        
        for frm, to, rel, conf in self._spatial_relations:
            # Mapear relacoes espaciais para tipos TCG
            tcg_rel = rel if rel in ('LEFT_OF', 'RIGHT_OF', 'ABOVE', 'BELOW') else 'CORRELATED'
            self.tcg.add_relation(frm, to, tcg_rel, confidence=conf, observation_evidence=conf)
    
    def get_state(self) -> Dict:
        """Retorna estado completo para visualizacao."""
        return {
            'frame': self._frame_count,
            'n_objects': len(self.objects),
            'objects': {oid: {
                'centroid': obj.centroid(),
                'velocity': obj.velocity,
                'speed': obj.speed(),
                'bbox': list(obj.bbox_history[-1]) if obj.bbox_history else None,
                'frames': obj.frames_tracked,
                'occluded': obj.is_occluded,
            } for oid, obj in self.objects.items()},
            'spatial_relations': [(f, t, r, f"{c:.2f}") 
                                  for f, t, r, c in self._spatial_relations[:20]],
            'events': self._events[-5:],
        }
