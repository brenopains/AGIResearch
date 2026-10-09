"""Histórias visuais com média e momentos locais, sem semântica do jogo.

As primitivas espaciais são fornecidas. Os centros/coeficientes de funções
posteriores poderão ser aprendidos; não chamar este codec de representação
inventada. Este formato mantém uma trajetória lossless para futuras perguntas.
"""
import numpy as np


def validate_frames(frames):
    frames=np.asarray(frames)
    if frames.ndim!=3 or len(frames)<1 or frames.shape[1:]!=(210,160) or frames.dtype!=np.uint8:
        raise ValueError("esperados quadros uint8 210x160")
    return frames


def coarse(frames):
    frames=validate_frames(frames)
    return frames.reshape(len(frames),21,10,16,10).mean((2,4),dtype=np.float32).reshape(len(frames),336)/255.


def moments(frames):
    """Média e momentos espaciais brutos até ordem dois em cada bloco 10x10.

Não recorta placar, bola ou raquete, não usa recompensa/ação e não reconhece
objetos. Variáveis xy são coordenadas locais genéricas, não estados físicos.
"""
    frames=validate_frames(frames)
    patches=frames.reshape(len(frames),21,10,16,10).astype(np.float32)/255.
    axis=(np.arange(10,dtype=np.float32)-4.5)/4.5
    x,y=axis[None,None,None,None,:],axis[None,None,:,None,None]
    weights=(1.,x,y,x*x,y*y,x*y)
    return np.stack([(patches*w).mean((2,4),dtype=np.float32).reshape(len(frames),336) for w in weights],axis=1)


def windows(trajectory,endpoints):
    trajectory=validate_frames(trajectory)
    endpoints=np.asarray(endpoints)
    if endpoints.ndim!=1 or not np.issubdtype(endpoints.dtype,np.integer) or np.any(endpoints<16) or np.any(endpoints>len(trajectory)) or np.any(np.diff(endpoints)<0):
        raise ValueError("janelas fora da trajetória")
    return np.stack([trajectory[int(end)-16:int(end)] for end in endpoints])


def encoded_windows(trajectory,endpoints,mode):
    trajectory=validate_frames(trajectory)
    # Validate without materializing all full windows.
    endpoints=np.asarray(endpoints)
    if endpoints.ndim!=1 or not np.issubdtype(endpoints.dtype,np.integer) or np.any(endpoints<16) or np.any(endpoints>len(trajectory)) or np.any(np.diff(endpoints)<0):
        raise ValueError("janelas fora da trajetória")
    if mode=="coarse":
        encoded=coarse(trajectory)
    elif mode=="moments":
        encoded=moments(trajectory)
    else:
        raise ValueError("codec desconhecido")
    return np.stack([encoded[int(end)-16:int(end)] for end in endpoints])
