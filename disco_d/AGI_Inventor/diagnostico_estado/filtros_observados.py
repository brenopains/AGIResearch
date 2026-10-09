"""Funções locais geradas de inovações visuais, com corpos inspecionáveis.

PCA de patches de diferenças temporais. Operação com precedentes conhecidos,
não invenção irrestrita nem AGI. Nunca consulta ação, recompensa ou nome do jogo.
"""
import json
from pathlib import Path
import numpy as np
from sensor_resolucao import validate_frames


def patches(frames):
    frames=validate_frames(frames)
    return frames.reshape(len(frames),21,10,16,10).transpose(0,1,3,2,4).reshape(len(frames),336,100).astype(np.float32)/255.


class ObservedFilters:
    def __init__(self,weights,mode):
        self.weights=np.asarray(weights,float)
        if self.weights.shape!=(5,100) or not np.isfinite(self.weights).all() or not np.allclose(self.weights.sum(1),0,atol=1e-10):
            raise ValueError("cinco funções locais de contraste são necessárias")
        self.mode=mode

    @classmethod
    def fit(cls,innovations):
        innovations=np.asarray(innovations,float)
        if innovations.ndim!=2 or innovations.shape[1]!=100 or len(innovations)<5 or not np.isfinite(innovations).all():
            raise ValueError("experiência de patches inválida")
        x=innovations-innovations.mean(1,keepdims=True)
        covariance=x.T@x/len(x)
        eigenvalues,eigenvectors=np.linalg.eigh(covariance)
        weights=eigenvectors[:,-5:].T[::-1].copy()
        weights-=weights.mean(1,keepdims=True)
        for w in weights:
            if w[np.argmax(np.abs(w))]<0:
                w*=-1
        weights/=np.abs(weights).sum(1,keepdims=True)
        body=cls(weights,"observed_temporal_innovation_pca")
        body.eigenvalues=eigenvalues[-5:][::-1]
        body.sample_count=len(x)
        return body

    @classmethod
    def random(cls):
        weights=np.random.default_rng(19771).normal(size=(5,100))
        weights-=weights.mean(1,keepdims=True)
        weights/=np.abs(weights).sum(1,keepdims=True)
        return cls(weights,"random_zero_mean_local_filters")

    def encode(self,frames):
        x=patches(frames)
        return np.concatenate((x.mean(2)[:,None,:],(x@self.weights.T).transpose(0,2,1)),axis=1).astype(np.float32)

    def save(self,path,origin):
        path=Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("funções existentes; preservar")
        np.savez_compressed(path,weights=self.weights,mode=self.mode,eigenvalues=getattr(self,"eigenvalues",np.empty(0)),sample_count=getattr(self,"sample_count",0))
        with path.with_suffix(".json").open("x",encoding="utf-8") as stream:
            json.dump({"body":"patch_response=sum(gray_patch[pixel]*weights[filter,pixel]);prepend_patch_mean",
                "mode":self.mode,"parameters":path.name,"primitive_patch_shape":[10,10],"origin":origin,
                "no_action_reward_or_world_name_used_to_fit_filters":True,"not_unrestricted_representation_invention":True},stream,indent=2,allow_nan=False)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            body=cls(data["weights"],str(data["mode"]))
            body.eigenvalues=data["eigenvalues"].copy()
            body.sample_count=int(data["sample_count"])
        return body
