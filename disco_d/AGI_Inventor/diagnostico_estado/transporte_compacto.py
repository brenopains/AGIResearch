"""Mesmo dicionário numérico, com chaves binárias ordenadas e distâncias em cache."""
import time
import numpy as np
from transporte_funcoes import FunctionTransport,distances


def unique_numeric_rows(values):
    values=np.asarray(values)
    if values.ndim!=2 or len(values)==0 or not np.issubdtype(values.dtype,np.floating) or values.dtype.itemsize>8 or not np.isfinite(values).all():
        raise ValueError("matriz binária finita de ponto flutuante requerida")
    # Normalize signed zeros for numeric equality. All float16/32 values are
    # represented exactly in binary64. Preserve originals in the returned rows.
    canonical=np.asarray(values,dtype=np.float64).copy(order="C")
    canonical[canonical==0]=0.
    bits=canonical.view(np.uint64)
    sign=np.uint64(1<<63)
    negative=(bits&sign)!=0
    bits^=np.where(negative,np.uint64(0xffffffffffffffff),sign)
    if np.little_endian:
        bits.byteswap(inplace=True)
    records=bits.view(np.dtype((np.void,8*values.shape[1]))).reshape(-1)
    _,index=np.unique(records,return_index=True)
    return values[index].copy()


class CompactTransport(FunctionTransport):
    def fit(self,source,successor,actions,reward,continuation,shuffled=False):
        started=time.perf_counter()
        source,successor=np.asarray(source,float),np.asarray(successor,float)
        actions,reward,continuation=np.asarray(actions),np.asarray(reward,float),np.asarray(continuation,float)
        n=len(source)
        if (source.shape!=successor.shape or source.ndim<2 or n==0 or actions.shape!=reward.shape or reward.shape!=continuation.shape
            or reward.shape!=(n,) or not all(np.isfinite(v).all() for v in (source,successor,reward,continuation))
            or not np.issubdtype(actions.dtype,np.integer) or np.any(actions<0) or np.any(actions>=self.actions)
            or not np.isin(continuation,[0,1]).all()):
            raise ValueError("transições inválidas")
        begin=time.perf_counter()
        raw=np.concatenate((source.reshape(n,-1),successor.reshape(n,-1)))
        unique=unique_numeric_rows(raw)
        del raw
        unique_time=time.perf_counter()-begin
        begin=time.perf_counter()
        norms=(unique*unique).sum(1)[:,None]
        doubled=2*unique
        def distance_to(points):
            return np.maximum(norms+(points*points).sum(1)[None,:]-doubled@points.T,0)
        middle=unique.mean(0,keepdims=True)
        ids=[int(distance_to(middle).argmin())]
        nearest=distance_to(unique[ids]).ravel()
        for _ in range(1,min(self.supports_limit,len(unique))):
            candidate=int(nearest.argmax())
            if nearest[candidate]<=1e-14:
                break
            ids.append(candidate)
            nearest=np.minimum(nearest,distance_to(unique[candidate:candidate+1]).ravel())
        self.support=unique[ids].copy()
        del unique,norms,doubled
        dd=distances(self.support,self.support)
        positive=dd[np.triu_indices(len(dd),1)]
        positive=positive[positive>1e-14]
        self.scale=float(np.median(positive)) if len(positive) else 1.
        dictionary_time=time.perf_counter()-begin
        begin=time.perf_counter()
        p,pn=self.features(source),self.features(successor)
        projection_time=time.perf_counter()-begin
        self.initial_prior=p.mean(0)
        self.reward,self.transport,self.action_counts,self.local_residual_mse=[],[],[],[]
        rng=np.random.default_rng(9861)
        begin=time.perf_counter()
        for action in range(self.actions):
            ids=np.flatnonzero(actions==action)
            if not len(ids):
                raise ValueError("ação sem experiência")
            x=p[ids]
            y=np.column_stack((reward[ids],continuation[ids,None]*pn[ids]))
            if shuffled:
                y=y[rng.permutation(len(ids))]
            coefficient=np.linalg.solve(x.T@x+self.alpha*np.eye(x.shape[1]),x.T@y)
            self.reward.append(coefficient[:,0]);self.transport.append(coefficient[:,1:])
            self.action_counts.append(len(ids))
            self.local_residual_mse.append(float(np.mean((x@coefficient-y)**2)))
        self.reward,self.transport=np.asarray(self.reward),np.asarray(self.transport)
        self.mode="transport"
        self.fit_phi,self.fit_next_phi=p,pn
        self.fit_timing={"unique_seconds":unique_time,"dictionary_seconds":dictionary_time,
            "projection_seconds":projection_time,"solve_seconds":time.perf_counter()-begin,
            "total_seconds":time.perf_counter()-started}
        return self
