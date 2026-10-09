"""Deduplicação exata e ordenação lexicográfica de observações de alta dimensão.

Evita o dtype estruturado com milhares de campos de numpy.unique(axis=0).
Preserva a geometria, a ordem numérica e o desempate do dicionário original.
Hashes só indexam candidatos; colisões são comparadas por igualdade real.
"""
import hashlib
import numpy as np
from transporte_funcoes import FunctionTransport


def exact_unique_sorted(values):
    values=np.asarray(values)
    if values.ndim!=2 or len(values)==0 or not np.issubdtype(values.dtype,np.floating) or not np.isfinite(values).all():
        raise ValueError("matriz finita não vazia de ponto flutuante requerida")
    canonical=np.ascontiguousarray(values.copy())
    # IEEE signed zero is numerically equal; normalize solely for hashing.
    canonical[canonical==0]=0
    buckets,ids={},[]
    for i,row in enumerate(canonical):
        key=hashlib.sha256(row.tobytes()).digest()
        bucket=buckets.setdefault(key,[])
        if not any(np.array_equal(row,canonical[j]) for j in bucket):
            bucket.append(i)
            ids.append(i)
    unique=values[np.asarray(ids)].copy()
    groups=[np.arange(len(unique))]
    variable=np.flatnonzero(np.any(unique!=unique[:1],axis=0))
    for column in variable:
        next_groups=[]
        unresolved=False
        for group in groups:
            if len(group)==1:
                next_groups.append(group)
                continue
            order=group[np.argsort(unique[group,column],kind="stable")]
            split=np.flatnonzero(np.diff(unique[order,column])!=0)+1
            children=np.split(order,split)
            next_groups.extend(children)
            unresolved|=any(len(child)>1 for child in children)
        groups=next_groups
        if not unresolved:
            break
    return unique[np.concatenate(groups)]


class FastFunctionTransport(FunctionTransport):
    """Mesmo problema numérico; dicionário e distâncias sem trabalho redundante."""
    def fit(self,source,successor,actions,reward,continuation):
        source,successor=np.asarray(source,float),np.asarray(successor,float)
        actions,reward,continuation=np.asarray(actions),np.asarray(reward,float),np.asarray(continuation,float)
        n=len(source)
        if (source.shape!=successor.shape or source.ndim<2 or n==0 or actions.shape!=reward.shape or reward.shape!=continuation.shape
            or reward.shape!=(n,) or not all(np.isfinite(v).all() for v in (source,successor,reward,continuation))
            or not np.issubdtype(actions.dtype,np.integer) or np.any(actions<0) or np.any(actions>=self.actions) or not np.isin(continuation,[0,1]).all()):
            raise ValueError("transições inválidas")
        raw=np.concatenate((source.reshape(n,-1),successor.reshape(n,-1)))
        unique=exact_unique_sorted(raw)
        del raw
        norms=(unique*unique).sum(1)[:,None]
        def distance_to(points):
            return np.maximum(norms+(points*points).sum(1)[None,:]-2*unique@points.T,0)
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
        del unique,norms
        from transporte_funcoes import distances
        dd=distances(self.support,self.support)
        positive=dd[np.triu_indices(len(dd),1)]
        positive=positive[positive>1e-14]
        self.scale=float(np.median(positive)) if len(positive) else 1.
        phi,pn=self.features(source),self.features(successor)
        self.initial_prior=phi.mean(0)
        self.reward,self.transport,self.action_counts,self.local_residual_mse=[],[],[],[]
        for action in range(self.actions):
            ids=np.flatnonzero(actions==action)
            if not len(ids):
                raise ValueError("ação sem experiência")
            x=phi[ids]
            y=np.column_stack((reward[ids],continuation[ids,None]*pn[ids]))
            coefficient=np.linalg.solve(x.T@x+self.alpha*np.eye(x.shape[1]),x.T@y)
            self.reward.append(coefficient[:,0])
            self.transport.append(coefficient[:,1:])
            self.action_counts.append(len(ids))
            self.local_residual_mse.append(float(np.mean((x@coefficient-y)**2)))
        self.reward,self.transport=np.asarray(self.reward),np.asarray(self.transport)
        self.mode="transport"
        return self
