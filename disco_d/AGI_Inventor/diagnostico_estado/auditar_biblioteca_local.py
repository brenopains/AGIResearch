"""Conferir origem e execução de filtros reais, sem veredito de competência."""
import os
import platform
import sys
if sys.platform=="win32":
    platform.machine=platform._get_machine_win32
    platform.system=lambda:"Windows"
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="2"
import argparse
import json
import numpy as np
from piloto_resolucao import HERE,STEM,PROPOSAL,ObservedFilters,innovation_samples,root_chunk,verify,require,save_new,sha

MANIFEST=STEM.with_name(STEM.name+"_biblioteca_manifesto.json")
OUTPUT=STEM.with_name(STEM.name+"_biblioteca_auditoria.json")


def prepare():
    verify()
    require(not MANIFEST.exists() and not OUTPUT.exists(),"biblioteca auditada antes; preservar")
    paths=[HERE/"auditar_biblioteca_local.py",PROPOSAL]
    for game in ("breakout","pong"):
        for kind in ("learned","random"):
            path=STEM.with_name(STEM.name+f"_{game}_{kind}_filters.npz")
            paths.extend((path,path.with_suffix(".json")))
    save_new(MANIFEST,{"status":"local_library_frozen_before_origin_audit","source_hashes":{p.relative_to(HERE).as_posix():sha(p) for p in paths},
        "scope":"function_origins_and_execution_not_useful_transfer","physical_calls":0,"fit_calls":0})
    return {"status":"local_library_audit_frozen","sha256":sha(MANIFEST)}


def run():
    verify()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path,expected in manifest["source_hashes"].items():
        require(sha(HERE/path)==expected,"função/fonte mudou")
    require(not OUTPUT.exists(),"auditoria existente")
    libraries,max_eigen,max_value={},0.,0.
    for game in ("breakout","pong"):
        samples=innovation_samples(game).astype(float)
        samples-=samples.mean(1,keepdims=True)
        covariance=samples.T@samples/len(samples)
        eigenvalues=np.linalg.eigvalsh(covariance)[-5:][::-1]
        body=ObservedFilters.load(STEM.with_name(STEM.name+f"_{game}_learned_filters.npz"))
        require(body.sample_count==8192 and np.allclose(body.eigenvalues,eigenvalues,atol=1e-10),"autovalores não derivam da experiência")
        for weight,value in zip(body.weights,body.eigenvalues):
            error=float(np.max(np.abs(covariance@weight-value*weight)))
            max_eigen=max(max_eigen,error)
            require(error<1e-8 and abs(weight.sum())<1e-10 and abs(np.abs(weight).sum()-1)<1e-10,"corpo local inválido")
        applications={}
        for target in ("breakout","pong"):
            with np.load(root_chunk(target,149,0),allow_pickle=False) as data:
                frames=data["frames_0"][:16].copy()
            blocks=frames.astype(float).reshape(16,21,10,16,10)/255.
            independent=np.einsum("nyixj,kij->nkyx",blocks,body.weights.reshape(5,10,10)).reshape(16,5,336)
            actual=body.encode(frames)[:,1:]
            error=float(np.max(np.abs(actual-independent)))
            max_value=max(max_value,error)
            require(error<1e-6,"função não corresponde aos coeficientes publicados")
            applications[target]={"frames":16,"maximum_response_error":error,"source_weights_changed":False}
        libraries[game]={"filters":5,"samples":8192,"eigenvalues":body.eigenvalues.tolist(),"applications":applications}
    output={"status":"local_function_library_origin_audit_passed","libraries":libraries,
        "maximum_eigen_equation_residual":max_eigen,"maximum_independent_function_response_error":max_value,
        "physical_calls":0,"fit_calls":0,"reuse_without_rewriting_bodies_verified":True,
        "useful_cross_world_transfer_not_demonstrated":True,"human_level_goal_achieved":False,"manifest_sha256":sha(MANIFEST)}
    save_new(OUTPUT,output)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
