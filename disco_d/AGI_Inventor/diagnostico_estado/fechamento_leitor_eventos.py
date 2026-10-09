"""Testar o fechamento algébrico do leitor de transição no transporte congelado.

Com X=phi(antes), Z=phi(depois), ridge conjunta dá
X.T@y=(X.T@X+alpha*I)@beta+X.T@Z@gamma.
Logo r_antigo=beta+K@gamma, K=ridge(X,Z), exatamente na aritmética ideal.
Reconhecimento melhor não cria previsão melhor usando o mesmo projetor linear.
Esta é uma identidade dessa classe/objetivo; não um teorema sobre toda inteligência.
"""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
import json
import numpy as np
from piloto_leitor_eventos import (HERE, STEM, BODIES, RESULT, BASES, GeneratedTransport, EventReader,
    load_bank, load_fresh, transitions, view, verify_frozen, require, save_new, sha)
from auditar_funcoes_contraexemplos import forward
from auditar_semantica_prefixo import scores

MANIFEST = STEM.with_name(STEM.name + "_fechamento_manifesto.json")
OUTPUT = STEM.with_name(STEM.name + "_fechamento.json")


def prepare():
    verify_frozen()
    require(not MANIFEST.exists() and not OUTPUT.exists(), "fechamento existe; preservar")
    sources = ("fechamento_leitor_eventos.py", "auditar_funcoes_contraexemplos.py", BODIES.relative_to(HERE).as_posix(),
        RESULT.relative_to(HERE).as_posix(), "resultados/eventos_auditoria.json")
    save_new(MANIFEST,{"status":"posthoc_algebraic_check_frozen","source_hashes":{p:sha(HERE/p) for p in sources},
        "claim_scope":"same_features_same_training_same_alpha_linear_ridge_readout_and_transport",
        "fit_calls":0,"ale_calls":0,"does_not_change_capability_gate":True})
    return {"status":"event_readout_closure_check_frozen","sha256":sha(MANIFEST)}


def run():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        require(sha(HERE/path) == expected,"fonte mudou")
    require(not OUTPUT.exists(),"verificação existe; preservar")
    frozen, reports, checked = verify_frozen(), {}, 0
    maximum_reward, maximum_prediction, maximum_normal = 0.,0.,0.
    for game in ("breakout","pong"):
        training = transitions(load_bank(game,149))
        fresh = load_fresh(game)
        observed = transitions(fresh)
        reports[game] = {}
        for basis in BASES:
            files = frozen["bodies"][game][basis]
            body = GeneratedTransport.load(HERE/files["feature"])
            reader = EventReader.load(HERE/files["readers"]["transition"])
            require(reader.alpha == body.alpha,"regularização diverge; identidade não se aplica")
            with np.load(HERE/files["planner"],allow_pickle=False) as params:
                k = body.transport+params["terminal_transport"]
            d = len(body.initial_prior)
            beta,gamma = reader.coefficient[:,:d],reader.coefficient[:,d:]
            transported_reward = beta+np.einsum("aij,aj->ai",k,gamma)
            normal_error = []
            p,pn = body.features(training["source"]),body.features(training["successor"])
            for action in range(body.actions):
                ids=training["actions"]==action
                xx,zz,yy=p[ids],pn[ids],training["reward"][ids]
                residual=(xx.T@xx+body.alpha*np.eye(d))@beta[action]+xx.T@zz@gamma[action]-xx.T@yy
                normal_error.append(float(np.max(np.abs(residual))))
            local_error=float(np.max(np.abs(transported_reward-body.reward)))
            maximum_reward=max(maximum_reward,local_error)
            maximum_normal=max(maximum_normal,max(normal_error))
            require(local_error<1e-8 and max(normal_error)<1e-8,"identidade não reproduz; investigar instrumento")
            rewards=np.zeros_like(fresh["rewards"],float)
            predictions=reader.predict(body.features(observed["source"]),body.features(observed["successor"]),observed["actions"])
            for index,prediction in zip(observed["bank_indices"],predictions):
                rewards[tuple(index)]=prediction
            partitions={}
            for partition,offset in (("known",0),("new",4)):
                horizons={}
                for horizon in (4,8,16):
                    cases=view(fresh,offset,horizon)
                    phi=body.features(cases["history"])
                    old=body.comparison(cases["history"],cases["left"],cases["right"])
                    transported=forward(phi,transported_reward,body.transport,cases["left"],cases["right"])
                    retrospective=np.array([sum(rewards[i,a,:horizon])-sum(rewards[i,b,:horizon]) for i,a,b in cases["bank_cases"]])
                    error=float(np.max(np.abs(old-transported)))
                    maximum_prediction=max(maximum_prediction,error)
                    require(error<1e-8,"programas não fecham no preditor antigo")
                    horizons[str(horizon)]={"old_before_decision":scores(cases["actual"],old),
                        "recognition_transported_before_decision":scores(cases["actual"],transported),
                        "recognition_actual_future_diagnostic_only":scores(cases["actual"],retrospective),
                        "maximum_prediction_difference":error,
                        "decision_changes":int(np.count_nonzero((old>=-1e-12)!=(transported>=-1e-12)))}
                    checked+=len(old)
                partitions[partition]=horizons
            reports[game][basis]={"maximum_reward_coefficient_difference":local_error,
                "normal_equation_residual_per_action":normal_error,"partitions":partitions}
    output={"status":"linear_event_readout_closes_into_old_predictor","games":reports,"program_comparisons_checked":checked,
        "maximum_reward_coefficient_difference":maximum_reward,"maximum_normal_equation_residual":maximum_normal,
        "maximum_program_prediction_difference":maximum_prediction,"fit_calls":0,"ale_calls":0,
        "manifest_sha256":sha(MANIFEST),"human_level_goal_achieved":False,"claim_scope":manifest["claim_scope"],
        "exact_symbolic_identity_numeric_instances_checked_not_exact_numeric_proof":True}
    save_new(OUTPUT,output)
    return {k:v for k,v in output.items() if k!="games"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    print(json.dumps({"prepare":prepare,"run":run}[parser.parse_args().mode](),ensure_ascii=False,indent=2))
