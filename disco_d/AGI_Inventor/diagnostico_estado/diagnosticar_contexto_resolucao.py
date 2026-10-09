"""Diagnóstico descritivo de decisões: contexto visual ou preferência por programa?

Não ajusta funções e não transforma respostas reservadas em políticas do agente.
O melhor sinal constante é um oráculo retrospectivo no conjunto finito.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
STEM=HERE/"resultados"/"resolucao_compacta_contexto"
MANIFEST=STEM.with_name(STEM.name+"_manifesto.json")
OUTPUT=STEM.with_suffix(".json")
EPS=1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def sign(value):
    require(math.isfinite(value),"previsão não finita")
    return 0 if abs(value)<=EPS else 1 if value>0 else -1


def cost(actual,predicted):
    require(math.isfinite(actual),"resposta não finita")
    choice=sign(predicted)
    return abs(actual)/2 if choice==0 else max(-choice*actual,0.)


def summarize(arms):
    require(arms and "no_context" in arms,"ablação de contexto ausente")
    indexed={}
    for arm,rows in arms.items():
        mapping={(r["bank_row"],r["left"],r["right"]):r for r in rows}
        require(len(mapping)==len(rows)>0,"caso ausente/duplicado")
        indexed[arm]=mapping
    keys=sorted(indexed["no_context"])
    for mapping in indexed.values():
        require(sorted(mapping)==keys,"identidades incompatíveis")
        require(all(mapping[k]["actual_delta"]==indexed["no_context"][k]["actual_delta"] for k in keys),"resposta difere entre braços")
    groups={}
    for key in keys:
        groups.setdefault(key[1:],[]).append(key)
    constant_cost=0.
    constant=[]
    for pair,ids in groups.items():
        actual=[indexed["no_context"][k]["actual_delta"] for k in ids]
        left=sum(cost(d,1.) for d in actual)
        right=sum(cost(d,-1.) for d in actual)
        constant_cost+=min(left,right)
        constant.append({"program_pair":list(pair),"count":len(ids),"optimal_constant_sign_retrospective":1 if left<right else -1 if right<left else 0,
            "fair_regret_of_best_constant":min(left,right)/len(ids),
            "actual_positive":sum(d>EPS for d in actual),"actual_negative":sum(d<-EPS for d in actual)})
    floor=constant_cost/len(keys)
    base=indexed["no_context"]
    base_cost=sum(cost(base[k]["actual_delta"],base[k]["predicted_delta"]) for k in keys)/len(keys)
    results={}
    for arm,mapping in indexed.items():
        arm_cost=sum(cost(mapping[k]["actual_delta"],mapping[k]["predicted_delta"]) for k in keys)/len(keys)
        patterns=[]
        for pair,ids in groups.items():
            signs=[sign(mapping[k]["predicted_delta"]) for k in ids]
            patterns.append({"program_pair":list(pair),"positive":signs.count(1),"negative":signs.count(-1),"ties":signs.count(0),
                "choice_varies_with_context":len(set(signs))>1})
        flips=[k for k in keys if sign(mapping[k]["predicted_delta"])!=sign(base[k]["predicted_delta"])]
        beneficial=sum(cost(mapping[k]["actual_delta"],mapping[k]["predicted_delta"])<cost(base[k]["actual_delta"],base[k]["predicted_delta"])-EPS for k in flips)
        harmful=sum(cost(mapping[k]["actual_delta"],mapping[k]["predicted_delta"])>cost(base[k]["actual_delta"],base[k]["predicted_delta"])+EPS for k in flips)
        results[arm]={"fair_tie_regret":arm_cost,"gain_over_same_body_without_context":base_cost-arm_cost,
            "gain_over_retrospectively_best_constant_per_pair":floor-arm_cost,"choice_patterns":patterns,
            "changed_choices_vs_no_context":len(flips),"beneficial_changed_choices":beneficial,"harmful_changed_choices":harmful,
            "neutral_changed_choices":len(flips)-beneficial-harmful}
    return {"count":len(keys),"retrospective_constant_per_pair_floor":floor,"constant_oracle_pairs":constant,"arms":results}


def prepare():
    from piloto_resolucao_compacto import verify_bodies
    verify_bodies()
    require(not MANIFEST.exists() and not OUTPUT.exists(),"diagnóstico existente; preservar")
    paths=["diagnosticar_contexto_resolucao.py","testes_contexto_resolucao.py","CONTEXTO_RESOLUCAO.md",
        "resultados/resolucao_compacta.json","resultados/resolucao_compacta.jsonl","resultados/resolucao_compacta_auditoria.json",
        "resultados/resolucao_compacta_proposta.json","resultados/resolucao_compacta_corpos_manifesto.json"]
    value={"status":"post_evaluation_descriptive_diagnostic_frozen","fit_calls":0,"ale_calls":0,"maximum_wall_seconds":30,
        "constant_oracle_must_not_be_deployed":True,"source_hashes":{p:sha(HERE/p) for p in paths}}
    with MANIFEST.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
    return {"status":value["status"],"manifest_sha256":sha(MANIFEST)}


def run():
    started=time.perf_counter()
    value=json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(not OUTPUT.exists(),"saída existente; preservar")
    for name,expected in value["source_hashes"].items():
        require(sha(HERE/name)==expected,"fonte mudou: "+name)
    prior=json.loads((HERE/"resultados/resolucao_compacta.json").read_text(encoding="utf-8"))
    rows=[json.loads(line) for line in (HERE/"resultados/resolucao_compacta.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows)==prior["event_count"]==6912,"casos incompletos")
    result={}
    for game,cells in prior["games"].items():
        result[game]={}
        for partition,cell in cells.items():
            curves={}
            for horizon in (4,8,16):
                selected=[r for r in rows if (r["game"],r["partition"],r["horizon"])==(game,partition,horizon)]
                arms={arm:[r for r in selected if r["arm"]==arm] for arm in cell["curves"][str(horizon)]}
                require(all(len(v)==64 for v in arms.values()),"braço incompleto")
                summary=summarize(arms)
                for arm,r in summary["arms"].items():
                    require(abs(r["fair_tie_regret"]-cell["curves"][str(horizon)][arm]["fair_tie_regret"])<EPS,"métrica não reconcilia")
                curves[str(horizon)]=summary
            result[game][partition]=curves
    elapsed=time.perf_counter()-started
    require(elapsed<30,"teto do diagnóstico excedido")
    output={"status":"context_choice_diagnostic_complete_no_promotion","games":result,"previous_gate_preserved":prior["status"],
        "wall_seconds":elapsed,"fit_calls":0,"ale_calls":0,"comparisons_reconciled":len(rows),"manifest_sha256":sha(MANIFEST),
        "descriptive_not_independent_confirmation":True,"constant_oracle_not_a_learned_policy":True,"human_level_goal_achieved":False}
    with OUTPUT.open("x",encoding="utf-8") as stream:
        json.dump(output,stream,ensure_ascii=False,allow_nan=False,indent=2)
    return {"status":output["status"],"wall_seconds":elapsed,"primary16":{g:{p:r["16"] for p,r in cells.items()} for g,cells in result.items()}}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    mode=parser.parse_args().mode
    print(json.dumps(prepare() if mode=="prepare" else run(),ensure_ascii=False,indent=2))
