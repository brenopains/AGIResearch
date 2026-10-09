"""Copiar V1 e corrigir somente o contrato de restauração, preservando evidências."""

import json

from experiencia_cruzada import save_json
from piloto_planos_propostos import HERE, sha


def main():
    names = ("experiencia_cruzada.py", "coletar_experiencia_cruzada.py", "avaliar_experiencia_cruzada.py",
             "piloto_experiencia_cruzada.py", "auditar_experiencia_cruzada.py", "testes_experiencia_cruzada.py")
    index_path = HERE / "resultados" / "experiencia_cruzada_v2_copias.json"
    if index_path.exists():
        raise RuntimeError("cópiasV2 já existem; preservar")
    originals, copies = {}, {}
    for name in names:
        source = HERE / name
        target = HERE / name.replace("experiencia_cruzada", "experiencia_cruzada_v2")
        if target.exists():
            raise RuntimeError("alvoV2 existe; preservar")
        text = source.read_text(encoding="utf-8").replace("experiencia_cruzada", "experiencia_cruzada_v2")
        if name == "experiencia_cruzada.py":
            anchor = '"coleta.py", "coletar_transicoes_mistas.py")'
            replacement = ('"coleta.py", "coletar_transicoes_mistas.py", "CORRECAO_CRUZADA_V2.md", '
                           '"preparar_cruzada_v2.py", "calibrar_restauracao_ale.py", '
                           '"resultados/experiencia_cruzada_v2_copias.json", '
                           '"resultados/experiencia_cruzada_propostas.json", "resultados/experiencia_cruzada_recursos.json", '
                           '"resultados/breakout_experiencia_cruzada_131_interrupcao.json", '
                           '"resultados/calibracao_restauracao_ale.json")')
            if text.count(anchor) != 1:
                raise RuntimeError("âncora de fontes divergiu")
            text = text.replace(anchor, replacement)
        if name == "coletar_experiencia_cruzada.py":
            anchor = 'require_frame = digest(gray(ale)) == digest(history[-1])'
            if text.count(anchor) != 1:
                raise RuntimeError("âncora do contrato divergiu")
            text = text.replace(anchor, 'require_frame = bool(ale.cloneSystemState().equals(state))')
            text = text.replace('"restauração da raiz visual divergiu"', '"estado clonado não foi restaurado"')
            anchor = 'actions = list(ale.getMinimalActionSet()[:4])'
            text = text.replace(anchor, 'prior_path = HERE / "resultados" / f"{game}_experiencia_cruzada_{seed}_interrupcao.json"\n'
                                '    prior_calls = json.loads(prior_path.read_text(encoding="utf-8"))["ale_calls"] if prior_path.exists() else 0\n'
                                '    if not 0 <= prior_calls < 50000:\n'
                                '        raise RuntimeError("custo anterior inválido")\n'
                                '    budget = StepBudget(50000 - prior_calls)\n'
                                '    ' + anchor)
            anchor = '"walk_calls": walks, "proposal_sha256": sha(PROPOSAL)'
            text = text.replace(anchor, '"walk_calls": walks, "previous_attempt_ale_calls": prior_calls,\n'
                                '                  "ale_calls_including_previous_attempt": budget.used + prior_calls, "proposal_sha256": sha(PROPOSAL)')
        if name == "auditar_experiencia_cruzada.py":
            anchor = 'calls = sum(e["ale_calls"] for e in events)'
            text = text.replace(anchor, anchor + '\n    require(report["ale_calls_including_previous_attempt"] == calls + report["previous_attempt_ale_calls"] <= 50000, "teto incluindo tentativaV1 excedido")')
        target.write_text(text, encoding="utf-8")
        originals[name], copies[target.name] = sha(source), sha(target)
    save_json(index_path, {"status": "instrument_only_versioned_copy", "original_sha256": originals, "copy_sha256": copies,
                           "correction": "state_equality_instead_of_cached_framebuffer", "seeds_models_gates_unchanged": True,
                           "failed_attempt_preserved": True, "reserved137_observed_before_copy": False,
                           "calibration_sha256": sha(HERE / "resultados" / "calibracao_restauracao_ale.json")})
    print(json.dumps({"status": "six_versioned_sources_created", "manifest": str(index_path)}))


if __name__ == "__main__":
    main()
