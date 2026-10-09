"""
lab/body_map.py — CHECADOR MULTI-SISTEMA: mapeia cada hipotese (do gauntlet) -> sistemas do corpo afetados,
orgao-sitio de acao, e RISCOS de interacao multi-sistema. Versao bounded do "corpo humano mapeado" do usuario.
Saida: body_result.json (consumido pelo frontend holografico Stark).
"""
import os, json, re, time
os.environ.setdefault('LLM_MODEL','qwen3-235b-a22b-instruct-2507')
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__),'..','.env'))
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from shared.llm_client import _openai_compat_post, VENICE_URL, VENICE_KEY
M="qwen3-235b-a22b-instruct-2507"
SYSTEMS=["nervoso","muscular","imune","cardiovascular","respiratorio","digestivo","hepatico","renal","endocrino","esqueletico"]
def llm(sysp,usr,mx=400):
    for _ in range(3):
        try: return _openai_compat_post(VENICE_URL,VENICE_KEY,M,[{"role":"system","content":sysp},{"role":"user","content":usr}],0.3,mx,is_venice=True)
        except: time.sleep(2)
    return ""
def enrich(disease, items):
    out=[]
    for it in items:
        alvo=it.get('alvo','?'); droga=it.get('droga','?'); verdict=it.get('verdict','')
        r=llm(f"Voce e farmacologista. Sistemas validos: {', '.join(SYSTEMS)}. Responda SO JSON.",
              f"Doenca: {disease}\nAlvo: {alvo}\nDroga/classe: {droga}\n\nJSON com: "
              f'{{"alvo_sistema":"sistema primario onde o alvo age","orgao":"orgao-sitio","sistemas_afetados":["lista de 1-3 sistemas"],'
              f'"riscos":["1-2 riscos/efeitos adversos multi-sistema concretos desta droga/classe"]}}')
        try: m=json.loads(re.search(r'\{.*\}', r, re.S).group(0))
        except: m={"alvo_sistema":"nervoso","orgao":"?","sistemas_afetados":["nervoso"],"riscos":[]}
        # status p/ cor: verde=validada, dourado=nova-ancorada, vermelho=esgotada/falhou, cinza=sem ancoragem
        v=verdict.lower()
        if 'sem ancoragem' in v: status='cinza'; label='SEM EVIDENCIA'
        elif 'esgotada' in v or 'falhou' in v: status='vermelho'; label='FALHOU/ESGOTADA'
        elif 'nova' in v and 'ancorada' in v: status='dourado'; label='NOVA PROMISSORA'
        elif 'validada' in v: status='verde'; label='VALIDADA ATIVA'
        else: status='azul'; label='ANCORADA'
        out.append({"alvo":alvo,"droga":droga,"verdict":verdict,"status":status,"label":label,
                    "pubmed":it.get('pubmed',0),"trials":it.get('trials',0),"fase_max":it.get('fase_max',0),
                    "score":it.get('score',0),"alvo_sistema":m.get('alvo_sistema','nervoso'),"orgao":m.get('orgao','?'),
                    "sistemas_afetados":m.get('sistemas_afetados',['nervoso'])[:3],"riscos":m.get('riscos',[])[:2]})
        print(f"[{alvo}] {label} -> sistema={m.get('alvo_sistema')} orgao={m.get('orgao')} afeta={m.get('sistemas_afetados')} riscos={m.get('riscos')}")
    return out
if __name__=="__main__":
    disease="esclerose lateral amiotrofica (ELA)"
    items=json.load(open(os.path.join(os.path.dirname(__file__),'gauntlet_result.json')))
    enriched=enrich(disease, items)
    payload={"disease":disease,"systems":SYSTEMS,"hypotheses":enriched,"generated":"in-silico"}
    json.dump(payload, open(os.path.join(os.path.dirname(__file__),'body_result.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\n[salvo lab/body_result.json] {len(enriched)} hipoteses mapeadas no corpo")
