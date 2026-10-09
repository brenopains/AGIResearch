"""
Cliente LLM unificado: Venice (GLM-4.7-Flash barato) -> Nemotron local (fallback).
Mantem a API ask(messages, system, ...) que todos os agentes ja usam.
"""
import os, json, time, hashlib, pathlib, requests
from typing import Optional, List, Dict

# ---- Config ----
VENICE_URL  = os.environ.get("VENICE_URL", "https://api.venice.ai/api/v1")
VENICE_KEY  = os.environ.get("VENICE_API_KEY") or os.environ.get("VENICE_ADMIN_KEY")
NEMOTRON_URL= os.environ.get("NEMOTRON_URL", "http://localhost:1234/v1")

# Default: GLM-4.7-Flash ($0.125/$0.50 por MTok — ~37x mais barato que Sonnet)
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "zai-org-glm-4.7-flash")
# "Smart" upgrade quando precisa raciocinio mais forte (~7x mais barato que Sonnet ainda)
SMART_MODEL   = os.environ.get("LLM_SMART_MODEL", "zai-org-glm-4.7")
DEFAULT_HAIKU = DEFAULT_MODEL  # compat com nome antigo (utilitarios)

# ---- Cache de disco p/ temperatura 0 ----
_CACHE_DIR = pathlib.Path(__file__).parent.parent / "runs" / "_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _ckey(model, system, messages, temperature, max_tokens):
    h = hashlib.sha256(json.dumps({"m":model,"s":system,"u":messages,"t":temperature,"x":max_tokens}, sort_keys=True).encode()).hexdigest()[:32]
    return _CACHE_DIR / f"{h}.json"

# throttle global p/ Venice: gap minimo entre requisicoes (rate limit ~ poucas req/s)
_VENICE_MIN_GAP = float(os.environ.get("VENICE_MIN_GAP", "1.2"))
_venice_last = [0.0]
def _throttle_venice():
    dt = time.time() - _venice_last[0]
    if dt < _VENICE_MIN_GAP:
        time.sleep(_VENICE_MIN_GAP - dt)
    _venice_last[0] = time.time()

# ---- Backend implementations ----
def _openai_compat_post(base_url: str, api_key: Optional[str], model: str, messages: List[Dict],
                       temperature: float, max_tokens: int, timeout: int = 180, is_venice: bool = False) -> str:
    headers = {"Content-Type":"application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if is_venice:
        # Por padrao desabilita thinking (rapido/barato). VENICE_THINKING=1 LIGA o raciocinio
        # (essencial p/ modelos de reasoning tipo deepseek-v4-pro em tarefas dificeis como ARC).
        if os.environ.get("VENICE_THINKING") == "1":
            body["venice_parameters"] = {"disable_thinking": False, "strip_thinking_response": True}
        else:
            body["venice_parameters"] = {"disable_thinking": True, "strip_thinking_response": True}
    r = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    txt = msg.get("content") or ""
    # se vier vazio mas houver reasoning_content (thinking nao foi desabilitado), aproveita
    if not txt and msg.get("reasoning_content"):
        txt = msg["reasoning_content"]
    return txt

# Hook opcional p/ telemetria/UI: ('venice_ok'|'venice_fail'|'nemotron_ok'|'nemotron_fail', model, tokens_estimate, err)
_TELEMETRY_CB = None
def set_telemetry(cb):
    global _TELEMETRY_CB; _TELEMETRY_CB = cb

def _telemetry(event, **kwargs):
    if _TELEMETRY_CB:
        try: _TELEMETRY_CB(event, kwargs)
        except Exception: pass

# ---- API publica ----
def ask(messages: List[Dict], system: Optional[str]=None,
        model: Optional[str]=None, temperature: float=0.7, max_tokens: int=2048,
        cache: bool=False, use_prompt_cache: bool=True) -> str:
    """messages: [{role,content}]; system: str. Tenta Venice (model padrao), fallback Nemotron local."""
    mdl = model or DEFAULT_MODEL

    # monta payload com system como primeira mensagem (OpenAI-compatible)
    full_msgs = []
    if system: full_msgs.append({"role":"system","content":system})
    full_msgs.extend(messages)

    if cache and temperature == 0:
        p = _ckey(mdl, system, messages, temperature, max_tokens)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["text"]

    # ---- 1. tenta Venice (primario) — com backoff agressivo p/ 429 (rate limit) ----
    if VENICE_KEY:
        _throttle_venice()   # gap minimo entre chamadas (evita 429)
        for attempt in range(5):
            try:
                text = _openai_compat_post(VENICE_URL, VENICE_KEY, mdl, full_msgs, temperature, max_tokens, is_venice=True)
                _telemetry("venice_ok", model=mdl, attempt=attempt+1)
                if cache and temperature == 0:
                    p.write_text(json.dumps({"text": text}), encoding="utf-8")
                return text
            except Exception as e:
                is_429 = "429" in str(e) or "Too Many" in str(e)
                _telemetry("venice_fail", model=mdl, attempt=attempt+1, err=str(e)[:200])
                if attempt < 4:
                    # 429: espera bem mais (3,6,12,24s); outros erros: backoff curto
                    wait = (3.0 * (2**attempt)) if is_429 else (1.5**attempt)
                    time.sleep(wait)
                else:
                    print(f"[llm] Venice falhou {attempt+1}x: {e}; caindo p/ Nemotron local...")
    else:
        print("[llm] VENICE_API_KEY nao definida; usando Nemotron local direto.")

    # ---- 2. fallback Nemotron local (LM Studio) ----
    try:
        # LM Studio tem o modelo carregado; o ID e diferente
        nemotron_model = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-4b")
        text = _openai_compat_post(NEMOTRON_URL, None, nemotron_model, full_msgs, temperature, max_tokens, timeout=240)
        _telemetry("nemotron_ok", model=nemotron_model)
        if cache and temperature == 0:
            p.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text
    except Exception as e:
        _telemetry("nemotron_fail", err=str(e)[:200])
        raise RuntimeError(f"Venice e Nemotron ambos falharam. Verifique VENICE_API_KEY e LM Studio. ultimo erro: {e}")

def ask_smart(messages, system=None, temperature=0.5, max_tokens=2048, cache=False):
    """Versao p/ tarefas que precisam mais raciocinio — usa GLM-4.7 full (~7x mais barato que Sonnet)."""
    return ask(messages, system=system, model=SMART_MODEL, temperature=temperature, max_tokens=max_tokens, cache=cache)
