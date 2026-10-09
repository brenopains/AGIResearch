"""
Cliente Nemotron local via LM Studio (OpenAI-compatible).
Papel: utilitarios baratos paralelizaveis — NER, sumarizacao de abstract, rerank, compressao.
NAO usar p/ raciocinio de fronteira (use Claude).

Pre-requisito: LM Studio aberto, modelo NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf carregado,
"Start Server" ligado (default localhost:1234). Endpoint OpenAI-compatible.
"""
import os, json, requests, time

BASE_URL = os.environ.get("NEMOTRON_URL", "http://localhost:1234/v1")
MODEL    = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-nano-4b")  # ID exato do LM Studio (ver GET /v1/models)

def available():
    try:
        r = requests.get(BASE_URL + "/models", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def ask(messages, system=None, temperature=0.3, max_tokens=512):
    """Chat OpenAI-compatible. messages=[{'role','content'},...]."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)
    for attempt in range(3):
        try:
            r = requests.post(
                BASE_URL + "/chat/completions",
                json={"model": MODEL, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Nemotron indisponivel (LM Studio ligado e modelo carregado?): {e}")
            time.sleep(1.5 ** attempt)

def summarize_abstract(abstract: str, target_tokens=120) -> str:
    """Compressao economica de abstract antes de mandar pro Claude (economiza contexto)."""
    if len(abstract) < 400:
        return abstract
    return ask(
        [{"role": "user", "content": f"Resuma este abstract cientifico em ate {target_tokens} tokens, preservando: mecanismo, achado principal, populacao/modelo, limitacao. Texto:\n\n{abstract}"}],
        system="Voce comprime abstracts biomedicos preservando fato verificavel.",
        temperature=0.1,
        max_tokens=target_tokens + 60,
    )
