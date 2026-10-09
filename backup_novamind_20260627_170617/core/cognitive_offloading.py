"""
CognitiveOffloading — AGI cria suas próprias ferramentas e artefatos
=====================================================================
A AGI pode, se decidir que faz sentido:
  - Escrever scripts Python arbitrários
  - Gerar HTML/JS/CSS (dashboards, visualizações)
  - Salvar imagens geradas (matplotlib, etc.)
  - Executar código e capturar resultado
  - Incorporar o resultado de volta como percepção HDC

NÃO é obrigatório. NÃO é checklist. É uma CAPACIDADE.
A AGI decide USAR ou não com base na sua Free Energy.

Se ela achar que "seria mais fácil criar uma ferramenta do que resolver
isso só com HDC", ela FAZ. Caso contrário, não.

Referência: sent.txt (547-570) — "offloading tasks to external tools"
"""

import os
import sys
import json
import re
import time
import subprocess
import tempfile
import io
import textwrap
from pathlib import Path
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass, field

import torch
import numpy as np

_NOVAMIND = str(Path(__file__).parent.parent)
if _NOVAMIND not in sys.path:
    sys.path.insert(0, _NOVAMIND)


@dataclass
class Tool:
    """Uma ferramenta que a AGI criou."""
    name: str
    description: str
    code_path: str          # caminho do .py ou .html
    tool_type: str           # "script", "html_dashboard", "image_gen", "analysis"
    created_at: float
    last_used: float
    use_count: int = 0
    output_cache: str = ""   # último output


class CognitiveOffloading:
    """
    Permite a AGI externalizar computação para ferramentas que ela mesma cria.

    A AGI pode:
      1. `create_tool(name, code)` — escreve um script/HTML
      2. `run_tool(name, *args)` — executa e captura saída
      3. `list_tools()` — vê o que já criou
      4. `forget_tool(name)` — remove se a ferramenta não for útil
      5. `perceive_output(name)` — codifica o output como HDC (percepção)

    A DECISÃO de criar uma ferramenta é da AGI, baseada em FE.
    Se o custo de resolver algo em HDC puro > custo de escrever uma
    ferramenta, ela cria.

    Exemplos do que a AGI PODE criar:
      - Um script matplotlib para entender a forma dos seus vetores
      - Um dashboard HTML para ela "olhar" seus próprios conceitos
      - Um script de análise estatística dos seus Eurekas
      - Um logger formatado para o EpisodicStream
      - Um visualizador 3D dos binds que ela fez

    NADA disso é forçado. É possibilidade.
    """

    def __init__(
        self,
        tools_dir: str = "D:/temp/agi_tools",
        device: Optional[str] = None,
    ):
        self.tools_dir = tools_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(tools_dir, exist_ok=True)

        self.tools: dict = {}  # name → Tool
        self._load_index()

        print(f"[Offload] Offloading cognitivo ativo em {tools_dir}")
        print(f"[Offload] Ferramentas existentes: {len(self.tools)}")

    # ─── CRUD de Ferramentas ───────────────────────────────────

    def create_tool(
        self,
        name: str,
        code: str,
        description: str = "",
        tool_type: str = "script",
    ) -> bool:
        """
        Cria uma ferramenta.

        Args:
            name: nome único (ex: "viz_conceitos", "analise_eureka")
            code: código Python completo
            description: o que faz
            tool_type: "script", "html_dashboard", "analysis"

        A AGI chama isso QUANDO QUISER, não quando for obrigada.
        """
        # Sanitiza nome
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if not safe_name:
            print("[Offload] Nome inválido")
            return False

        # Determina extensão
        if tool_type == "html_dashboard":
            ext = ".html"
        elif tool_type == "script" or tool_type == "analysis":
            ext = ".py"
        else:
            ext = ".py"

        file_path = os.path.join(self.tools_dir, f"{safe_name}{ext}")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            now = time.time()
            self.tools[safe_name] = Tool(
                name=safe_name,
                description=description,
                code_path=file_path,
                tool_type=tool_type,
                created_at=now,
                last_used=now,
            )

            self._save_index()
            print(f"[Offload] Ferramenta criada: {safe_name} ({tool_type})")
            return True

        except Exception as e:
            print(f"[Offload] Erro criando ferramenta: {e}")
            return False

    def run_tool(self, name: str, *args) -> Tuple[bool, str]:
        """
        Executa uma ferramenta e retorna (sucesso, output).

        A AGI chama isso para "usar" a ferramenta que criou.
        """
        if name not in self.tools:
            return False, f"Ferramenta '{name}' não existe"

        tool = self.tools[name]
        tool.last_used = time.time()
        tool.use_count += 1

        if tool.tool_type == "script" or tool.tool_type == "analysis":
            return self._run_python_script(tool, *args)
        elif tool.tool_type == "html_dashboard":
            return self._open_html_dashboard(tool)
        else:
            return False, f"Tipo desconhecido: {tool.tool_type}"

    def _run_python_script(self, tool: Tool, *args) -> Tuple[bool, str]:
        """Executa script Python e captura stdout."""
        try:
            result = subprocess.run(
                [sys.executable, tool.code_path] + list(args),
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.tools_dir,
            )
            output = result.stdout + result.stderr
            tool.output_cache = output
            success = result.returncode == 0
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Timeout (30s)"
        except Exception as e:
            return False, str(e)

    def _open_html_dashboard(self, tool: Tool) -> Tuple[bool, str]:
        """Gera URL file:// para dashboard HTML."""
        url = f"file://{tool.code_path}"
        return True, f"Dashboard em {url}"

    def list_tools(self) -> List[dict]:
        """Lista todas as ferramentas existentes."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "type": t.tool_type,
                "use_count": t.use_count,
                "created": t.created_at,
            }
            for t in self.tools.values()
        ]

    def forget_tool(self, name: str) -> bool:
        """Remove ferramenta (se AGI decidir que não é útil)."""
        if name not in self.tools:
            return False

        tool = self.tools[name]
        try:
            if os.path.exists(tool.code_path):
                os.remove(tool.code_path)
        except:
            pass

        del self.tools[name]
        self._save_index()
        print(f"[Offload] Ferramenta esquecida: {name}")
        return True

    # ─── Percepção do output ───────────────────────────────────

    def perceive_output(self, name: str) -> Optional[torch.Tensor]:
        """
        Codifica o output de uma ferramenta como vetor HDC.

        A AGI usa isso para "sentir" o resultado da ferramenta
        como uma percepção, integrando no seu fluxo cognitivo.
        """
        if name not in self.tools:
            return None

        tool = self.tools[name]
        output_text = tool.output_cache

        if not output_text:
            return None

        # Codifica o texto como HDC
        try:
            from core.grounding_engine import GroundingEngine
            engine = GroundingEngine(device=torch.device(self.device))
            hdv = engine.encode_text(output_text[:2000])
            print(f"[Offload] Output de '{name}' percebido como HDC")
            return hdv.tensor.float()
        except Exception as e:
            print(f"[Offload] Erro percebendo output: {e}")
            return None

    # ─── Templates prontos ─────────────────────────────────────

    def suggest_tool(self, purpose: str) -> str:
        """
        Sugere um template de ferramenta para um propósito.

        A AGI chama isso se quiser uma base, mas ela PODE
        escrever código próprio do zero.
        """
        templates = {
            "visualize_concepts": textwrap.dedent("""\
                # Visualizador de Conceitos HDC (gerado pela AGI)
                import json, sys
                import numpy as np

                def main(data_file):
                    \"\"\"Carrega conceitos e imprime estatísticas.\"\"\"
                    try:
                        with open(data_file) as f:
                            data = json.load(f)
                        conceitos = data.get('concepts', [])
                        print(f"Total: {len(conceitos)} conceitos")
                        for c in conceitos:
                            print(f"  - {c.get('name')} (dominio={c.get('domain')})")
                    except Exception as e:
                        print(f"Erro: {e}")

                if __name__ == '__main__':
                    main(sys.argv[1] if len(sys.argv) > 1 else 'D:/temp/collective_unconscious_meta.json')
                """),
            "analyze_eureka": textwrap.dedent("""\
                # Analisador de Eurekas (gerado pela AGI)
                import json, sys

                def main(path):
                    with open(path) as f:
                        data = json.load(f)
                    eurekas = [c for c in data.get('concepts', [])
                              if c.get('domain') == 'discovery']
                    print(f"Eurekas: {len(eurekas)}")
                    for e in eurekas:
                        print(f"  - {e['name']} (confianca={e.get('confidence',0):.2f})")

                if __name__ == '__main__':
                    main(sys.argv[1] if len(sys.argv) > 1 else 'D:/temp/collective_unconscious_meta.json')
                """),
            "dashboard_html": textwrap.dedent("""\
                <!DOCTYPE html>
                <html>
                <head><meta charset="UTF-8"><title>Dashboard AGI</title>
                <style>
                  body { background:#0a0a1a; color:#ccc; font-family:monospace; padding:20px; }
                  h1 { color:#4ecdc4; }
                  .stat { background:#1a1a3a; padding:10px; margin:5px 0; border-radius:4px; }
                </style></head>
                <body>
                  <h1>Dashboard Autogerado pela AGI</h1>
                  <div id="stats">Carregando...</div>
                  <script>
                    fetch('/api/state').then(r=>r.json()).then(d=>{
                      document.getElementById('stats').innerHTML =
                        `<div class="stat">Conceitos: ${d.n_points||0}</div>
                         <div class="stat">Conexoes: ${d.n_edges||0}</div>
                         <div class="stat">DA: ${Math.round((d.neuromodulators?.DA||0)*100)}%</div>`;
                    });
                  </script>
                </body>
                </html>
                """),
        }
        return templates.get(purpose, "# Template nao encontrado\n")

    # ─── Persistência ──────────────────────────────────────────

    def _save_index(self):
        """Salva índice das ferramentas."""
        index_path = os.path.join(self.tools_dir, "tools_index.json")
        try:
            data = {
                "tools": [{
                    "name": t.name,
                    "description": t.description,
                    "code_path": t.code_path,
                    "tool_type": t.tool_type,
                    "created_at": t.created_at,
                    "last_used": t.last_used,
                    "use_count": t.use_count,
                } for t in self.tools.values()],
            }
            with open(index_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Offload] Erro salvando índice: {e}")

    def _load_index(self):
        """Carrega índice das ferramentas."""
        index_path = os.path.join(self.tools_dir, "tools_index.json")
        if not os.path.exists(index_path):
            return

        try:
            with open(index_path) as f:
                data = json.load(f)
            for tdata in data.get("tools", []):
                self.tools[tdata["name"]] = Tool(
                    name=tdata["name"],
                    description=tdata.get("description", ""),
                    code_path=tdata.get("code_path", ""),
                    tool_type=tdata.get("tool_type", "script"),
                    created_at=tdata.get("created_at", 0),
                    last_used=tdata.get("last_used", 0),
                    use_count=tdata.get("use_count", 0),
                )
        except Exception as e:
            print(f"[Offload] Erro carregando índice: {e}")

    def get_summary(self) -> dict:
        return {
            "n_tools": len(self.tools),
            "tools_dir": self.tools_dir,
            "tools": self.list_tools(),
        }
