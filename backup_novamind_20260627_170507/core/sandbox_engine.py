"""
Sandbox Engine — Motor de Composição Generativa
=================================================
Decompõe, Simula e Recompõe conceitos no espaço HDC.

Em vez de buscar frases inteiras na memória (como um banco vetorial),
o Sandbox quebra o input em palavras individuais, recupera a
"essência" geométrica de cada uma, e monta um NOVO manifold
composto — mesmo que a combinação nunca tenha sido vista antes.

Isto é a Generatividade Composicional:
  input:  "a dog is playing guitar"
  passo 1: extrai dog, playing, guitar
  passo 2: busca vetor de CADA palavra
  passo 3: funde com papéis sintáticos (suj * dog + ver * playing + obj * guitar)
  passo 4: BrocaDecoder traduz o novo pensamento em fala
"""

import re
import unicodedata
import torch
from typing import List, Tuple, Optional, Dict

from .coglang import (Manifold, bind_roles, unbind_role, decode_roles,
                       ALL_ROLES, CODE_ROLES,
                       ROLE_FUNCTION, ROLE_ARGUMENT, ROLE_ACTION)
from .grounding_engine import GroundingEngine


class SandboxEngine:
    """
    Sandbox mental: monta conceitos compostos no hiperespaço HDC.
    
    Uso:
        sandbox = SandboxEngine(engine)
        pensamento = sandbox.compose_concept("a dog is playing guitar")
        # pensamento é um Manifold NOVO, composto na hora
        # BrocaDecoder pode decodificá-lo em suj=dog, ver=playing, obj=guitar
    """

    def __init__(self, engine: GroundingEngine):
        self.engine = engine
        self.device = engine.device

    def extract_words(self, text: str) -> List[str]:
        """
        Extrai palavras significativas de um texto.
        Usa a mesma lógica do extract_svo do GroundingEngine.
        """
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        stop = GroundingEngine.STOPWORDS_SVO
        return [w for w in words if w not in stop]

    def compose_concept(self, input_text: str,
                        roles: Optional[List[str]] = None) -> Optional[Manifold]:
        """
        Decompõe o texto, encode cada palavra, e monta um NOVO Manifold
        no espaço HDC via Role-Filler Binding.
        
        Args:
            input_text: frase em linguagem natural
            roles: papéis sintáticos para binding (default: suj, ver, obj)
                   Se houver mais palavras que roles, as extras vão para adj/adv
        
        Returns:
            Manifold composto, ou None se não houver palavras suficientes
        """
        if roles is None:
            roles = ["suj", "ver", "obj"]

        words = self.extract_words(input_text)
        if not words:
            return None

        # Constrói pares role-palavra
        pairs = []
        for i, word in enumerate(words[:len(roles)]):
            role_name = roles[i]
            word_mf = self.engine.encode_text(word)
            pairs.append((role_name, word_mf))

        if len(pairs) < 1:
            return None

        # Se só uma palavra, retorna ela diretamente
        if len(pairs) == 1:
            return pairs[0][1]

        # Compõe via bind_roles (que já binariza internamente)
        composed = bind_roles(*pairs)

        return composed

    def compose_from_svo(self, svo: Dict[str, str]) -> Optional[Manifold]:
        """
        Compõe um manifold a partir de um dicionário SVO já extraído.
        
        Args:
            svo: dict com chaves 'suj', 'ver', 'obj' (valores podem ser "")
        
        Returns:
            Manifold composto ou None
        """
        pairs = []
        role_map = {"suj": "suj", "verb": "ver", "obj": "obj",
                    "subject": "suj", "verb": "ver", "object": "obj"}
        
        for key, word in svo.items():
            rn = role_map.get(key, key)
            if word and rn in dict(ALL_ROLES):
                word_mf = self.engine.encode_text(word)
                pairs.append((rn, word_mf))

        if len(pairs) < 1:
            return None
        if len(pairs) == 1:
            return pairs[0][1]

        return bind_roles(*pairs)

    def compose_code_concept(self, code_string: str,
                             prefix_args: bool = True) -> Optional[Manifold]:
        """
        Lê um código Python, extrai os papéis lógicos (Function, Arguments, Action)
        via AST e monta um Manifold HDC composto com os CODE_ROLES.

        Quebra underscores em nomes de função para melhor encoding:
            "sorted_list_sum" → encode_text("sorted list sum")

        Exemplo:
            "def soma(a, b):\\n    return a + b"
            → bind_roles(("func", mf_soma), ("arg", mf_a_b), ("act", mf_a_Add_b))

        Args:
            code_string: código Python como string
            prefix_args: se True, prefixa args de 1 letra com 'var_'
                         (ex: 'a' → 'var_a') para melhor entropia HDC

        Returns:
            Manifold composto, ou None se AST inválida
        """
        from .code_extractor import CodeExtractor

        roles = CodeExtractor.extract(code_string)
        if not roles:
            return None

        pairs = []
        code_role_map = {"function": "func", "arguments": "arg", "action": "act"}

        for key, role_name in code_role_map.items():
            word = roles.get(key, "")
            if word:
                # Quebra underscores para melhor hash n-grama
                word = word.replace('_', ' ')
                # Prefixa args de 1 letra para dar entropia
                if prefix_args and role_name == 'arg':
                    word = ' '.join(f'var_{w}' if len(w) == 1 else w
                                    for w in word.split())
                word_mf = self.engine.encode_text(word)
                pairs.append((role_name, word_mf))

        if len(pairs) < 1:
            return None
        if len(pairs) == 1:
            return pairs[0][1]

        return bind_roles(*pairs)

    def decode_roles(self, manifold: Manifold, vocab_matrix: torch.Tensor,
                     word_list: List[str], min_sim: float = 0.12,
                     roles: Optional[List[Tuple[str, object]]] = None) -> List[Tuple[str, str, float]]:
        """
        Decodifica um manifold composto em papéis sintáticos.
        Aceita roles customizados (ex: CODE_ROLES para programação).
        """
        from .coglang import decode_roles as _dr
        target_roles = roles or ALL_ROLES
        return _dr(manifold, vocab_matrix, word_list, min_sim=min_sim, roles=target_roles)

    # ─────────────────────────────────────────────────────────────────
    # Associação Multimodal — Redes de Projeção Lineares no Espaço HDC
    # ─────────────────────────────────────────────────────────────────

    def _get_or_create_projection(self, mode: str) -> torch.Tensor:
        """Retorna (ou inicializa) a matriz de projeção para o modo dado."""
        if not hasattr(self, '_projections'):
            self._projections: Dict[str, torch.Tensor] = {}
        if mode not in self._projections:
            dim = Manifold.DIMENSIONALITY
            # Inicialização com identidade + ruído pequeno para evitar simetria
            W = torch.eye(dim, device=self.device) + 0.01 * torch.randn(dim, dim, device=self.device)
            self._projections[mode] = W
        return self._projections[mode]

    def train_association(
        self,
        inputs: List[Manifold],
        targets: List[Manifold],
        mode: str = "audio_to_visual",
        epochs: int = 80,
        lr: float = 0.005,
    ) -> None:
        """
        Treina uma projeção linear W : HDC_dim → HDC_dim mapeando inputs→targets.

        O aprendizado ocorre diretamente nos vetores bipolares do espaço HDC
        usando descida de gradiente online (sem PyTorch autograd — multiplicação
        matricial explícita + regra delta Widrow-Hoff).

        Args:
            inputs:  lista de Manifolds de entrada (ex: áudio)
            targets: lista de Manifolds alvo (ex: visual)
            mode:    chave do mapa de projeção ('audio_to_visual', 'text_to_visual',
                     'visual_to_audio', ...)
            epochs:  número de passagens sobre o conjunto de treino
            lr:      taxa de aprendizado
        """
        if not inputs or not targets:
            print(f"[SANDBOX] train_association({mode}): lista vazia, pulando.")
            return

        W = self._get_or_create_projection(mode)
        n = min(len(inputs), len(targets))
        dim = Manifold.DIMENSIONALITY

        print(f"  [SANDBOX] Treinando '{mode}' com {n} pares, {epochs} épocas, lr={lr}")
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(n):
                x = inputs[i].tensor.float()   # (dim,)
                y = targets[i].tensor.float()  # (dim,)

                # Forward: ŷ = W @ x
                y_hat = W @ x                  # (dim,)

                # Erro e loss MSE
                err = y - y_hat
                total_loss += (err ** 2).mean().item()

                # Gradiente: dW = lr * err ⊗ x  (regra delta)
                W += lr * torch.outer(err, x)

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n
                print(f"    Época {epoch + 1:3d}/{epochs} | loss={avg_loss:.6f}")

        self._projections[mode] = W
        print(f"  [SANDBOX] '{mode}' treinado.")

    def _project(self, manifold: Manifold, mode: str) -> Manifold:
        """Aplica a projeção treinada para 'mode' e binariza o resultado."""
        W = self._get_or_create_projection(mode)
        x = manifold.tensor.float()
        y_hat = torch.sign(W @ x)
        return Manifold(y_hat, device=self.device)

    def predict_visual_from_audio(self, audio_manifold: Manifold) -> Manifold:
        """Prediz manifold visual a partir de um manifold de áudio."""
        return self._project(audio_manifold, "audio_to_visual")

    def predict_visual_from_text(self, text_manifold: Manifold) -> Manifold:
        """Prediz manifold visual a partir de um manifold de texto."""
        return self._project(text_manifold, "text_to_visual")

    def predict_audio_from_visual(self, visual_manifold: Manifold) -> Manifold:
        """Prediz manifold de áudio a partir de um manifold visual."""
        return self._project(visual_manifold, "visual_to_audio")

    def predict_sequence_from_text(
        self,
        text_manifold: Manifold,
        position_manifolds: List[Manifold],
    ) -> List[Manifold]:
        """
        "Sonho Acordado": projeta o manifold de texto em espaço visual e
        decompõe o manifold composto em N frames temporais via unbinding posicional.

        Args:
            text_manifold:      manifold do texto de entrada
            position_manifolds: lista de manifolds de posição temporal (P_t)

        Returns:
            lista de Manifolds, um por posição temporal
        """
        # 1. Projeta texto → espaço visual
        visual_pred = self.predict_visual_from_text(text_manifold)

        # 2. Faz unbinding frame-a-frame: frame_t = visual_pred * P_t
        #    (binding é sua própria inversa em MAP-HDC quando P é gerado aleatório)
        frames = []
        for pos_mf in position_manifolds:
            frame = visual_pred * pos_mf
            frames.append(frame)

        return frames
