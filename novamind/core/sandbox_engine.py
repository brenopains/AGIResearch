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
