"""
SelfVector — Manifold Egocêntrico Estável (Mecanismo #3)
==========================================================
Um vetor SELF persistente que ancora todas as percepções.
Cada composição de conceito é bindada com SELF antes de
ser armazenada ou usada para raciocínio, criando um
referencial egocêntrico.

Propriedades:
  - Estável por padrão (muda lentamente)
  - Adapta-se via média exponencial das últimas percepções
  - Fornece "affordances": o que este conceito SIGNIFICA para mim?

Uso:
    sv = SelfVector(engine)
    percepcao = sv.compose_and_anchor("dog chase cat")
    # bind automático com SELF
"""

import torch
import torch.nn.functional as F


class SelfVector:
    """
    Vetor SELF adaptativo.
    self.vector = Manifold bipolar (±1) com taxa de adaptação lenta.
    """

    def __init__(self, engine, lr: float = 0.01):
        """
        Args:
            engine: HDC engine (para encode)
            lr: taxa de adaptação do SELF (default 0.01 = 1%)
        """
        self.engine = engine
        self.device = engine.device
        self.lr = lr

        # Inicializa SELF como vetor aleatório bipolar
        raw = torch.randn(10000, device=self.device)
        self.vector = torch.where(raw > 0, 1.0, -1.0)
        self.steps = 0

    def anchor(self, manifold: torch.Tensor) -> torch.Tensor:
        """
        Âncorar percepção com SELF: perception * SELF
        Isso cria um referencial egocêntrico.

        Returns:
            Manifold egocêntrico (percepção no contexto do SELF)
        """
        return manifold * self.vector

    def update(self, manifold: torch.Tensor):
        """
        Adapta SELF lentamente à nova percepção.
        self.vector += lr * (manifold - self.vector)  (unbind)
        Depois binariza de volta para ±1.
        """
        self.steps += 1
        # SELF move-se na direção da percepção (aprendizado lento)
        delta = self.lr * (manifold - self.vector)
        self.vector = self.vector + delta
        # Binariza de volta para bipolar
        self.vector = torch.where(self.vector > 0, 1.0, -1.0)

    def compose_anchored(self, text: str) -> torch.Tensor:
        """
        Compõe um conceito e ancora com SELF.

        Args:
            text: texto do conceito

        Returns:
            Manifold egocêntrico (conceito * SELF)
        """
        if hasattr(self.engine, 'compose_concept'):
            manifold = self.engine.compose_concept(text)
        else:
            manifold = self.engine.encode_text(text)
        return self.anchor(manifold)

    def affordance(self, text: str) -> float:
        """
        Mede a "afordância" de um conceito: similaridade com SELF.
        Quanto mais próximo de SELF, mais o conceito é "significativo"
        para o agente.

        Returns:
            similaridade cosseno entre [0, 1]
        """
        if hasattr(self.engine, 'compose_concept'):
            mf = self.engine.compose_concept(text)
        else:
            mf = self.engine.encode_text(text)
        sim = F.cosine_similarity(mf.unsqueeze(0), self.vector.unsqueeze(0))
        return sim.item()

    def is_self_related(self, text: str, threshold: float = 0.3) -> bool:
        """True se o conceito tem alta afordância (similar com SELF)."""
        return self.affordance(text) > threshold

    def reset(self):
        """Reseta SELF para vetor aleatório."""
        raw = torch.randn(10000, device=self.device)
        self.vector = torch.where(raw > 0, 1.0, -1.0)
        self.steps = 0
