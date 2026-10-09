"""
TemporalBuffer — Memória de Trabalho Hierárquica com Decaimento (Mecanismo #4)
================================================================================
Buffer deslizante que mantém os últimos N chunks processados,
cada um com peso exponencialmente menor com a idade.

Isso substitui o `sign(accumulation)` simples do
AutoregressiveGenerator por uma memória de trabalho
genuína que preserva o contexto temporal.

Uso:
    tb = TemporalBuffer(capacity=10, decay=0.85)
    tb.push(chunk_vector)          # adiciona novo chunk
    context = tb.get_context()     # soma ponderada + normalização
"""

import torch


class TemporalBuffer:
    """
    Buffer temporal hierárquico.

    Capacidade finita (Lei de Miller: 7±2 chunks).
    Decaimento exponencial: chunk mais antigo contribui menos.
    """

    def __init__(self, capacity: int = 7, decay: float = 0.8):
        """
        Args:
            capacity: número máximo de chunks no buffer (default 7)
            decay: fator de decaimento por posição (default 0.8)
        """
        self.capacity = capacity
        self.decay = decay
        self.device = None

        # Buffer circular
        self.buffer = []  # lista de (timestamp, vector)
        self.tick = 0

    def push(self, vector: torch.Tensor):
        """
        Adiciona um chunk ao buffer.
        Se o buffer está cheio, descarta o chunk mais antigo.

        Args:
            vector: Manifold bipolar (±1) do chunk
        """
        if self.device is None:
            self.device = vector.device

        self.tick += 1
        self.buffer.append((self.tick, vector.clone()))

        # Mantém apenas os últimos N
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def get_context(self) -> torch.Tensor:
        """
        Retorna o contexto temporal como soma ponderada.
        Chunks mais recentes têm peso maior (decay^(posição_inversa)).

        Returns:
            Manifold float normalizado (não binarizado)
        """
        if not self.buffer:
            return torch.zeros(10000, device=self.device)

        n = len(self.buffer)
        weighted_sum = torch.zeros(10000, device=self.device)

        for i, (_, vec) in enumerate(self.buffer):
            # O chunk mais recente (i=n-1) tem peso 1.0
            # O mais antigo (i=0) tem peso decay^(n-1)
            age = n - 1 - i  # 0 = mais recente, n-1 = mais antigo
            weight = self.decay ** age
            weighted_sum += weight * vec

        # Normaliza para manter magnitude ~sqrt(10000)
        norm = weighted_sum.norm()
        if norm > 0:
            weighted_sum = weighted_sum / norm * (10000 ** 0.5)

        return weighted_sum

    def get_decayed(self) -> torch.Tensor:
        """
        Versão binarizada (±1) do contexto.
        Útil para bindar com o próximo chunk.
        """
        context = self.get_context()
        return torch.where(context > 0, 1.0, -1.0)

    def clear(self):
        """Limpa o buffer."""
        self.buffer = []
        self.tick = 0

    @property
    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    @property
    def usage(self) -> float:
        return len(self.buffer) / self.capacity
