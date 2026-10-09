"""
StructuralMetrics — Mede propriedades estruturais da arquitetura cognitiva.

Nao mede erro ou acerto. Mede COMO a AGI esta organizada.

Metricas:
  - Modularity: quao bem dividida em modulos
  - Clustering coefficient: qto os vizinhos de um no sao vizinhos entre si
  - Small-worldness: equilibrio entre clustering e path length
  - Centrality: quais nos sao hubs
  - Entropy: diversidade de dinamica entre variaveis
  - Integrated information (phi): qto o sistema e mais que a soma das partes
  - Compressibility: qto o modelo pode ser simplificado

Cada metrica e computada a partir da MATRIZ DE CONECTIVIDADE do modelo ODE.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class StructuralProfile:
    """Perfil estrutural completo de um modelo."""
    name: str
    n_vars: int
    modularity: float  # 0-1, 1 = perfeitamente modular
    clustering: float  # 0-1, 1 = todos vizinhos conectados
    small_worldness: float  # >1 = small-world
    centrality_max: float  # 0-1, quao concentrado em hubs
    centrality_entropy: float  # entropia da distribuicao de centralidade
    entropy: float  # diversidade de dinamica (normalizada)
    integrated_info: float  # phi aproximado (0-1)
    compressibility: float  # 0-1, 1 = totalmente comprimivel

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_vars": self.n_vars,
            "modularity": round(self.modularity, 4),
            "clustering": round(self.clustering, 4),
            "small_worldness": round(self.small_worldness, 4),
            "centrality_max": round(self.centrality_max, 4),
            "centrality_entropy": round(self.centrality_entropy, 4),
            "entropy": round(self.entropy, 4),
            "integrated_info": round(self.integrated_info, 4),
            "compressibility": round(self.compressibility, 4),
        }


class StructuralAnalyzer:
    """Analisa a estrutura de conectividade de modelos ODE.

    Uso:
        analyzer = StructuralAnalyzer()
        profile = analyzer.analyze(model)
        print(f"Modularity: {profile.modularity:.2f}")
    """

    def analyze(self, model) -> StructuralProfile:
        """Analisa a estrutura de conectividade de um modelo."""
        n = len(model.variables)
        if n == 0:
            return StructuralProfile(model.name, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # 1. Constroi matriz de conectividade a partir da ODE
        adj = self._build_adjacency(model)
        
        # 2. Computa metricas
        modularity = self._modularity(adj)
        clustering = self._clustering(adj)
        small_world = self._small_worldness(adj)
        centrality = self._centrality(adj)
        entropy = self._entropy(model, adj)
        integrated = self._integrated_info(adj)
        compress = self._compressibility(adj)

        return StructuralProfile(
            name=model.name[:30],
            n_vars=n,
            modularity=modularity,
            clustering=clustering,
            small_worldness=small_world,
            centrality_max=centrality["max"],
            centrality_entropy=centrality["entropy"],
            entropy=entropy,
            integrated_info=integrated,
            compressibility=compress,
        )

    def _build_adjacency(self, model) -> np.ndarray:
        """Constroi matriz de adjacencia a partir da ODE.
        
        Para cada par (i,j), testa se y[j] influencia dy[i]:
          - Avalia a ODE com y[j] = y[j] + epsilon
          - Se dy[i] mudar > threshold, existe conexao
        """
        n = len(model.variables)
        adj = np.zeros((n, n), dtype=float)
        y0 = model.initial_state.copy()
        eps = 0.01
        threshold = 1e-6
        
        try:
            base_deriv = model.ode_fn(0, y0)
            for j in range(n):
                y_pert = y0.copy()
                y_pert[j] += eps
                pert_deriv = model.ode_fn(0, y_pert)
                for i in range(n):
                    if i < len(base_deriv) and i < len(pert_deriv):
                        diff = abs(pert_deriv[i] - base_deriv[i])
                        adj[i, j] = 1.0 if diff > threshold else 0.0
        except:
            # Fallback: identidade
            np.fill_diagonal(adj, 1)
        
        return adj

    def _modularity(self, adj: np.ndarray) -> float:
        """Modularidade: quao bem o grafo se divide em comunidades.
        
        Usa algoritmo de Newman (aproximado): mede densidade de edges
        dentro vs entre grupos.
        """
        n = adj.shape[0]
        if n < 2:
            return 0.0
        
        degree = adj.sum(axis=1)
        m = degree.sum()
        if m == 0:
            return 0.0
        
        # Modularidade Q (Newman, 2006) - versao simplificada
        # Q = 1/(2m) * sum(A_ij - k_i*k_j/(2m)) * delta(c_i, c_j)
        # Aproximacao: usa correlacao entre colunas como "comunidade"
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    Q += 1.0 - (degree[i] * degree[j]) / (2 * m)
        
        return max(0, min(1, Q / (2 * m))) if m > 0 else 0

    def _clustering(self, adj: np.ndarray) -> float:
        """Coeficiente de clustering medio (Watts-Strogatz).
        
        Mede: dos vizinhos de um no, quantos sao vizinhos entre si.
        """
        n = adj.shape[0]
        if n < 3:
            return 0.0
        
        total_c = 0.0
        n_nodes = 0
        
        for i in range(n):
            neighbors = [j for j in range(n) if adj[i, j] > 0 and j != i]
            k = len(neighbors)
            if k < 2:
                continue
            
            # Conta edges entre vizinhos
            edges = 0
            for a in range(k):
                for b in range(a + 1, k):
                    if adj[neighbors[a], neighbors[b]] > 0:
                        edges += 1
            
            # Coeficiente = 2*edges / (k*(k-1))
            ci = (2.0 * edges) / (k * (k - 1))
            total_c += ci
            n_nodes += 1
        
        return total_c / max(1, n_nodes)

    def _small_worldness(self, adj: np.ndarray) -> float:
        """Small-worldness: (C/C_rand) / (L/L_rand).
        
        > 1 = small-world network.
        C = clustering coefficient, L = average path length.
        """
        n = adj.shape[0]
        if n < 3:
            return 1.0
        
        C = self._clustering(adj)
        
        # Path length media (aproximada por BFS limitada)
        L = 0.0
        n_pairs = 0
        for i in range(min(n, 20)):  # limitado para performance
            # BFS
            dist = [-1] * n
            dist[i] = 0
            queue = [i]
            for q in queue:
                for j in range(n):
                    if adj[q, j] > 0 and dist[j] == -1:
                        dist[j] = dist[q] + 1
                        queue.append(j)
            for j in range(i + 1, min(n, 20)):
                if dist[j] > 0:
                    L += dist[j]
                    n_pairs += 1
        
        L = L / max(1, n_pairs) if n_pairs > 0 else 1.0
        
        # Compara com grafo aleatorio equivalente
        p = adj.sum() / max(1, n * (n - 1))
        C_rand = p
        L_rand = math.log(n) / max(0.01, math.log(n * p + 1))
        
        if C_rand == 0 or L_rand == 0:
            return 1.0
        
        return (C / C_rand) / (L / L_rand)

    def _centrality(self, adj: np.ndarray) -> Dict[str, float]:
        """Centralidade: quais nos sao hubs.
        
        Max: hub mais conectado.
        Entropia: quao concentrada e a conectividade.
        """
        n = adj.shape[0]
        if n == 0:
            return {"max": 0, "entropy": 0}
        
        degree = adj.sum(axis=1)
        total = degree.sum()
        
        if total == 0:
            return {"max": 0, "entropy": 0}
        
        max_cent = degree.max() / max(1, degree.sum())
        
        # Entropia de Shannon da distribuicao de graus
        probs = degree / total
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normaliza por log2(n)
        entropy = entropy / max(1, math.log2(n))
        
        return {"max": float(max_cent), "entropy": float(entropy)}

    def _entropy(self, model, adj: np.ndarray) -> float:
        """Entropia da dinamica: diversidade de comportamento entre variaveis.
        
        Simula o modelo e mede quao diferentes sao as trajetorias.
        """
        from scipy.integrate import solve_ivp
        
        try:
            sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                          method="RK45", max_step=50, rtol=1e-3, atol=1e-6)
            if sol.y.shape[1] < 2:
                return 0.0
            
            # Variancia entre variaveis no final da simulacao
            final_vars = sol.y[:, -1]
            var = np.var(final_vars)
            
            # Normaliza
            return float(min(1.0, var / (var + 1)))
        except:
            return 0.0

    def _integrated_info(self, adj: np.ndarray) -> float:
        """Informacao integrada (phi) aproximada.
        
        Mede: qto a informacao do todo e maior que a soma das partes.
        Se phi = 0, o sistema e puramente reducionista.
        Se phi > 0, existe integracao.
        """
        n = adj.shape[0]
        if n < 4:
            return 0.0
        
        # Aproximacao: phi ~ (edges_global / sum(edges_modulos)) - 1
        # Quanto mais edges entre modulos vs dentro, maior phi
        
        # Divide em 2 grupos aleatorios, mede edges entre grupos
        best_phi = 0.0
        for _ in range(10):
            split = np.random.RandomState(_).randint(0, 2, n)
            g1 = np.where(split == 0)[0]
            g2 = np.where(split == 1)[0]
            
            if len(g1) == 0 or len(g2) == 0:
                continue
            
            # Edges dentro de cada grupo
            within = 0
            between = 0
            for i in range(n):
                for j in range(n):
                    if adj[i, j] > 0:
                        if split[i] == split[j]:
                            within += 1
                        else:
                            between += 1
            
            total = within + between
            if total == 0:
                continue
            
            # Phi = 1 - within/total (maior between = maior integracao)
            phi = between / total
            if phi > best_phi:
                best_phi = phi
        
        return float(best_phi)

    def _compressibility(self, adj: np.ndarray) -> float:
        """Compressibilidade: qto o modelo pode ser simplificado.
        
        Mede a regularidade da matriz de adjacencia.
        Matrizes aleatorias = baixa compressibilidade.
        Matrizes com padrao = alta compressibilidade.
        """
        n = adj.shape[0]
        if n < 2:
            return 1.0
        
        # Comprime a matriz como string e mede razao
        # Matrizes regulares = alta compressao
        flat = adj.flatten()
        
        # Conta transicoes (0->0, 0->1, 1->0, 1->1)
        transitions = 0
        for i in range(1, len(flat)):
            if flat[i] != flat[i-1]:
                transitions += 1
        
        # Compressibilidade inversamente proporcional a transicoes
        max_trans = len(flat) - 1
        if max_trans == 0:
            return 1.0
        
        return 1.0 - (transitions / max_trans)


def run_full_analysis(models: list) -> Dict[str, StructuralProfile]:
    """Analisa multiplos modelos e retorna perfis."""
    analyzer = StructuralAnalyzer()
    profiles = {}
    for m in models:
        profiles[m.name] = analyzer.analyze(m)
    return profiles
