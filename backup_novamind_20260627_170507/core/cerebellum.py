"""
Cerebellum — Forward Model + Inverse Model + Timing Articulatório
=================================================================
O cerebelo real tem 80% dos neurônios do encéfalo. Não controla
movimento — PREDIZ consequências de ações e corrige ANTES de executar.

Dois modelos:
  Forward Model:  (estado, ação) → estado_previsto
  Inverse Model:  (estado, estado_desejado) → ação_necessária

Circuito biológico:
  córtex motor → ponte → cerebelo → núcleo denteado → tálamo → córtex
  (copy of motor command → prediction → error → correction)

Usado em:
  1. Timing articulatório de fala (função original)
  2. Predição de consequências de ações motoras
  3. Correção feedforward de ações cognitivas

Referências:
  Wolpert & Kawato (1998) - Multiple paired forward and inverse models
  Ito (2008) - Control of mental activities by internal models
  Bastian (2011) - Moving, sensing and learning with cerebellar damage
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque


# ─── Constantes ────────────────────────────────────────────────────────────────
STATE_DIM  = 15   # dimensão do estado corporal
ACTION_DIM = 4    # dimensão da ação [dx, dy, dz, grasp]

GESTURE_TIMING_PROTOTYPES = {
    "jaw_open":   {"min_ms": 30,  "max_ms": 200, "default_ms": 80},
    "tongue":     {"min_ms": 40,  "max_ms": 250, "default_ms": 100},
    "lip":        {"min_ms": 20,  "max_ms": 150, "default_ms": 60},
    "velum":      {"min_ms": 30,  "max_ms": 120, "default_ms": 50},
    "vocal_fold": {"min_ms": 10,  "max_ms": 80,  "default_ms": 30},
}


# ─── MLP simples em numpy puro (sem torch/tensorflow) ─────────────────────────

class TinyMLP:
    """
    Rede MLP de 2 camadas implementada com numpy puro.
    Online learning (1 amostra por vez). Sem dependências externas.

    Arquitetura: input → hidden(relu) → output(linear)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, lr: float = 0.01):
        self.in_dim     = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.lr         = lr

        # Inicialização He (boa para ReLU)
        scale_h = np.sqrt(2.0 / in_dim)
        scale_o = np.sqrt(2.0 / hidden_dim)
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((hidden_dim, in_dim)).astype(np.float32) * scale_h
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.standard_normal((out_dim, hidden_dim)).astype(np.float32) * scale_o
        self.b2 = np.zeros(out_dim, dtype=np.float32)

        # Buffer de experiências (mini-batch de 1)
        self._last_h  = np.zeros(hidden_dim, dtype=np.float32)
        self._last_in = np.zeros(in_dim, dtype=np.float32)
        self._last_out = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Armazena ativações para backprop."""
        x = x.astype(np.float32)
        h = np.maximum(0.0, self.W1 @ x + self.b1)   # ReLU
        out = self.W2 @ h + self.b2                    # linear
        self._last_h   = h
        self._last_in  = x
        self._last_out = out
        return out

    def update(self, target: np.ndarray) -> float:
        """
        Backprop manual para 1 amostra.
        Returns: MSE loss.
        """
        target = target.astype(np.float32)
        pred   = self._last_out

        # Loss e gradiente de saída
        loss   = float(np.mean((pred - target) ** 2))
        d_out  = 2.0 * (pred - target) / self.out_dim

        # Gradientes W2, b2
        dW2 = np.outer(d_out, self._last_h)
        db2 = d_out

        # Backprop através da camada hidden
        d_h    = self.W2.T @ d_out
        d_relu = d_h * (self._last_h > 0).astype(np.float32)  # derivada ReLU

        # Gradientes W1, b1
        dW1 = np.outer(d_relu, self._last_in)
        db1 = d_relu

        # Atualização (gradient descent)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss


# ─── Forward Model ─────────────────────────────────────────────────────────────

class ForwardModel:
    """
    Prediz o próximo estado dado o estado atual e a ação proposta.
    (estado, ação) → estado_previsto

    Treinado online com experiências reais do corpo.
    Erro de predição → sinal para neuromodulação (NA sobe com surpresa).
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 64, lr: float = 0.01):
        self.in_dim    = state_dim + action_dim
        self.out_dim   = state_dim
        self.mlp       = TinyMLP(self.in_dim, hidden, self.out_dim, lr)
        self.error_history: deque = deque(maxlen=100)
        self.train_steps = 0

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Prediz próximo estado.

        Args:
            state:  vetor de estado atual (STATE_DIM,)
            action: vetor de ação proposta (ACTION_DIM,)

        Returns:
            estado previsto (STATE_DIM,)
        """
        inp = np.concatenate([state.flatten()[:STATE_DIM],
                               action.flatten()[:ACTION_DIM]])
        return self.mlp.forward(inp)

    def update(self, state: np.ndarray, action: np.ndarray,
               actual_next_state: np.ndarray) -> float:
        """
        Atualiza o modelo com experiência real.

        Returns: erro de predição (para neuromodulação).
        """
        # Forward para obter predição
        self.predict(state, action)

        # Backprop com estado real como target
        target = actual_next_state.flatten()[:STATE_DIM].astype(np.float32)
        loss   = self.mlp.update(target)

        self.error_history.append(loss)
        self.train_steps += 1
        return loss

    def get_prediction_error(self) -> float:
        """Erro médio de predição recente (para NA / surpresa)."""
        if not self.error_history:
            return 0.0
        return float(np.mean(list(self.error_history)[-20:]))


# ─── Inverse Model ─────────────────────────────────────────────────────────────

class InverseModel:
    """
    Dado estado atual e estado desejado, sugere a ação necessária.
    (estado_atual, estado_desejado) → ação

    Permite planejamento: "para chegar em X, que ação devo fazer?"
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 64, lr: float = 0.01):
        self.in_dim  = state_dim * 2
        self.out_dim = action_dim
        self.mlp     = TinyMLP(self.in_dim, hidden, self.out_dim, lr)
        self.train_steps = 0

    def suggest_action(self, current_state: np.ndarray,
                       desired_state: np.ndarray) -> np.ndarray:
        """
        Sugere ação para ir de current_state → desired_state.

        Returns:
            vetor de ação sugerida (ACTION_DIM,)
        """
        inp = np.concatenate([current_state.flatten()[:STATE_DIM],
                               desired_state.flatten()[:STATE_DIM]])
        return self.mlp.forward(inp)

    def update(self, current_state: np.ndarray, desired_state: np.ndarray,
               actual_action: np.ndarray) -> float:
        """Atualiza com ação real que levou do estado atual ao desejado."""
        self.suggest_action(current_state, desired_state)
        target = actual_action.flatten()[:ACTION_DIM].astype(np.float32)
        loss   = self.mlp.update(target)
        self.train_steps += 1
        return loss


# ─── Cerebellum (unificado) ────────────────────────────────────────────────────

class Cerebellum:
    """
    Cerebelo completo: timing articulatório + Forward Model + Inverse Model.

    Mantém compatibilidade total com a interface anterior (predict_timing,
    predict_interval, correct_timing, learn, learn_interval).

    Adiciona:
      predict_consequence() — o que vai acontecer se eu fizer X?
      suggest_correction()  — o que devo fazer para chegar em Y?
      observe_result()      — aprende com o que realmente aconteceu
    """

    def __init__(self, learning_rate: float = 0.05,
                 state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM):
        self.lr = learning_rate

        # ─── Interface original (timing articulatório) ──────────────────────
        self.timing_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.interval_predictions: Dict[Tuple, deque] = defaultdict(lambda: deque(maxlen=10))

        # ─── Forward e Inverse Models ────────────────────────────────────────
        self.forward_model  = ForwardModel(state_dim, action_dim, hidden=64, lr=learning_rate)
        self.inverse_model  = InverseModel(state_dim, action_dim, hidden=64, lr=learning_rate)

        # Histórico de erros para interface com neuromodulação
        self._error_history: deque = deque(maxlen=200)
        self._observation_count = 0

        # Buffer de experiências para aprendizado
        self._experience_buffer: List[Dict] = []
        self._max_buffer = 1000

    # ─── Interface original (mantida para compatibilidade) ────────────────────

    def predict_timing(self, gesture: Dict) -> Dict[str, float]:
        """Prediz duração ideal para cada articulador no gesto atual."""
        result = {}
        for key in ["jaw_open", "lip_rounding", "velum", "vocal_fold_tension"]:
            if key in gesture:
                proto_key = key.split("_")[0]
                proto = GESTURE_TIMING_PROTOTYPES.get(proto_key,
                        GESTURE_TIMING_PROTOTYPES["jaw_open"])
                delta     = abs(gesture.get(key, 0.5) - 0.5) * 2
                predicted = proto["min_ms"] + delta * (proto["max_ms"] - proto["min_ms"])

                history = self.timing_predictions.get(key)
                if history:
                    predicted = predicted * 0.7 + np.mean(list(history)) * 0.3
                result[key + "_ms"] = float(predicted)

        if "tongue_dorsum_x" in gesture or "tongue_dorsum_y" in gesture:
            tx    = gesture.get("tongue_dorsum_x", 0.5)
            ty    = gesture.get("tongue_dorsum_y", 0.5)
            delta = (abs(tx - 0.5) + abs(ty - 0.5)) * 0.5
            proto = GESTURE_TIMING_PROTOTYPES["tongue"]
            predicted = proto["min_ms"] + delta * (proto["max_ms"] - proto["min_ms"])
            result["tongue_ms"] = float(predicted)

        result["burst_ms"]   = 20.0
        result["closure_ms"] = 80.0
        return result

    def predict_interval(self, prev_gesture: Dict, next_gesture: Dict) -> float:
        """Prediz intervalo entre dois gestos consecutivos em ms."""
        key = (prev_gesture.get("type", ""), next_gesture.get("type", ""))
        defaults = {
            ("plosive",   "vowel"):   20.0,
            ("fricative", "vowel"):   30.0,
            ("nasal",     "vowel"):   40.0,
            ("vowel",     "vowel"):   50.0,
            ("vowel",     "plosive"): 30.0,
            ("closure",   "burst"):    5.0,
            ("burst",     "vowel"):   10.0,
        }
        predicted = defaults.get(key, 40.0)
        history   = self.interval_predictions.get(key)
        if history:
            predicted = predicted * 0.6 + np.mean(list(history)) * 0.4
        return float(predicted)

    def correct_timing(self, actual_ms: float, predicted_ms: float) -> float:
        """Corrige timing via erro de predição (sinal de Purkinje)."""
        error      = actual_ms - predicted_ms
        correction = -self.lr * error
        return float(np.clip(correction, -50.0, 50.0))

    def learn(self, key: str, actual_duration_ms: float):
        """Atualiza predição de timing baseado no resultado real."""
        self.timing_predictions[key].append(actual_duration_ms)

    def learn_interval(self, prev_type: str, next_type: str, actual_interval_ms: float):
        """Atualiza predição de intervalo entre gestos."""
        self.interval_predictions[(prev_type, next_type)].append(actual_interval_ms)

    # ─── Novos métodos (Forward + Inverse Model) ──────────────────────────────

    def predict_consequence(self, state: np.ndarray,
                            action: np.ndarray) -> np.ndarray:
        """
        O que vai acontecer se eu fizer esta ação neste estado?

        Args:
            state:  estado atual (STATE_DIM,) ou array de sensores
            action: ação proposta (ACTION_DIM,)

        Returns:
            estado previsto após a ação (STATE_DIM,)
        """
        s = self._normalize_state(state)
        a = self._normalize_action(action)
        return self.forward_model.predict(s, a)

    def suggest_correction(self, current_state: np.ndarray,
                           desired_state: np.ndarray) -> np.ndarray:
        """
        Que ação devo fazer para sair do estado atual e chegar no desejado?

        Args:
            current_state: estado atual
            desired_state: estado que quero alcançar

        Returns:
            ação sugerida (ACTION_DIM,)
        """
        cs = self._normalize_state(current_state)
        ds = self._normalize_state(desired_state)
        return self.inverse_model.suggest_action(cs, ds)

    def observe_result(self, state: np.ndarray, action: np.ndarray,
                       actual_next_state: np.ndarray) -> float:
        """
        Aprende com o que realmente aconteceu após uma ação.
        Atualiza Forward Model e, com delay, Inverse Model.

        Returns:
            erro de predição (0.0 = perfeito, 1.0 = máxima surpresa)
        """
        s  = self._normalize_state(state)
        a  = self._normalize_action(action)
        ns = self._normalize_state(actual_next_state)

        # Atualiza Forward Model
        forward_loss = self.forward_model.update(s, a, ns)

        # Atualiza Inverse Model (estado → estado_next, ação real)
        inverse_loss = self.inverse_model.update(s, ns, a)

        # Normaliza erro para [0, 1]
        error = float(np.clip(forward_loss / 10.0, 0.0, 1.0))
        self._error_history.append(error)
        self._observation_count += 1

        # Salva experiência no buffer
        if len(self._experience_buffer) < self._max_buffer:
            self._experience_buffer.append({
                "state": s, "action": a, "next_state": ns, "error": error
            })

        return error

    def get_prediction_error(self) -> float:
        """
        Erro médio de predição recente.
        Alto → sistema está aprendendo (NA sobe no neuromodulador).
        Baixo → sistema domina o ambiente (estabilidade).
        """
        return self.forward_model.get_prediction_error()

    def should_correct(self, threshold: float = 0.3) -> bool:
        """
        Retorna True se o erro médio recente está acima do threshold.
        Indica que o cerebelo deve sugerir correção antes de executar.
        """
        return self.get_prediction_error() > threshold

    # ─── Utilitários ──────────────────────────────────────────────────────────

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Garante que state tem STATE_DIM e está normalizado."""
        s = np.asarray(state, dtype=np.float32).flatten()
        if len(s) < STATE_DIM:
            s = np.pad(s, (0, STATE_DIM - len(s)))
        return s[:STATE_DIM]

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Garante que action tem ACTION_DIM."""
        a = np.asarray(action, dtype=np.float32).flatten()
        if len(a) < ACTION_DIM:
            a = np.pad(a, (0, ACTION_DIM - len(a)))
        return a[:ACTION_DIM]

    def get_state(self) -> Dict:
        return {
            "timing_prototypes":    len(self.timing_predictions),
            "interval_prototypes":  len(self.interval_predictions),
            "forward_train_steps":  self.forward_model.train_steps,
            "inverse_train_steps":  self.inverse_model.train_steps,
            "prediction_error":     self.get_prediction_error(),
            "observations":         self._observation_count,
            "experience_buffer":    len(self._experience_buffer),
            "should_correct":       self.should_correct(),
        }


# ─── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cb  = Cerebellum(learning_rate=0.05)
    rng = np.random.default_rng(7)

    print("=== Cerebelo: Forward + Inverse Model ===\n")

    # Simula 100 experiências de queda (estado, ação, próximo estado)
    errors = []
    for i in range(100):
        # Estado: [pain, temp, pressure, px, py, pz, balance, gy, vx, vy, vz, ax, ay, az, circ]
        state  = rng.uniform(0, 1, STATE_DIM).astype(np.float32)
        action = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        # Próximo estado: determinístico + ruído pequeno
        next_s = np.clip(state + action[:STATE_DIM] * 0.1 + rng.normal(0, 0.01, STATE_DIM), 0, 1).astype(np.float32)
        err    = cb.observe_result(state, action, next_s)
        errors.append(err)

    print(f"Erro inicial (10 primeiras): {np.mean(errors[:10]):.4f}")
    print(f"Erro final (10 últimas):    {np.mean(errors[-10:]):.4f}")
    print(f"Deve_corrigir: {cb.should_correct()}")

    # Teste de predição de timing (compatibilidade)
    gesture = {"jaw_open": 0.8, "velum": 0.3, "vocal_fold_tension": 0.5}
    timing  = cb.predict_timing(gesture)
    print(f"\nTiming de gesto: {timing}")

    print(f"\nEstado do cerebelo: {cb.get_state()}")
