"""ARC-AGI-3 benchmark harness — Fase 0 do plano de fechamento do loop empirico.

Roda um agente sobre um conjunto FIXO e determinístico de jogos públicos,
100% offline (sem chave de API, sem rede — arc-agi OperationMode.OFFLINE
lendo os jogos comunitarios em agent3/environment_files/), com orçamento
fixo de ações, e registra o score OFICIAL do SDK (mesma fórmula RHAE-style
usada na competição: score = (baseline_actions/actions_taken)^2 * 100,
capped) em agent3/results/history.csv.

Nada é interpretado como "evidência de AGI" até que este número exista
para pelo menos os baselines random e greedy — ver Fase 0 do plano.

Uso:
    python agent3/bench.py --agent random  --games 10 --seed 0 --budget 300
    python agent3/bench.py --agent greedy  --games 10 --seed 0 --budget 300
    python agent3/bench.py --list
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE / "environment_files"
RESULTS_DIR = HERE / "results"
RESULTS_CSV = RESULTS_DIR / "history.csv"
TRACE_DIR = HERE / "results" / "traces"

ACTION_BY_ID = {a.value: a for a in GameAction}
NON_RESET_ACTION_IDS = [a.value for a in GameAction if a != GameAction.RESET]

DEFAULT_N_GAMES = 10
DEFAULT_BUDGET = 300

CSV_FIELDS = [
    "timestamp", "git_sha", "agent", "game_id", "seed", "budget",
    "actions_used", "levels_completed", "won", "final_state", "official_score",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_arcade() -> Arcade:
    for name in ("arc_agi", "arc_agi.base", "arc_agi.scorecard"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    quiet_logger = logging.getLogger("agent3.bench.arcade")
    quiet_logger.setLevel(logging.WARNING)
    quiet_logger.addHandler(logging.NullHandler())
    return Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
        logger=quiet_logger,
    )


def default_game_ids(arc: Arcade, n: int) -> list[str]:
    """Selecao determinista: ordem alfabetica por game_id, primeiros N.

    Fixo entre agentes e entre execucoes para que a comparacao no CSV seja
    valida (mesmo conjunto de jogos, mesmo orcamento, mesma seed).
    """
    ids = sorted(e.game_id for e in arc.available_environments)
    return ids[:n]


def _frame_hash(frame) -> int:
    try:
        return hash(tuple(tuple(tuple(row) for row in layer) for layer in frame.frame))
    except Exception:
        return 0


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, frame) -> tuple[int, dict]:
        avail = list(frame.available_actions) or [1]
        aid = self.rng.choice(avail)
        data = {}
        if aid == GameAction.ACTION6.value:
            data = {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}
        return aid, data


class GreedyAgent:
    """Repete a ultima acao se ela mudou o frame; senao cicla para a proxima
    acao disponivel. Baseline forte e barato — e o que a Fase 0 do plano
    pede para bater antes de qualquer coisa contar como progresso."""

    name = "greedy"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.last_action: Optional[int] = None
        self.last_data: dict = {}
        self.prev_hash: Optional[int] = None
        self._cycle_idx = 0

    def act(self, frame) -> tuple[int, dict]:
        avail = list(frame.available_actions) or [1]
        cur_hash = _frame_hash(frame)

        if (
            self.last_action is not None
            and self.prev_hash is not None
            and cur_hash != self.prev_hash
            and self.last_action in avail
        ):
            aid, data = self.last_action, self.last_data
        else:
            aid = avail[self._cycle_idx % len(avail)]
            self._cycle_idx += 1
            data = {}
            if aid == GameAction.ACTION6.value:
                data = {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}

        self.last_action = aid
        self.last_data = data
        self.prev_hash = cur_hash
        return aid, data


AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
}


def load_novamind_agent():
    from agent3.runner import NovaMindAgent  # local import: pacote pode nao existir ainda
    return NovaMindAgent


def run_game(arc: Arcade, agent, game_id: str, seed: int, budget: int,
             scorecard_id: str, trace_path: Optional[Path] = None) -> Optional[dict]:
    env = arc.make(game_id, seed=seed, scorecard_id=scorecard_id)
    if env is None:
        return None

    frame = env.reset()
    actions_used = 0
    trace_rows = []

    while actions_used < budget and frame.state not in (GameState.WIN, GameState.GAME_OVER):
        aid, data = agent.act(frame)
        action_obj = ACTION_BY_ID[aid]
        new_frame = env.step(action_obj, data=data or None)
        if new_frame is None:
            break
        actions_used += 1
        if trace_path is not None:
            trace_rows.append({
                "step": actions_used,
                "action": aid,
                "data": data,
                "state": new_frame.state.name,
                "levels_completed": new_frame.levels_completed,
            })
        frame = new_frame

    if hasattr(agent, "observe_terminal"):
        # fecha o credito/penalidade da ULTIMA acao do episodio — o loop
        # acima para assim que WIN/GAME_OVER chega, sem chamar agent.act()
        # de novo, entao sem este gancho essa acao nunca seria aprendida.
        agent.observe_terminal(frame)

    if trace_path is not None:
        import json
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            for row in trace_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "actions_used": actions_used,
        "levels_completed": frame.levels_completed,
        "won": frame.state == GameState.WIN,
        "final_state": frame.state.name,
    }


def append_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", choices=["random", "greedy", "novamind"], default="random")
    ap.add_argument("--games", type=int, default=DEFAULT_N_GAMES,
                     help="numero de jogos publicos (selecao deterministica, alfabetica)")
    ap.add_argument("--game-ids", type=str, default=None,
                     help="lista explicita de game_ids separados por virgula (sobrepoe --games)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                     help="orcamento maximo de acoes por jogo")
    ap.add_argument("--trace", action="store_true",
                     help="grava trace JSONL por jogo em agent3/results/traces/")
    ap.add_argument("--list", action="store_true", help="lista os jogos publicos disponiveis e sai")
    args = ap.parse_args()

    arc = make_arcade()

    if args.list:
        for e in sorted(arc.available_environments, key=lambda x: x.game_id):
            print(e.game_id, "|", e.title, "|", e.tags)
        print(f"\ntotal: {len(arc.available_environments)} jogos")
        return

    game_ids = (
        [g.strip() for g in args.game_ids.split(",")]
        if args.game_ids
        else default_game_ids(arc, args.games)
    )

    if args.agent == "novamind":
        agent = load_novamind_agent()(seed=args.seed)
    else:
        agent = AGENTS[args.agent](seed=args.seed)

    sha = git_sha()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    scorecard_id = arc.create_scorecard(tags=[f"bench:{args.agent}"])

    csv_rows = []
    print(f"=== bench: agent={args.agent} games={len(game_ids)} seed={args.seed} budget={args.budget} sha={sha} ===")
    for game_id in game_ids:
        trace_path = TRACE_DIR / f"{args.agent}_{game_id}_seed{args.seed}.jsonl" if args.trace else None
        result = run_game(arc, agent, game_id, args.seed, args.budget, scorecard_id, trace_path)
        if result is None:
            print(f"  {game_id}: FALHOU AO CARREGAR")
            continue
        print(f"  {game_id}: actions={result['actions_used']:4d} "
              f"levels={result['levels_completed']} won={result['won']} state={result['final_state']}")
        csv_rows.append({
            "timestamp": ts, "git_sha": sha, "agent": args.agent, "game_id": game_id,
            "seed": args.seed, "budget": args.budget,
            "actions_used": result["actions_used"],
            "levels_completed": result["levels_completed"],
            "won": result["won"], "final_state": result["final_state"],
            "official_score": None,  # preenchido apos fechar o scorecard
        })
        # reinicializa o agente por jogo (estado de greedy/novamind e local a um jogo)
        if args.agent == "novamind":
            agent = load_novamind_agent()(seed=args.seed)
        else:
            agent = AGENTS[args.agent](seed=args.seed)

    scorecard = arc.close_scorecard(scorecard_id)
    if scorecard is not None:
        for row in csv_rows:
            env_score = scorecard.find_environment(row["game_id"])
            row["official_score"] = round(env_score.score, 2) if env_score else None
        print(f"\nscore oficial agregado (scorecard): {scorecard.score:.2f}")
        print(f"niveis completados: {scorecard.total_levels_completed}/{scorecard.total_levels}")
        print(f"acoes totais: {scorecard.total_actions}")

    append_csv(csv_rows)
    print(f"\n{len(csv_rows)} linhas anexadas em {RESULTS_CSV}")


if __name__ == "__main__":
    main()
