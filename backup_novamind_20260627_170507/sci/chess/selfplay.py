"""
selfplay.py — XADREZ SELF-PLAY: aprende uma SKILL real a partir de GROUND-TRUTH GRATIS (o resultado da partida),
sem rotulo humano. Demonstra o loop "agir -> consequencia -> aprender" e escreve experiencia no ExperienceStore
(xadrez = 2o dominio na memoria cross-dominio). Honesto: rede de VALOR + busca 1-ply (nao AlphaZero/MCTS); um
demonstrador do loop, nao um motor forte. Metrica: win-rate vs jogador ALEATORIO sobe ao longo do treino.
"""
import os, sys, random, torch, torch.nn as nn, torch.nn.functional as F, chess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.cross_domain import ExperienceStore

torch.manual_seed(0); random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STORE = ExperienceStore()                      # memoria compartilhada (mesma do cross-dominio)

def encode(board):
    """canonicaliza p/ a perspectiva de quem joga (sempre 'brancas') -> 12 planos 8x8."""
    b = board if board.turn == chess.WHITE else board.mirror()
    t = torch.zeros(12, 8, 8)
    for sq, pc in b.piece_map().items():
        plane = (pc.piece_type - 1) + (0 if pc.color == chess.WHITE else 6)
        t[plane, sq // 8, sq % 8] = 1.0
    return t

class ValueNet(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(12, ch, 3, 1, 1), nn.ReLU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU())
        s.f = nn.Sequential(nn.Flatten(), nn.Linear(ch * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())
    def forward(s, x): return s.f(s.c(x)).squeeze(-1)   # valor em [-1,1] p/ quem joga

net = ValueNet().to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)

@torch.no_grad()
def pick_move(board, eps=0.0):
    """1-ply: avalia cada lance pelo VALOR do oponente apos o lance (= -nosso valor); escolhe o melhor."""
    moves = list(board.legal_moves)
    if not moves: return None
    if random.random() < eps: return random.choice(moves)
    after = []
    for mv in moves:
        board.push(mv); after.append(encode(board)); board.pop()
    v_opp = net(torch.stack(after).to(DEV))         # valor do oponente apos cada lance
    return moves[int(torch.argmin(v_opp).item())]   # minimiza valor do oponente = maximiza o nosso

def play_game(eps, max_plies=60):
    board = chess.Board(); states, turns = [], []
    for _ in range(max_plies):
        if board.is_game_over(): break
        states.append(encode(board)); turns.append(board.turn)
        mv = pick_move(board, eps)
        if mv is None: break
        board.push(mv)
    # ground-truth: resultado real; se cortou, adjudica por material (proxy honesto)
    res = board.result(claim_draw=True)
    if res == '1-0': white = 1.0
    elif res == '0-1': white = -1.0
    elif res == '1/2-1/2': white = 0.0
    else:
        val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        m = sum((1 if p.color == chess.WHITE else -1) * val[p.piece_type] for p in board.piece_map().values())
        white = max(-1.0, min(1.0, m / 10.0))
    for st, turn in zip(states, turns):
        z = white if turn == chess.WHITE else -white      # resultado da perspectiva de quem jogava
        STORE.add("chess", st, 0, z)                       # escreve no store (memoria cross-dominio)
    return [(st, (white if t == chess.WHITE else -white)) for st, t in zip(states, turns)]

@torch.no_grad()
def winrate_vs_random(games=40):
    w = d = 0
    for g in range(games):
        board = chess.Board(); ours = chess.WHITE if g % 2 == 0 else chess.BLACK
        for _ in range(80):
            if board.is_game_over(): break
            mv = pick_move(board, 0.0) if board.turn == ours else random.choice(list(board.legal_moves))
            board.push(mv)
        r = board.result(claim_draw=True)
        if r == '1/2-1/2' or r == '*':
            val = {chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:0}
            m = sum((1 if p.color==chess.WHITE else -1)*val[p.piece_type] for p in board.piece_map().values())
            won = (m > 0) == (ours == chess.WHITE) and m != 0
            if m == 0: d += 1
            elif won: w += 1
        else:
            won = (r == '1-0') == (ours == chess.WHITE)
            if won: w += 1
    return w / games, d / games

if __name__ == "__main__":
    ITERS, GAMES_PER = 5, 80
    print(f"[{DEV}] xadrez self-play — win-rate vs ALEATORIO deve subir:", flush=True)
    wr0, _ = winrate_vs_random(); print(f"  inicio (rede nao treinada): win-rate vs random {wr0:.0%}", flush=True)
    for it in range(ITERS):
        data = []
        for _ in range(GAMES_PER): data += play_game(eps=max(0.1, 0.5 - it * 0.1))
        random.shuffle(data)
        S = torch.stack([d[0] for d in data]).to(DEV); Z = torch.tensor([d[1] for d in data], device=DEV)
        for _ in range(6):                                  # treina a rede de valor no ground-truth dos jogos
            idx = torch.randperm(len(Z), device=DEV)
            for b in range(0, len(Z), 256):
                j = idx[b:b + 256]; loss = F.mse_loss(net(S[j]), Z[j])
                opt.zero_grad(); loss.backward(); opt.step()
        wr, dr = winrate_vs_random()
        print(f"  iter {it+1}/{ITERS}: {len(data)} posicoes | loss {loss.item():.3f} | win-rate vs random {wr:.0%} (empates {dr:.0%}) | store={len(STORE)}", flush=True)
    print(f"\nRESULTADO: win-rate vs random {wr0:.0%} -> {wr:.0%}. {'APRENDEU a jogar (skill de ground-truth gratis)' if wr-wr0>0.15 else 'pouco ganho'}", flush=True)
