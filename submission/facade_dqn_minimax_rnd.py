"""
Deep Q-Network (DQN) agent for the Hex game.

Key components:
  - HexDQN       : CNN that maps a board state to Q-values for all cells
  - ReplayBuffer : experience replay with random mini-batch sampling
  - train()      : self-play training with a frozen target network
  - agent()      : greedy inference; auto-loads or trains on first call

Both players share one network via the perspective transformation —
each player always sees the board as if they were RED, so a single policy
covers both sides.
"""

import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import hex_engine as engine

EMPTY = 0
RED   = 1
BLUE  = -1

# ── hyperparameters ───────────────────────────────────────────────────────────
NUM_EPISODES    = 10_000
BATCH_SIZE      = 128
REPLAY_CAPACITY = 50_000   # larger buffer = more diverse, less correlated samples
LR              = 2.5e-4   # lower LR prevents runaway Q-value updates
GAMMA           = 0.95
EPSILON_START   = 1.0
EPSILON_END     = 0.05
TARGET_UPDATE   = 200      # less frequent sync = more stable bootstrap targets
EVAL_EVERY      = 500    # evaluate win rate vs random every N episodes

# Inference-time α-β search depth (plies). 0 = greedy Q (original behaviour).
# 2 = fast (<0.5 s/move on 7×7). 3 = ~1–3 s/move, strong. 4 = ~10–30 s/move.
SEARCH_DEPTH    = 3

# ── device ────────────────────────────────────────────────────────────────────
_device: torch.device | None = None

def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] device: {_device}")
    return _device


# ── Q-network ─────────────────────────────────────────────────────────────────

class HexDQN(nn.Module):
    """
    Input : (B, 2, size, size)
              channel 0 = current player's stones
              channel 1 = opponent's stones
    Output: (B, size²)  — one Q-value per board cell
    """
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.conv = nn.Sequential(
            nn.Conv2d(2,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * size * size, 128), nn.ReLU(),
            nn.Linear(128, size * size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ── experience replay ─────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: int, r: float,
             s_next: torch.Tensor, done: bool) -> None:
        self.buf.append((s, a, r, s_next, done))

    def sample(self, n: int):
        s, a, r, s_next, done = zip(*random.sample(self.buf, n))
        return (
            torch.stack(s),
            torch.tensor(a,    dtype=torch.long),
            torch.tensor(r,    dtype=torch.float32),
            torch.stack(s_next),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ── perspective & tensor helpers ──────────────────────────────────────────────

def _persp_board(game: engine.hexPosition):
    return game.board if game.player == RED else game.recode_blue_as_red()


def _persp_action(game: engine.hexPosition, action: tuple) -> tuple:
    """Map a coordinate to/from perspective space (transformation is self-inverse)."""
    return action if game.player == RED else game.recode_coordinates(action)


def _board_tensor(board, device: torch.device) -> torch.Tensor:
    """Board → (2, size, size) float tensor on *device*."""
    t = torch.tensor(board, dtype=torch.float32, device=device)
    return torch.stack([(t == RED).float(), (t == BLUE).float()])


def _valid_mask(game: engine.hexPosition, device: torch.device) -> torch.Tensor:
    """
    Additive mask for action selection: 0.0 at valid perspective-space indices,
    -inf everywhere else. Apply as  (q + mask).argmax()  to block illegal cells.
    """
    mask = torch.full((game.size * game.size,), float("-inf"), device=device)
    for a in game.get_action_space():
        pa = _persp_action(game, a)
        mask[game.coordinate_to_scalar(pa)] = 0.0
    return mask


def _pick_action(net: HexDQN, s_t: torch.Tensor,
                 game: engine.hexPosition, epsilon: float,
                 device: torch.device) -> int:
    """ε-greedy action selection; returns a perspective-space scalar."""
    if random.random() < epsilon:
        a = random.choice(game.get_action_space())
        return game.coordinate_to_scalar(_persp_action(game, a))
    with torch.no_grad():
        q    = net(s_t.unsqueeze(0)).squeeze(0)
        mask = _valid_mask(game, device)
        return int((q + mask).argmax())


# ── evaluation ───────────────────────────────────────────────────────────────

def _eval_win_rate(net: HexDQN, size: int,
                   n_games: int = 30, device: "torch.device | None" = None) -> float:
    """
    Play *n_games* greedy games against a random opponent, alternating colors.
    Returns win fraction (0–1). Does not affect training state.
    """
    if device is None:
        device = _get_device()
    net.eval()
    wins = 0
    for i in range(n_games):
        game     = engine.hexPosition(size)
        q_color  = RED if i % 2 == 0 else BLUE
        while game.winner == EMPTY:
            if game.player == q_color:
                s_t  = _board_tensor(_persp_board(game), device)
                mask = _valid_mask(game, device)
                with torch.no_grad():
                    q = net(s_t.unsqueeze(0)).squeeze(0)
                a_idx = int((q + mask).argmax())
                real  = _persp_action(game, game.scalar_to_coordinates(a_idx))
                game.move(real)
            else:
                game.move(random.choice(game.get_action_space()))
        if game.winner == q_color:
            wins += 1
    net.train()
    return wins / n_games


# ── persistence ───────────────────────────────────────────────────────────────

def _model_path(size: int) -> str:
    return os.path.join(os.path.dirname(__file__), f"dqn_minimax_{size}x{size}.pt")


def save(net: HexDQN, size: int) -> None:
    path = _model_path(size)
    torch.save({"size": size, "state_dict": net.state_dict()}, path)
    print(f"[DQN] Saved → {path}")


def load(size: int) -> bool:
    """Load model for *size* into _network. Returns True if file found."""
    global _network, _trained_size
    path = _model_path(size)
    if not os.path.exists(path):
        return False
    device = _get_device()
    try:
        data = torch.load(path, map_location=device, weights_only=True)
    except TypeError:                           # weights_only not in PyTorch <2
        data = torch.load(path, map_location=device)
    net = HexDQN(size).to(device)
    net.load_state_dict(data["state_dict"])
    net.eval()
    _network      = net
    _trained_size = size
    print(f"[DQN] Loaded ← {path}")
    return True


# ── training ──────────────────────────────────────────────────────────────────

_network:      HexDQN | None = None
_trained_size: int           = 0


def train(size: int = 7, episodes: int = NUM_EPISODES) -> dict:
    """
    Train via self-play.  Returns a metrics dict with keys:
        "losses"    : list[float]  — Huber loss per gradient update
        "win_rates" : list[tuple]  — (episode, win_rate) evaluated every EVAL_EVERY episodes

    Transition storage: each player's (state, action) pairs are collected
    per episode. At episode end, same-player consecutive turns are linked as
    (s_t, a_t, 0, s_{t+2}, False), and the final turn receives (s_T, a_T, ±1, 0, True).
    This treats the opponent's response as part of the environment.
    """
    global _network, _trained_size

    device  = _get_device()
    net     = HexDQN(size).to(device)
    target  = HexDQN(size).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()

    opt    = torch.optim.Adam(net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    losses:      list[float] = []
    win_history: list[tuple] = []

    print(f"[DQN] Training {episodes:,} episodes on {size}×{size} …")
    print(f"{'ep':>8}  {'ε':>6}  {'loss':>8}  {'win% vs rand':>12}")
    print("-" * 44)

    for ep in range(episodes):
        game    = engine.hexPosition(size)
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * ep / episodes

        traj: dict[int, list] = {RED: [], BLUE: []}

        # ── play one self-play game ───────────────────────────────────────────
        while game.winner == EMPTY:
            player = game.player
            s_t    = _board_tensor(_persp_board(game), device)
            a_idx  = _pick_action(net, s_t, game, epsilon, device)

            # perspective scalar → real board coordinate (self-inverse)
            real = _persp_action(game, game.scalar_to_coordinates(a_idx))

            traj[player].append((s_t, a_idx))
            game.move(real)

        # ── push transitions: reward only at the final step ───────────────────
        zero   = torch.zeros(2, size, size, device=device)
        winner = game.winner

        for player, steps in traj.items():
            outcome = 1.0 if player == winner else -1.0
            n       = len(steps)
            for k, (s, a) in enumerate(steps):
                if k < n - 1:
                    s_next, _ = steps[k + 1]   # same player's next turn
                    replay.push(s, a, 0.0, s_next, False)
                else:
                    replay.push(s, a, outcome, zero, True)

        # ── gradient update ───────────────────────────────────────────────────
        if len(replay) >= BATCH_SIZE:
            s_b, a_b, r_b, sn_b, done_b = replay.sample(BATCH_SIZE)
            s_b     = s_b.to(device)
            sn_b    = sn_b.to(device)
            r_b     = r_b.to(device)
            done_b  = done_b.to(device)

            # Q(s, a) for actions actually taken
            q_pred = net(s_b).gather(1, a_b.to(device).unsqueeze(1)).squeeze(1)

            # Double DQN target with valid-action masking
            #
            # Vanilla DQN:   argmax taken over ALL cells → occupied cells can
            # have large Q-values (no gradient forcing them down) → targets are
            # overestimated → Q-values diverge exponentially.
            #
            # Fix 1 – valid mask: derive empty cells directly from the stored
            # tensor (ch0 = current player, ch1 = opponent; occupied iff either
            # channel is 1).  Apply −∞ to occupied indices so argmax can only
            # pick a legal next move.
            #
            # Fix 2 – Double DQN: main network *selects* the action (with mask),
            # target network only *evaluates* it.  This decouples selection from
            # evaluation and reduces overestimation bias.
            with torch.no_grad():
                occupied  = (sn_b[:, 0] + sn_b[:, 1]).view(sn_b.shape[0], -1) > 0
                next_mask = torch.zeros(occupied.shape, device=device)
                next_mask[occupied] = float("-inf")

                a_next    = (net(sn_b) + next_mask).argmax(1)          # main selects
                q_next    = target(sn_b).gather(1, a_next.unsqueeze(1)).squeeze(1)
                q_target  = r_b + GAMMA * q_next * (1.0 - done_b)      # target evaluates

            loss = F.smooth_l1_loss(q_pred, q_target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # ── periodically sync target network ─────────────────────────────────
        if (ep + 1) % TARGET_UPDATE == 0:
            target.load_state_dict(net.state_dict())

        # ── periodic evaluation ───────────────────────────────────────────────
        if (ep + 1) % EVAL_EVERY == 0:
            win_rate = _eval_win_rate(net, size, n_games=30, device=device)
            win_history.append((ep + 1, win_rate))
            recent = losses[-(EVAL_EVERY):] if losses else [0.0]
            avg_loss = sum(recent) / len(recent)
            pct = (ep + 1) / episodes * 100
            bar = "█" * int(win_rate * 20) + "░" * (20 - int(win_rate * 20))
            print(f"{ep+1:>8,}  {epsilon:>6.3f}  {avg_loss:>8.4f}  "
                  f"{win_rate*100:>5.1f}%  {bar}  ({pct:.0f}%)")

    net.eval()
    _network      = net
    _trained_size = size
    save(net, size)
    return {"losses": losses, "win_rates": win_history, "size": size}


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_metrics(metrics: dict, smoothing: int = 200) -> None:
    """
    Visualise training progress from the dict returned by train().

    Two subplots:
      top    — Huber loss per gradient update (raw + smoothed moving average)
      bottom — win rate vs random opponent, sampled every EVAL_EVERY episodes

    Usage:
        metrics = train(size=5, episodes=5_000)
        plot_metrics(metrics)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[DQN] pip install matplotlib numpy  to enable plotting")
        return

    losses    = metrics.get("losses", [])
    win_rates = metrics.get("win_rates", [])   # [(episode, wr), …]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(f"DQN training — board {metrics.get('size', '?')}×{metrics.get('size', '?')}")

    # ── loss ─────────────────────────────────────────────────────────────────
    if losses:
        xs = list(range(len(losses)))
        ax1.plot(xs, losses, alpha=0.2, color="steelblue", linewidth=0.8)
        if len(losses) >= smoothing:
            kernel = np.ones(smoothing) / smoothing
            smooth = np.convolve(losses, kernel, mode="valid")
            ax1.plot(range(smoothing - 1, len(losses)), smooth,
                     color="steelblue", linewidth=2,
                     label=f"moving avg (window={smoothing})")
            ax1.legend()
    ax1.set_xlabel("Gradient update")
    ax1.set_ylabel("Huber loss")
    ax1.set_title("Training loss")
    ax1.grid(alpha=0.3)

    # ── win rate ──────────────────────────────────────────────────────────────
    if win_rates:
        eps, wrs = zip(*win_rates)
        ax2.plot(eps, [w * 100 for w in wrs],
                 marker="o", color="darkorange", linewidth=2)
        ax2.axhline(50, color="gray", linestyle="--", linewidth=1,
                    label="random baseline (50 %)")
        ax2.fill_between(eps, 50, [w * 100 for w in wrs],
                         where=[w >= 0.5 for w in wrs],
                         alpha=0.15, color="green", label="beating random")
        ax2.fill_between(eps, 50, [w * 100 for w in wrs],
                         where=[w < 0.5 for w in wrs],
                         alpha=0.15, color="red",   label="below random")
        ax2.set_ylim(0, 105)
        ax2.legend()
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win rate vs random (%)")
    ax2.set_title("Win rate vs random opponent (greedy policy, 30 games)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── inference-time α-β search ────────────────────────────────────────────────
#
# Pure DQN picks the move with the highest Q-value and stops. That has no
# lookahead, which is why it can crush a random opponent but lose to a human
# who plans two moves ahead. We wrap the trained Q-network in a negamax α-β
# search, using the network as the leaf evaluator:
#
#     V(s)  ≈  max_a  Q(s, a)        over legal actions
#
# Move-ordering by Q gives near-optimal α-β cutoffs (β-cuts on the first child
# of most nodes), so depth 3 is cheap even on a small CNN.

def _make_move(game: engine.hexPosition, action: tuple):
    """In-place move that skips the history deepcopy in game.move().
    Returns the state needed by _undo_move to restore the previous position."""
    r, c              = action
    prev_player       = game.player
    prev_winner       = game.winner
    game.board[r][c]  = prev_player
    game.player       = -prev_player
    game.evaluate()                 # updates game.winner
    return prev_player, prev_winner


def _undo_move(game: engine.hexPosition, action: tuple,
               prev_player: int, prev_winner: int) -> None:
    r, c             = action
    game.board[r][c] = EMPTY
    game.player      = prev_player
    game.winner      = prev_winner


def _q_at(q: torch.Tensor, game: engine.hexPosition, action: tuple) -> float:
    """Q-value for `action` from current player's perspective."""
    return q[game.coordinate_to_scalar(_persp_action(game, action))].item()


def _alphabeta(game: engine.hexPosition, depth: int,
               alpha: float, beta: float,
               net: HexDQN, device: torch.device) -> float:
    """
    Negamax α-β. Returns position value from `game.player`'s POV
    (the player to move). +1 = winning, -1 = losing.

    game.move() switches the player *before* setting `winner`, so when
    `winner != EMPTY` here, `game.player` is the loser → return -1.
    """
    if game.winner != EMPTY:
        return -1.0

    actions = game.get_action_space()
    if not actions:
        return 0.0

    s_t = _board_tensor(_persp_board(game), device).unsqueeze(0)
    with torch.no_grad():
        q = net(s_t).squeeze(0)

    if depth == 0:
        # Leaf: best legal Q from current player's POV
        return max(_q_at(q, game, a) for a in actions)

    # Order moves by Q (best-first) for tight α-β pruning
    actions.sort(key=lambda a: _q_at(q, game, a), reverse=True)

    best = -float("inf")
    for a in actions:
        prev_p, prev_w = _make_move(game, a)
        val = -_alphabeta(game, depth - 1, -beta, -alpha, net, device)
        _undo_move(game, a, prev_p, prev_w)

        if val > best:  best  = val
        if best > alpha: alpha = best
        if alpha >= beta:                 # β-cut
            break
    return best


def _search_best_action(game: engine.hexPosition, depth: int,
                        net: HexDQN, device: torch.device) -> tuple:
    """Root α-β. Returns the real-coordinate action with the highest score."""
    actions = game.get_action_space()
    s_t = _board_tensor(_persp_board(game), device).unsqueeze(0)
    with torch.no_grad():
        q = net(s_t).squeeze(0)

    actions.sort(key=lambda a: _q_at(q, game, a), reverse=True)

    best_action, best_score = actions[0], -float("inf")
    alpha = -float("inf")
    for a in actions:
        prev_p, prev_w = _make_move(game, a)
        val = -_alphabeta(game, depth - 1, -float("inf"), -alpha, net, device)
        _undo_move(game, a, prev_p, prev_w)

        if val > best_score:
            best_score, best_action = val, a
        if val > alpha:
            alpha = val
    return best_action


# ── public interface ──────────────────────────────────────────────────────────

def _build_game(board) -> engine.hexPosition:
    size        = len(board)
    game        = engine.hexPosition(size)
    game.board  = [row[:] for row in board]
    red_n       = sum(c == RED  for row in board for c in row)
    blue_n      = sum(c == BLUE for row in board for c in row)
    game.player = RED if red_n == blue_n else BLUE
    return game


def agent(board, action_set):
    global _network, _trained_size

    size = len(board)
    if _network is None or _trained_size != size:
        if not load(size):
            train(size=size)

    # First move on an empty board: pick a random interior cell (excludes the
    # outer ring of corner/edge openers, which are weak in Hex).
    if all(c == EMPTY for row in board for c in row):
        interior = [(r, c) for (r, c) in action_set
                    if 0 < r < size - 1 and 0 < c < size - 1]
        return random.choice(interior or action_set)

    game   = _build_game(board)
    device = _get_device()

    # SEARCH_DEPTH = 0 → original greedy-Q behaviour. Otherwise α-β minimax
    # with the Q-network as leaf evaluator (adds the lookahead DQN lacks).
    if SEARCH_DEPTH <= 0:
        s_t = _board_tensor(_persp_board(game), device).unsqueeze(0)
        with torch.no_grad():
            q = _network(s_t).squeeze(0)
        return max(action_set,
                   key=lambda a: q[game.coordinate_to_scalar(_persp_action(game, a))].item())

    return _search_best_action(game, SEARCH_DEPTH, _network, device)
