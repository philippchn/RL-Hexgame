"""
DQN-with-MCTS-at-inference agent for Hex.

This is a drop-in alternative to `facade_dqn.py` / `facade_dqn_minimax.py` that
keeps the *same trained DQN model* and the *same training procedure*, but
replaces the inference search:

    facade_dqn.py           greedy argmax Q     (no lookahead)
    facade_dqn_minimax.py   α-β minimax with Q  (depth-bounded full-width)
    facade_dqn_mcts.py      MCTS with Q          (asymmetric, scales with sims)

Why MCTS for a DQN
──────────────────
DQN gives action values Q(s, a) but no policy. To run MCTS we need both a
*prior* over moves (to focus selection) and a *leaf value*. Both come from
the Q-net:

    prior(s, a)  =  softmax( Q(s, ·) / T_prior )[a]      over legal moves
    V(s)         =  max_a  Q(s, a)                       over legal moves

PUCT then selects via  −Q_child + c · prior · √N_parent / (1 + N_child)
(child Q is from the *child's* player POV, so we negate).

vs. α-β: MCTS allocates more sims to promising lines automatically, and
scales smoothly with compute (more N_SIMS ⇒ stronger play). α-β has fixed
width — every move at every depth gets the same compute, regardless of how
promising it is.

vs. AlphaZero: AlphaZero has a *learned* policy head trained on MCTS visit
counts, which gives much sharper priors than softmax(Q). This file is a
retrofit for an existing DQN — strictly stronger than α-β at the same
wall-clock budget, but not as strong as a proper AlphaZero net.
"""

import math
import random

import torch

import hex_engine as engine

# Reuse model + training from the canonical DQN file. Loading & saving share
# the `dqn_NxN.pt` checkpoint, so the *same trained network* powers all three
# DQN variants — no retraining needed.
from submission import facade_dqn as _fd
from submission.facade_dqn import (
    HexDQN, EMPTY, RED, BLUE,
    _persp_board, _persp_action, _board_tensor, _valid_mask,
    _build_game, _get_device,
    train, save, load,
)

# ── hyperparameters ───────────────────────────────────────────────────────────
N_SIMS     = 200    # MCTS sims per move at inference
C_PUCT     = 1.5    # PUCT exploration constant
TEMP_PRIOR = 1.0    # softmax(Q / TEMP_PRIOR) temperature.  Lower = sharper priors.
                    # 0.5–1.0 works well for Hex if Q is well-trained.


# ── MCTS node ─────────────────────────────────────────────────────────────────

class _Node:
    """One node in the search tree; owns a copy of the game state it represents."""
    __slots__ = ("game", "parent", "move", "children", "N", "W", "Q", "P", "expanded")

    def __init__(self, game: engine.hexPosition,
                 parent: "_Node | None" = None,
                 move:   "tuple | None" = None,
                 prior:  float          = 0.0):
        self.game     = game
        self.parent   = parent
        self.move     = move
        self.children: dict[tuple, "_Node"] = {}
        self.N        = 0
        self.W        = 0.0
        self.Q        = 0.0
        self.P        = prior
        self.expanded = False


def _copy_game(game: engine.hexPosition) -> engine.hexPosition:
    g        = engine.hexPosition(game.size)
    g.board  = [row[:] for row in game.board]
    g.player = game.player
    g.winner = game.winner
    return g


# ── PUCT selection ────────────────────────────────────────────────────────────

def _puct(child: _Node, c: float) -> float:
    """
    PUCT score for a child node from the *parent's* (selector's) perspective.
    child.Q is stored from the child's player POV — negate so the parent
    prefers children where the *opponent* fares worst.
    """
    sqrt_n_parent = math.sqrt(child.parent.N) if child.parent else 1.0
    return -child.Q + c * child.P * sqrt_n_parent / (1 + child.N)


def _select_leaf(root: _Node, c: float) -> tuple[_Node, list[_Node]]:
    """Walk root → leaf via PUCT.  Returns (leaf, path)."""
    path: list[_Node] = []
    node = root
    while node.expanded and node.game.winner == EMPTY:
        path.append(node)
        node = max(node.children.values(), key=lambda ch: _puct(ch, c))
    path.append(node)
    return node, path


# ── expansion: priors and leaf value both from Q ──────────────────────────────

@torch.no_grad()
def _expand_with_q(node: _Node, net: HexDQN, device: torch.device) -> float:
    """
    Expand `node` and return the leaf value V(s) = max_a Q(s, a).

    The Q-net replaces both heads of an AlphaZero net:
      • priors[a]  = softmax( Q[a] / T )   masked to legal moves
      • V(s)       = max over legal Q values
    """
    state = _board_tensor(_persp_board(node.game), device).unsqueeze(0)
    q     = net(state).squeeze(0)
    mask  = _valid_mask(node.game, device)
    q_leg = q + mask                                      # −inf on occupied

    # Leaf value: best Q from current player's POV. Clip to [−1, +1] in case
    # the network overshoots the reward range.
    v = max(-1.0, min(1.0, q_leg.max().item()))

    # Priors: softmax over legal Q's; occupied → 0 via the −∞ mask.
    priors = torch.softmax(q_leg / TEMP_PRIOR, dim=0)

    for a in node.game.get_action_space():
        idx = node.game.coordinate_to_scalar(_persp_action(node.game, a))
        cg  = _copy_game(node.game)
        cg.move(a)
        node.children[a] = _Node(cg, parent=node, move=a, prior=priors[idx].item())
    node.expanded = True
    return v


# ── negamax backup ────────────────────────────────────────────────────────────

def _backup(path: list[_Node], v: float) -> None:
    """v is from the leaf's player POV; negate at each step toward the root."""
    for n in reversed(path):
        n.N += 1
        n.W += v
        n.Q  = n.W / n.N
        v = -v


# ── full MCTS pass ────────────────────────────────────────────────────────────

def _run_mcts(game: engine.hexPosition, net: HexDQN,
              n_sims: int, device: torch.device) -> dict[tuple, int]:
    """
    Run `n_sims` simulations from `game`. Returns {real_move: visit_count}.

    Single-sim sequential search (no virtual loss / batched eval).  At inference
    on a small HexDQN this is fine: 200 sims × ~1 ms each ≈ 0.2 s/move on CPU.
    For bigger budgets or larger boards, batched parallel selection (as in
    facade_alphazero.py) would speed things up.
    """
    net.eval()
    root = _Node(_copy_game(game))
    _expand_with_q(root, net, device)

    for _ in range(n_sims):
        leaf, path = _select_leaf(root, C_PUCT)
        if leaf.game.winner != EMPTY:
            v = -1.0                                       # current player just lost
        else:
            v = _expand_with_q(leaf, net, device)
        _backup(path, v)

    return {mv: ch.N for mv, ch in root.children.items()}


# ── public interface ──────────────────────────────────────────────────────────

def agent(board, action_set):
    """
    MCTS-DQN agent. Loads facade_dqn's `dqn_NxN.pt` checkpoint (trains on first
    call if absent), then picks the most-visited move after `N_SIMS` rollouts.
    """
    size = len(board)
    if _fd._network is None or _fd._trained_size != size:
        if not _fd.load(size):
            _fd.train(size=size)

    # First move on an empty board: pick a random interior cell (excludes the
    # outer ring of corner/edge openers, which are weak in Hex).
    if all(c == EMPTY for row in board for c in row):
        interior = [(r, c) for (r, c) in action_set
                    if 0 < r < size - 1 and 0 < c < size - 1]
        return random.choice(interior or action_set)

    net    = _fd._network
    game   = _build_game(board)
    device = _get_device()

    visits = _run_mcts(game, net, N_SIMS, device)
    return max(visits, key=visits.get)
