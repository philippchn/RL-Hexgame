"""
AlphaZero-style agent for the Hex game.

AlphaZero vs PPO
────────────────
  PPO         learns policy + value directly from self-play outcomes.
              A single forward pass decides each move.
  AlphaZero   MCTS guided by a neural network.  At every move the agent
              runs N simulations, each of which expands the search tree
              one step using the network for priors and leaf evaluation.
              The resulting visit-count distribution π is a *stronger*
              policy target than the raw network output — the training
              signal therefore gets better as the network improves.
              This self-improving loop is what gives AlphaZero its strength.

Algorithm per iteration
───────────────────────
  1. Self-play  (generate training data)
     For each move in each game:
       a. Run N MCTS simulations from the current position.
          Selection : PUCT score = Q + c·P·√N_parent/(1+N_child)
          Expansion : evaluate leaf with network → priors p, value v
          Backup    : propagate v up the tree, negating at each level
                      (negamax: the player above sees the opposite value)
       b. π = softmax of visit counts (with temperature τ)
       c. Sample move from π; store (state, π, player)
     After the game ends, label each stored state with z = ±1 (did
     the player who moved at that state eventually win?).

  2. Train on a replay buffer of recent (state, π, z) tuples:
       L = cross_entropy(p_net, π)  +  MSE(v_net, z)  +  L2_reg

  3. Repeat: the stronger network guides better MCTS, which produces
     better training targets, which strengthen the network further.

Terminal value convention
─────────────────────────
  In Hex, whenever a node is terminal the *current player has just lost*
  (the previous player placed the winning stone).  So v = −1 at any
  terminal leaf, which propagates correctly under negamax.
"""

import os
import random
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as tmp
import numpy as np

# Workers ship many small tensors through Queues. The default 'file_descriptor'
# sharing strategy opens a new fd per tensor and exhausts the OS limit
# ("Too many open files"). 'file_system' uses shared-memory files instead —
# slightly slower per send, but no fd leak.
try:
    tmp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# Defensive: raise the soft fd limit to the hard cap so even brief fd spikes
# don't trip us up. No-op on Windows.
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except (ImportError, ValueError, OSError):
    pass

import hex_engine as engine

# Try to load the C++ MCTS extension (build with: python setup_mcts.py build_ext --inplace)
# Falls back to pure Python if the .so is not present.
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import hex_mcts as _hex_mcts
    print("[AlphaZero] C++ MCTS extension loaded ✓")
except ImportError:
    _hex_mcts = None
    print("[AlphaZero] C++ MCTS extension not found — using pure Python (run: python setup_mcts.py build_ext --inplace)")

EMPTY = 0
RED   = 1
BLUE  = -1

# ── hyperparameters ───────────────────────────────────────────────────────────
N_SIMULATIONS  = 100     # MCTS simulations per move during self-play
INFERENCE_SIMS = 200     # sims per move at inference (agent vs human) — no retraining needed
PARALLEL_SIMS  = 8       # leaves per batched GPU forward pass — must be << N_SIMULATIONS
                         # Rule: parallel ≤ sims/7  (otherwise tree search is only 1 level deep)
C_PUCT         = 1.5     # exploration constant in PUCT formula
VIRTUAL_LOSS   = 1       # pessimistic penalty applied before a leaf is evaluated,
                         # forcing parallel selections down different tree branches
DIR_ALPHA      = 0.3     # Dirichlet noise shape parameter  (root exploration)
DIR_EPS        = 0.25    # fraction of Dirichlet noise mixed into root priors
TEMP           = 1.0     # action-selection temperature (first TEMP_CUTOFF moves)
TEMP_CUTOFF    = 10      # switch to greedy after this many moves per game

GAMES_PER_ITER = 16      # self-play games per training iteration
TRAIN_EPOCHS   = 5       # gradient-update epochs per iteration
BATCH_SIZE     = 128     # scaled up automatically for larger boards (see train())
LR             = 1e-3
L2_REG         = 1e-4    # weight decay (L2 regularisation)
GRAD_CLIP      = 1.0
REPLAY_SIZE    = 20_000  # baseline capacity; scaled up automatically for larger boards

N_ITERATIONS   = 200     # total training iterations
EVAL_EVERY     = 20      # evaluate against random every N iterations

# Network size — override via train(channels=..., blocks=...)
NET_CHANNELS = 128       # residual feature channels
NET_BLOCKS   = 5         # number of residual blocks

# ── recent improvements ───────────────────────────────────────────────────────
# 1. Symmetry augmentation. Hex on a rhombus has one non-trivial symmetry: 180°
#    rotation around the center (under the perspective frame, RED's goal of
#    left↔right is preserved). Doubles training data for free.
SYMMETRY_AUG     = True

# 2. Tree reuse across moves. After choosing a move, instead of throwing away
#    the MCTS tree and rebuilding from scratch, keep the chosen child's subtree
#    as the new root. Cuts effective sim cost roughly in half once games are
#    deep enough that the tree is populated.
TREE_REUSE       = True

# 3. Resignation. When the MCTS root value stays below −threshold for several
#    consecutive moves, the current player concedes. Saves significant compute
#    on clearly-decided games. A fraction of games never resigns to keep the
#    value targets calibrated.
RESIGN_THRESHOLD     = 0.95   # |root_value| above this is "decisive"
RESIGN_PATIENCE      = 3      # consecutive moves below −threshold to resign
RESIGN_MIN_MOVE      = 6      # never resign in the opening
RESIGN_DISABLED_RATE = 0.10   # 10% of games disable resignation (calibration)

# 4. Mixed precision + torch.compile. Speeds up training & inference on CUDA
#    with no algorithmic change. Off on CPU (the bf16 path is sketchy outside
#    GPUs and torch.compile recompiles too often with MCTS's varying shapes).
USE_AMP          = True   # bf16 autocast in _train_step (CUDA only)
USE_TORCH_COMPILE = False # set True only if you've verified it speeds up your setup

# 5. AdamW + cosine LR schedule replaces Adam + manual L2 loss term.


# ── device ────────────────────────────────────────────────────────────────────
_device: torch.device | None = None

def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AlphaZero] device: {_device}")
    return _device


# ── network ───────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """One residual block: two 3×3 convolutions with BatchNorm and a skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class HexAlphaNet(nn.Module):
    """
    ResNet-style actor-critic network.

    Input:  (B, 2, size, size)
              ch 0 = current player's stones  (perspective frame)
              ch 1 = opponent's stones
    Output: logits (B, size²),  value (B,) ∈ [−1, +1]

    Architecture
    ─────────────
      stem  : 2 → channels via 3×3 conv + BN + ReLU
      trunk : blocks residual blocks
      heads : policy (channels → size²)  and  value (channels → 1 → tanh)

    BatchNorm requires net.eval() during inference (uses running statistics).
    The code calls net.eval() / net.train() at the appropriate points.
    """
    def __init__(self, size: int,
                 channels: int = NET_CHANNELS,
                 blocks: int   = NET_BLOCKS):
        super().__init__()
        self.size     = size
        self.channels = channels
        self.blocks   = blocks
        ch = channels

        self.stem = nn.Sequential(
            nn.Conv2d(2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[_ResBlock(ch) for _ in range(blocks)])

        # Policy head: conv → flatten → linear
        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * size * size, size * size),
        )
        # Value head: conv → flatten → FC → tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(size * size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        feat = self.trunk(self.stem(x))
        return self.policy_head(feat), self.value_head(feat).squeeze(-1)


# ── perspective & tensor helpers ──────────────────────────────────────────────

def _persp_board(game: engine.hexPosition):
    return game.board if game.player == RED else game.recode_blue_as_red()

def _persp_action(game: engine.hexPosition, action: tuple) -> tuple:
    return action if game.player == RED else game.recode_coordinates(action)

def _board_tensor(board, device: torch.device) -> torch.Tensor:
    t = torch.tensor(board, dtype=torch.float32, device=device)
    return torch.stack([(t == RED).float(), (t == BLUE).float()])

def _valid_mask(game: engine.hexPosition, device: torch.device) -> torch.Tensor:
    mask = torch.full((game.size * game.size,), float("-inf"), device=device)
    for a in game.get_action_space():
        mask[game.coordinate_to_scalar(_persp_action(game, a))] = 0.0
    return mask

def _masks_from_states(states: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Batch-reconstruct valid-action masks from stored perspective state tensors."""
    B, _, H, W = states.shape
    occupied = (states[:, 0] + states[:, 1]).view(B, -1) > 0
    masks = torch.zeros(B, H * W, device=device)
    masks[occupied] = float("-inf")
    return masks

def _copy_game(game: engine.hexPosition) -> engine.hexPosition:
    g        = engine.hexPosition(game.size)
    g.board  = [row[:] for row in game.board]
    g.player = game.player
    g.winner = game.winner
    return g


# ── MCTS ──────────────────────────────────────────────────────────────────────

class _Node:
    """MCTS node.  Each node owns a copy of the game state at that position."""
    __slots__ = ("game", "parent", "move", "children", "N", "W", "Q", "P", "expanded")

    def __init__(self, game: engine.hexPosition,
                 parent: "_Node | None" = None,
                 move: "tuple | None"   = None,
                 prior: float           = 0.0):
        self.game     = game
        self.parent   = parent
        self.move     = move      # real-coordinate move that created this node
        self.children: dict[tuple, "_Node"] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior
        self.expanded = False


def _puct_score(node: _Node, c: float) -> float:
    sqrt_n_parent = math.sqrt(node.parent.N) if node.parent else 1.0
    # node.Q is stored from the CHILD's player perspective (the opponent).
    # Negating gives the selector's perspective: prefer children where the
    # opponent wins LEAST.
    return -node.Q + c * node.P * sqrt_n_parent / (1 + node.N)


def _select_child(node: _Node, c: float) -> _Node:
    return max(node.children.values(), key=lambda ch: _puct_score(ch, c))


def _select_leaf(root: _Node) -> tuple["_Node", list["_Node"]]:
    """
    Walk root → leaf via PUCT, applying virtual loss at every visited node.

    Because PUCT uses −Q (child's perspective), the virtual-loss adjustment
    must be positive (+VL to W) so that Q increases, −Q decreases, and
    the PUCT score falls — correctly discouraging parallel selections from
    revisiting the same node.

    Returns (leaf, path) where path = [root, ..., leaf].
    """
    path: list[_Node] = []
    node = root
    while node.expanded and node.game.winner == EMPTY:
        node.W += VIRTUAL_LOSS   # raise Q → lower −Q → lower PUCT → repels next selection
        node.N += 1
        node.Q  = node.W / node.N
        path.append(node)
        node = _select_child(node, C_PUCT)
    # Apply virtual loss to the leaf itself
    node.W += VIRTUAL_LOSS
    node.N += 1
    node.Q  = node.W / node.N
    path.append(node)
    return node, path


def _backup(path: list["_Node"], v: float) -> None:
    """
    Undo virtual loss and apply actual negamax value along path.

    path[-1] is the leaf; v is the value from the leaf's current player's
    perspective.  The value is negated at each step toward the root.
    Net effect on W: += v  (the +VL from selection cancels the −VL here).
    """
    for node in reversed(path):
        node.W += v - VIRTUAL_LOSS   # undo +VL, add real value
        node.Q  = node.W / node.N
        v = -v


def _expand_nodes(leaves: list["_Node"], net: HexAlphaNet,
                  device: torch.device) -> dict[int, float]:
    """
    Batch-evaluate a list of unexpanded, non-terminal leaf nodes.

    One forward pass handles all leaves → fully utilises the GPU.
    Returns {id(leaf): value} for each expanded leaf.
    """
    # Deduplicate: multiple parallel selections can land on the same leaf
    seen: set[int] = set()
    unique: list[_Node] = []
    for l in leaves:
        if id(l) not in seen:
            seen.add(id(l))
            unique.append(l)

    states = torch.stack([
        _board_tensor(_persp_board(l.game), device) for l in unique
    ])
    masks = _masks_from_states(states, device)

    with torch.no_grad():
        logits_b, values_b = net(states)
    logits_masked = logits_b + masks

    value_map: dict[int, float] = {}
    for i, leaf in enumerate(unique):
        if leaf.expanded:
            continue   # a duplicate in a previous batch already expanded this node
        priors = F.softmax(logits_masked[i], dim=0)
        for a in leaf.game.get_action_space():
            persp_a = _persp_action(leaf.game, a)
            a_idx   = leaf.game.coordinate_to_scalar(persp_a)
            cg      = _copy_game(leaf.game)
            cg.move(a)
            leaf.children[a] = _Node(cg, parent=leaf, move=a,
                                     prior=priors[a_idx].item())
        leaf.expanded = True
        value_map[id(leaf)] = values_b[i].item()

    return value_map


def _run_mcts_py(game: engine.hexPosition, net: HexAlphaNet,
                 n_sims: int, device: torch.device,
                 dirichlet: bool = True,
                 parallel: int = PARALLEL_SIMS) -> dict[tuple, int]:
    """
    MCTS with batched parallel leaf evaluation.

    Each round:
      1. Run `parallel` PUCT selections simultaneously.
         Virtual loss forces each selection down a different branch.
      2. Batch-evaluate all collected leaves in ONE network forward pass.
         On GPU this gives a ~parallel× throughput improvement over
         sequential single-leaf evaluation.
      3. Backup all results, undoing the virtual losses.

    Correctness guarantee: virtual loss ensures different paths are explored;
    the actual Q values are restored correctly during backup.
    """
    net.eval()   # BatchNorm requires running statistics during inference

    root = _Node(_copy_game(game))

    # Expand root synchronously to initialise priors before any simulation
    _expand_nodes([root], net, device)

    if dirichlet and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([DIR_ALPHA] * len(moves))
        for mv, eta in zip(moves, noise):
            root.children[mv].P = (1 - DIR_EPS) * root.children[mv].P + DIR_EPS * eta

    done = 0
    while done < n_sims:
        batch_n = min(parallel, n_sims - done)

        # ── SELECTION ─────────────────────────────────────────────────────────
        leaves: list[_Node]      = []
        paths:  list[list[_Node]] = []
        for _ in range(batch_n):
            leaf, path = _select_leaf(root)
            leaves.append(leaf)
            paths.append(path)

        # ── EXPANSION (batched forward pass) ──────────────────────────────────
        to_expand = [l for l in leaves
                     if l.game.winner == EMPTY and not l.expanded]
        value_map = _expand_nodes(to_expand, net, device) if to_expand else {}

        # ── BACKUP ────────────────────────────────────────────────────────────
        for leaf, path in zip(leaves, paths):
            if leaf.game.winner != EMPTY:
                v = -1.0          # terminal: current player always lost
            else:
                v = value_map.get(id(leaf), 0.0)
                # 0.0 fallback: leaf was already expanded by another sim in this
                # batch; the value will be discovered on the next descent.
            _backup(path, v)

        done += batch_n

    return {mv: ch.N for mv, ch in root.children.items()}


def _run_mcts_cpp(game: "engine.hexPosition", net: HexAlphaNet,
                  n_sims: int, device: torch.device,
                  dirichlet: bool, parallel: int) -> dict:
    """C++ MCTS — drop-in replacement for _run_mcts using the hex_mcts extension."""
    board_flat = np.array(sum(game.board, []), dtype=np.int32)
    tree = _hex_mcts.MCTSTree(game.size, C_PUCT, VIRTUAL_LOSS)
    tree.set_root(board_flat, game.player)

    if tree.root_terminal():
        return {}

    # Expand root
    root_st = torch.from_numpy(tree.root_state()).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = net(root_st)
    mask = _masks_from_states(root_st, device)[0]
    priors = F.softmax(logits[0] + mask, dim=0).cpu().numpy().astype(np.float32)
    tree.expand_root(priors,
                     DIR_ALPHA if dirichlet else 0.0,
                     DIR_EPS   if dirichlet else 0.0)

    done = 0
    while done < n_sims:
        batch_n = min(parallel, n_sims - done)
        states_np, leaf_ids = tree.select_leaves(batch_n)
        if leaf_ids:
            states_t = torch.from_numpy(states_np).to(device)
            masks    = _masks_from_states(states_t, device)
            with torch.no_grad():
                logits_b, values_b = net(states_t)
            priors_b = F.softmax(logits_b + masks, dim=1).cpu().numpy().astype(np.float32)
            vals_b   = values_b.cpu().numpy().astype(np.float32)
            tree.expand_and_backup(leaf_ids, priors_b, vals_b)
        done += batch_n

    # Convert scalar keys → (row, col) tuples to match the Python _run_mcts API
    return {game.scalar_to_coordinates(s): n
            for s, n in tree.get_visit_counts().items()}


def _run_mcts(game: "engine.hexPosition", net: HexAlphaNet,
              n_sims: int, device: torch.device,
              dirichlet: bool = True,
              parallel: int = PARALLEL_SIMS) -> dict:
    """Dispatch to C++ MCTS if available, otherwise pure Python."""
    if _hex_mcts is not None:
        return _run_mcts_cpp(game, net, n_sims, device, dirichlet, parallel)
    return _run_mcts_py(game, net, n_sims, device, dirichlet, parallel)


# ── self-play data generation ─────────────────────────────────────────────────

def _self_play_game(net: HexAlphaNet, size: int, n_sims: int,
                    device: torch.device, parallel: int = PARALLEL_SIMS) -> list[tuple]:
    """
    Play one self-play game.
    Returns list of (state_tensor, policy_target, value_target).

    policy_target : (size²,) float tensor  — normalised MCTS visit counts
    value_target  : float  — +1 if the player who moved here eventually won, −1 otherwise
    """
    game    = engine.hexPosition(size)
    history = []  # (state_tensor, policy_tensor, player)
    move_n  = 0

    while game.winner == EMPTY:
        player = game.player
        state  = _board_tensor(_persp_board(game), device)

        visits = _run_mcts(game, net, n_sims, device, dirichlet=True, parallel=parallel)

        # Build policy target in perspective space
        policy = torch.zeros(size * size, device=device)
        for mv, n in visits.items():
            a_idx          = game.coordinate_to_scalar(_persp_action(game, mv))
            policy[a_idx]  = float(n)

        # Temperature: explore early, go greedy late
        if move_n < TEMP_CUTOFF:
            policy_sample = policy ** (1.0 / TEMP)
        else:
            policy_sample = (policy == policy.max()).float()
        policy_sample = policy_sample / policy_sample.sum()

        history.append((state, policy / (policy.sum() + 1e-8), player))

        # Sample action
        a_idx      = int(torch.multinomial(policy_sample, 1).item())
        persp_coord = game.scalar_to_coordinates(a_idx)
        real_coord  = _persp_action(game, persp_coord)   # self-inverse → real coords
        game.move(real_coord)
        move_n += 1

    winner = game.winner
    net.train()  # restore training mode for gradient updates
    return [(s, pi, 1.0 if pl == winner else -1.0) for s, pi, pl in history]


def _self_play_games_batched(net: HexAlphaNet, size: int, n_games: int,
                              n_sims: int, device: torch.device,
                              parallel: int = PARALLEL_SIMS) -> list[tuple]:
    """Dispatch to C++ or pure-Python batched self-play."""
    if _hex_mcts is not None:
        return _self_play_games_batched_cpp(net, size, n_games, n_sims, device, parallel)
    return _self_play_games_batched_py(net, size, n_games, n_sims, device, parallel)


def _self_play_games_batched_py(net: HexAlphaNet, size: int, n_games: int,
                                 n_sims: int, device: torch.device,
                                 parallel: int = PARALLEL_SIMS) -> list[tuple]:
    """
    Run n_games self-play games simultaneously, merging their leaf evaluations
    into a single GPU forward pass each round.

    Effective batch size ≈ n_games × parallel instead of just parallel.
    This is the main lever for GPU utilisation: the GPU stays busy processing
    large batches instead of idling between tiny single-game batches.
    """
    net.eval()

    class _GS:
        __slots__ = ("game", "history", "move_n", "root", "sims_done")
        def __init__(self):
            self.game      = engine.hexPosition(size)
            self.history   = []
            self.move_n    = 0
            self.root      = None
            self.sims_done = 0

    active: list[_GS] = [_GS() for _ in range(n_games)]
    all_samples: list[tuple] = []

    while active:
        # ── initialise roots for games that just started or just played a move ──
        new_roots_gs = [gs for gs in active if gs.root is None]
        if new_roots_gs:
            roots = [_Node(_copy_game(gs.game)) for gs in new_roots_gs]
            for gs, root in zip(new_roots_gs, roots):
                gs.root = root
                gs.sims_done = 0
            _expand_nodes(roots, net, device)
            for gs in new_roots_gs:
                root = gs.root
                if root.children:
                    moves = list(root.children.keys())
                    noise = np.random.dirichlet([DIR_ALPHA] * len(moves))
                    for mv, eta in zip(moves, noise):
                        root.children[mv].P = (
                            (1 - DIR_EPS) * root.children[mv].P + DIR_EPS * eta
                        )

        # ── selection: collect leaves from ALL games in one round ───────────────
        all_leaves: list[tuple] = []   # (leaf, path, gs)
        for gs in active:
            if gs.sims_done >= n_sims:
                continue
            batch_n = min(parallel, n_sims - gs.sims_done)
            for _ in range(batch_n):
                leaf, path = _select_leaf(gs.root)
                all_leaves.append((leaf, path, gs))

        # ── expansion: ONE forward pass for all games' leaves ───────────────────
        to_expand = [l for l, p, gs in all_leaves
                     if l.game.winner == EMPTY and not l.expanded]
        value_map = _expand_nodes(to_expand, net, device) if to_expand else {}

        # ── backup ──────────────────────────────────────────────────────────────
        sim_counts: dict[int, int] = {}
        for leaf, path, gs in all_leaves:
            v = -1.0 if leaf.game.winner != EMPTY else value_map.get(id(leaf), 0.0)
            _backup(path, v)
            sim_counts[id(gs)] = sim_counts.get(id(gs), 0) + 1
        gs_by_id = {id(gs): gs for gs in active}
        for gs_id, cnt in sim_counts.items():
            gs_by_id[gs_id].sims_done += cnt

        # ── advance games that finished their sims ───────────────────────────────
        finished: list[_GS] = []
        for gs in active:
            if gs.sims_done < n_sims:
                continue
            game   = gs.game
            root   = gs.root
            player = game.player

            policy = torch.zeros(size * size, device=device)
            for mv, ch in root.children.items():
                policy[game.coordinate_to_scalar(_persp_action(game, mv))] = float(ch.N)

            state = _board_tensor(_persp_board(game), device)
            if gs.move_n < TEMP_CUTOFF:
                policy_sample = policy ** (1.0 / TEMP)
            else:
                policy_sample = (policy == policy.max()).float()
            policy_sample = policy_sample / policy_sample.sum()

            gs.history.append((state, policy / (policy.sum() + 1e-8), player))

            a_idx       = int(torch.multinomial(policy_sample, 1).item())
            persp_coord = game.scalar_to_coordinates(a_idx)
            real_coord  = _persp_action(game, persp_coord)
            game.move(real_coord)
            gs.move_n += 1
            gs.root = None   # reset → new MCTS tree on next move

            if game.winner != EMPTY:
                winner = game.winner
                all_samples.extend(
                    (s, pi, 1.0 if pl == winner else -1.0)
                    for s, pi, pl in gs.history
                )
                finished.append(gs)

        for gs in finished:
            active.remove(gs)

    net.train()
    return all_samples


def _self_play_games_batched_cpp(net: HexAlphaNet, size: int, n_games: int,
                                  n_sims: int, device: torch.device,
                                  parallel: int = PARALLEL_SIMS) -> list[tuple]:
    """
    C++ variant of _self_play_games_batched_py.
    Uses hex_mcts.MCTSTree for tree traversal (10-50× faster than Python nodes)
    while still batching leaf evaluations across all n_games for GPU efficiency.
    """
    net.eval()

    class _GS:
        __slots__ = ("game", "history", "move_n", "tree", "sims_done",
                     "consec_resign", "no_resign")
        def __init__(self):
            self.game      = engine.hexPosition(size)
            self.history   = []
            self.move_n    = 0
            self.tree      = None   # C++ MCTSTree, reset each move (or reused)
            self.sims_done = 0
            self.consec_resign = 0
            self.no_resign     = random.random() < RESIGN_DISABLED_RATE

    active: list[_GS] = [_GS() for _ in range(n_games)]
    all_samples: list[tuple] = []

    while active:
        # ── init C++ trees for games that just started/played a move ──────────
        new_gs = [gs for gs in active if gs.tree is None]
        if new_gs:
            for gs in new_gs:
                gs.tree = _hex_mcts.MCTSTree(size, C_PUCT, VIRTUAL_LOSS)
                board_flat = np.array(sum(gs.game.board, []), dtype=np.int32)
                gs.tree.set_root(board_flat, gs.game.player)
                gs.sims_done = 0
            # Batch-expand all new roots in one forward pass
            root_states = torch.stack([
                torch.from_numpy(gs.tree.root_state()) for gs in new_gs
            ]).to(device)
            masks = _masks_from_states(root_states, device)
            with torch.no_grad():
                logits, _ = net(root_states)
            priors_batch = F.softmax(logits + masks, dim=1).cpu().numpy().astype(np.float32)
            for i, gs in enumerate(new_gs):
                gs.tree.expand_root(priors_batch[i], DIR_ALPHA, DIR_EPS)

        # ── select leaves across all games (C++ tree traversal) ──────────────
        all_pending: list[tuple] = []   # (gs, states_np, leaf_ids)
        for gs in active:
            if gs.sims_done >= n_sims:
                continue
            batch_n = min(parallel, n_sims - gs.sims_done)
            states_np, leaf_ids = gs.tree.select_leaves(batch_n)
            gs.sims_done += batch_n
            all_pending.append((gs, states_np, leaf_ids))

        # ── batch NN evaluation for all non-empty leaf sets ───────────────────
        non_empty = [(gs, s, l) for gs, s, l in all_pending if len(l) > 0]
        if non_empty:
            all_states = np.concatenate([s for _, s, _ in non_empty], axis=0)
            states_t   = torch.from_numpy(all_states).to(device)
            masks      = _masks_from_states(states_t, device)
            with torch.no_grad():
                logits_all, values_all = net(states_t)
            priors_all = F.softmax(logits_all + masks, dim=1).cpu().numpy().astype(np.float32)
            vals_all   = values_all.cpu().numpy().astype(np.float32)
            offset = 0
            for gs, _, leaf_ids in non_empty:
                B = len(leaf_ids)
                gs.tree.expand_and_backup(leaf_ids,
                                          priors_all[offset:offset + B],
                                          vals_all[offset:offset + B])
                offset += B

        # ── advance games that finished their sims ────────────────────────────
        finished: list[_GS] = []
        for gs in active:
            if gs.sims_done < n_sims:
                continue
            game   = gs.game
            player = game.player

            # ── resignation check ──────────────────────────────────────────
            # root_value() is Q from the player-to-move's POV; very negative
            # means "I am losing." Require RESIGN_PATIENCE consecutive moves
            # below −threshold so a single noisy estimate doesn't trigger.
            if (not gs.no_resign) and gs.move_n >= RESIGN_MIN_MOVE:
                if gs.tree.root_value() < -RESIGN_THRESHOLD:
                    gs.consec_resign += 1
                else:
                    gs.consec_resign = 0
                if gs.consec_resign >= RESIGN_PATIENCE:
                    # current player concedes; opponent is the winner
                    winner = -player
                    all_samples.extend(
                        (s, pi, 1.0 if pl == winner else -1.0)
                        for s, pi, pl in gs.history
                    )
                    finished.append(gs)
                    continue

            visits = gs.tree.get_visit_counts()  # {scalar_real: count}

            policy = torch.zeros(size * size, device=device)
            for scalar, n in visits.items():
                mv    = game.scalar_to_coordinates(scalar)
                a_idx = game.coordinate_to_scalar(_persp_action(game, mv))
                policy[a_idx] = float(n)

            state  = _board_tensor(_persp_board(game), device)
            if gs.move_n < TEMP_CUTOFF:
                policy_sample = policy ** (1.0 / TEMP)
            else:
                policy_sample = (policy == policy.max()).float()
            policy_sample = policy_sample / policy_sample.sum()

            gs.history.append((state, policy / (policy.sum() + 1e-8), player))

            a_idx       = int(torch.multinomial(policy_sample, 1).item())
            persp_coord = game.scalar_to_coordinates(a_idx)
            real_coord  = _persp_action(game, persp_coord)
            game.move(real_coord)
            gs.move_n += 1

            if game.winner != EMPTY:
                winner = game.winner
                all_samples.extend(
                    (s, pi, 1.0 if pl == winner else -1.0)
                    for s, pi, pl in gs.history
                )
                finished.append(gs)
                continue

            # ── tree reuse ────────────────────────────────────────────────
            # Try to keep the chosen child's subtree as the new root. Only
            # works if (a) advance_root finds a child for this move, and
            # (b) that child was expanded during the previous search. Both
            # are essentially always true for the most-visited move.
            real_scalar = game.coordinate_to_scalar(real_coord)
            if TREE_REUSE and gs.tree.advance_root(real_scalar) and gs.tree.root_expanded():
                gs.tree.add_root_dirichlet(DIR_ALPHA, DIR_EPS)
                gs.sims_done = 0
            else:
                gs.tree = None   # fall back to fresh tree on next iter

        for gs in finished:
            active.remove(gs)

    net.train()
    return all_samples


# ── symmetry augmentation ─────────────────────────────────────────────────────

def _augment_symmetries(samples: list) -> list:
    """
    Double every sample with its 180°-rotated equivalent.

    Under the perspective frame, every position is "RED to move, left↔right
    connector." Rotating the board 180° (flip both axes) maps RED's goal back
    to itself (col 0 ↔ col S−1, row 0 ↔ row S−1), so the policy/value targets
    transform deterministically:
      • state (2, S, S)  → spatial flip on both axes
      • π     (S²,)      → reverse the flat index (i ↔ S²−1−i)
      • z                → unchanged
    """
    out = list(samples)
    for state, pi, z in samples:
        state_rot = torch.flip(state, dims=(1, 2))     # 180° spatial rotation
        pi_rot    = torch.flip(pi,    dims=(0,))       # i → S²−1−i
        out.append((state_rot, pi_rot, z))
    return out


# ── replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, samples: list) -> None:
        self._buf.extend(samples)

    def sample(self, n: int) -> list:
        return random.sample(list(self._buf), min(n, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ── training step ─────────────────────────────────────────────────────────────

def _train_step(net: HexAlphaNet, opt: torch.optim.Optimizer,
                batch: list, device: torch.device) -> tuple[float, float]:
    """One gradient update on a mini-batch.  Returns (policy_loss, value_loss).

    Notes:
      • AdamW handles weight decay → no manual L2 term here.
      • bf16 autocast on CUDA: bf16 has the same exponent range as fp32, so no
        loss scaler is needed (unlike fp16). On CPU we fall back to fp32.
    """
    states = torch.stack([b[0] for b in batch]).to(device)
    pi_tgt = torch.stack([b[1] for b in batch]).to(device)
    z_tgt  = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)

    use_amp = USE_AMP and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        logits, values = net(states)
        masks    = _masks_from_states(states, device)
        log_probs = F.log_softmax(logits + masks, dim=1).clamp(min=-1e9)
        pi_loss = -(pi_tgt * log_probs).sum(dim=1).mean()
        vf_loss = F.mse_loss(values, z_tgt)
        loss    = pi_loss + vf_loss

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
    opt.step()

    return pi_loss.item(), vf_loss.item()


# ── evaluation ────────────────────────────────────────────────────────────────

def _eval_win_rate(net: HexAlphaNet, size: int,
                   n_games: int = 20, n_sims: int = 15,
                   parallel: int = PARALLEL_SIMS,
                   device: "torch.device | None" = None) -> float:
    """MCTS evaluation with a reduced sim budget (honest but fast enough for periodic checks)."""
    if device is None:
        device = _get_device()
    wins = 0
    for i in range(n_games):
        game    = engine.hexPosition(size)
        q_color = RED if i % 2 == 0 else BLUE
        while game.winner == EMPTY:
            if game.player == q_color:
                visits = _run_mcts(game, net, n_sims, device, dirichlet=False, parallel=parallel)
                move   = max(visits, key=visits.get)
                game.move(move)
            else:
                game.move(random.choice(game.get_action_space()))
        if game.winner == q_color:
            wins += 1
    return wins / n_games


# ── persistence ───────────────────────────────────────────────────────────────

def _model_path(size: int) -> str:
    return os.path.join(os.path.dirname(__file__), f"alphazero_{size}x{size}.pt")

def _checkpoint_path(size: int, iteration: int) -> str:
    return os.path.join(os.path.dirname(__file__), f"alphazero_{size}x{size}_ckpt{iteration:04d}.pt")

def save(net: HexAlphaNet, size: int) -> None:
    path = _model_path(size)
    torch.save({"size": size, "state_dict": net.state_dict()}, path)
    print(f"[AlphaZero] Saved → {path}")

def _save_checkpoint(net: HexAlphaNet, size: int, iteration: int) -> None:
    path = _checkpoint_path(size, iteration)
    torch.save({"size": size, "iteration": iteration, "state_dict": net.state_dict()}, path)
    print(f"[AlphaZero] Checkpoint → {path}")

def _torch_load(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load(size: int) -> bool:
    global _network, _trained_size
    path = _model_path(size)
    if not os.path.exists(path):
        return False
    device = _get_device()
    net = HexAlphaNet(size).to(device)
    net.load_state_dict(_torch_load(path, device)["state_dict"])
    net.eval()
    _network, _trained_size = net, size
    print(f"[AlphaZero] Loaded ← {path}")
    return True


def load_checkpoint(size: int, iteration: int) -> bool:
    """Load a checkpoint saved during training (e.g. alphazero_7x7_ckpt0060.pt).

    Returns True on success; False if the file does not exist.
    Use the returned state to resume training via train(..., resume_from=iteration).
    """
    global _network, _trained_size
    path = _checkpoint_path(size, iteration)
    if not os.path.exists(path):
        print(f"[AlphaZero] Checkpoint not found: {path}")
        return False
    device = _get_device()
    net = HexAlphaNet(size).to(device)
    net.load_state_dict(_torch_load(path, device)["state_dict"])
    net.eval()
    _network, _trained_size = net, size
    print(f"[AlphaZero] Loaded checkpoint ← {path}")
    return True


# ── multi-process self-play worker ───────────────────────────────────────────

def _worker_fn(rank: int,
               weight_q: "tmp.Queue",
               sample_q: "tmp.Queue",
               size: int, sims: int, parallel: int,
               n_games: int, channels: int, blocks: int) -> None:
    """
    Self-play worker process.

    Runs on CPU so it doesn't compete with the main process for the GPU.
    Lifecycle:
      1. Block on weight_q for the latest model weights.
      2. Run _self_play_games_batched on CPU.
      3. Push samples to sample_q.
      4. Repeat until weight_q yields None (shutdown signal).
    """
    torch.set_num_threads(1)          # prevent CPU over-subscription per worker
    device = torch.device("cpu")
    net    = HexAlphaNet(size, channels=channels, blocks=blocks).eval()

    while True:
        state_dict = weight_q.get()   # blocks until main sends weights or None
        if state_dict is None:
            return
        net.load_state_dict(state_dict)
        net.eval()
        samples = _self_play_games_batched(net, size, n_games, sims, device, parallel)
        sample_q.put(samples)


# ── training ──────────────────────────────────────────────────────────────────

_network:      HexAlphaNet | None = None
_trained_size: int                = 0


def train(size: int = 7, iterations: int = N_ITERATIONS,
          sims: int = N_SIMULATIONS, parallel: int = PARALLEL_SIMS,
          resume_from: int = 0,
          channels: int = NET_CHANNELS, blocks: int = NET_BLOCKS,
          n_workers: int = 1) -> dict:
    """
    Train via AlphaZero self-play.

    Each iteration:
      1. Play GAMES_PER_ITER self-play games using MCTS guided by current network.
         Each position is stored as (state, π_mcts, outcome).
      2. Run TRAIN_EPOCHS of mini-batch gradient updates from the replay buffer.

    resume_from : resume from this checkpoint iteration (0 = start fresh)
    channels    : residual feature channels (default 128; use 256 for boards >= 11×11)
    blocks      : number of residual blocks (default 5; use 8 for boards >= 11×11)
    n_workers   : CPU self-play workers (default 1 = single process).
                  Each worker runs GAMES_PER_ITER games in parallel on CPU,
                  while the main process does GPU training (pipelined).
                  Rule of thumb: n_workers = max(1, cpu_count - 1).

    Returns metrics dict with pi_losses, vf_losses, win_rates, size.
    """
    global _network, _trained_size

    device = _get_device()
    net    = HexAlphaNet(size, channels=channels, blocks=blocks).to(device)

    if resume_from > 0:
        ckpt_path = _checkpoint_path(size, resume_from)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        net.load_state_dict(_torch_load(ckpt_path, device)["state_dict"])
        print(f"[AlphaZero] Resuming from checkpoint {ckpt_path} (iter {resume_from})")

    if USE_TORCH_COMPILE:
        try:
            net = torch.compile(net, dynamic=True)
            print("[AlphaZero] torch.compile enabled (dynamic shapes)")
        except Exception as e:
            print(f"[AlphaZero] torch.compile unavailable ({e}); continuing without")

    # AdamW (decoupled weight decay) + cosine LR over the remaining iterations.
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=L2_REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, iterations - resume_from), eta_min=LR * 0.1
    )

    # Scale replay buffer and batch size with board complexity so the buffer
    # holds at least 70 iterations of self-play data regardless of board size.
    avg_moves     = size * size // 2            # rough moves/game estimate
    positions_per_iter = GAMES_PER_ITER * avg_moves
    replay_size   = max(REPLAY_SIZE, positions_per_iter * 70)
    batch_size    = min(512, max(BATCH_SIZE, positions_per_iter // 4))
    buf    = _ReplayBuffer(replay_size)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[AlphaZero] Network: {channels}ch × {blocks} blocks = {n_params/1e6:.2f}M params")
    print(f"[AlphaZero] Replay buffer: {replay_size:,}  batch: {batch_size}")

    pi_losses:   list[float] = []
    vf_losses:   list[float] = []
    win_history: list[tuple] = []

    # Guard: parallel >= sims means only 1 MCTS round per move — tree never
    # goes deeper than 2 levels, policy targets are nearly uniform, no learning.
    min_rounds = 7
    if parallel > sims // min_rounds:
        recommended = parallel * min_rounds
        print(f"[AlphaZero] WARNING: parallel={parallel} vs sims={sims} gives only "
              f"{max(1, sims // parallel)} MCTS round(s) per move — tree search too shallow.")
        print(f"[AlphaZero]   → Raise --sims to at least {recommended}  "
              f"(or lower --parallel to {sims // min_rounds}).")

    start_it    = resume_from
    remaining   = iterations - start_it
    total_games = remaining * GAMES_PER_ITER * max(1, n_workers)
    workers_str = f"  workers={n_workers}" if n_workers > 1 else ""
    print(f"[AlphaZero] Training iters {start_it+1}–{iterations} "
          f"({remaining:,} iters × {GAMES_PER_ITER * max(1,n_workers)} games = {total_games:,} games) "
          f"on {size}×{size}  (sims/move={sims}  parallel={parallel}"
          f"  rounds/move≈{sims//max(1,parallel)}{workers_str}) …")
    print(f"{'iter':>6}  {'buf':>6}  {'π loss':>8}  {'v loss':>8}  {'win% vs rand':>12}")
    print("-" * 52)

    def _do_train_step():
        ep_pi = ep_vf = 0.0
        n_up = 0
        if len(buf) >= batch_size:
            for _ in range(TRAIN_EPOCHS):
                pi_l, vf_l = _train_step(net, opt, buf.sample(batch_size), device)
                ep_pi += pi_l; ep_vf += vf_l; n_up += 1
        return (ep_pi / n_up, ep_vf / n_up) if n_up else (None, None)

    def _do_eval_and_log(it):
        win_rate = _eval_win_rate(net, size, n_games=20,
                                  n_sims=min(sims, 15), parallel=parallel, device=device)
        win_history.append((it + 1, win_rate))
        avg_pi = sum(pi_losses[-EVAL_EVERY:]) / max(1, len(pi_losses[-EVAL_EVERY:]))
        avg_vf = sum(vf_losses[-EVAL_EVERY:]) / max(1, len(vf_losses[-EVAL_EVERY:]))
        bar    = "█" * int(win_rate * 20) + "░" * (20 - int(win_rate * 20))
        pct    = (it + 1) / iterations * 100
        print(f"{it+1:>6,}  {len(buf):>6,}  {avg_pi:>8.4f}  {avg_vf:>8.4f}  "
              f"{win_rate*100:>5.1f}%  {bar}  ({pct:.0f}%)")
        _save_checkpoint(net, size, it + 1)

    def _maybe_augment(samples):
        return _augment_symmetries(samples) if SYMMETRY_AUG else samples

    if n_workers <= 1:
        # ── single-process training loop ──────────────────────────────────────
        for it in range(start_it, iterations):
            samples = _self_play_games_batched(
                net, size, GAMES_PER_ITER, sims, device, parallel=parallel)
            buf.push(_maybe_augment(samples))
            pi_l, vf_l = _do_train_step()
            if pi_l is not None:
                pi_losses.append(pi_l); vf_losses.append(vf_l)
            scheduler.step()
            if (it + 1) % EVAL_EVERY == 0:
                _do_eval_and_log(it)

    else:
        # ── multi-process pipelined training loop ─────────────────────────────
        # Workers run self-play on CPU (no GPU contention).
        # Main process does GPU training while workers generate the next batch.
        #
        # Timeline:
        #   iter N  workers: self-play[N]  |  main: train on buf[N-1]
        #   iter N+1 workers: self-play[N+1]|  main: collect[N], train on buf[N]

        # Sanity check: estimate per-iteration time on CPU. If it's hours, the
        # user almost certainly meant n_workers=1 — multi-worker only helps
        # when CPU inference is fast (small net), or with multiple GPUs.
        with torch.no_grad():
            torch.set_num_threads(1)
            probe_net = HexAlphaNet(size, channels=channels, blocks=blocks).eval()
            x = torch.randn(parallel, 2, size, size)
            for _ in range(2): probe_net(x)               # warmup
            import time as _time
            _t0 = _time.perf_counter()
            for _ in range(3): probe_net(x)
            t_fwd = (_time.perf_counter() - _t0) / 3
            del probe_net
        rounds      = max(1, sims // parallel)
        moves       = size * size // 2                    # rough
        sec_per_iter = t_fwd * rounds * moves * GAMES_PER_ITER
        if sec_per_iter > 600:                            # >10 min → warn
            print(f"[AlphaZero] ⚠ Estimated CPU self-play: ~{sec_per_iter/60:.0f} min/iter "
                  f"per worker — net is too big for CPU inference at this board+sims.")
            print(f"[AlphaZero]   → Recommended: drop --workers (use single GPU process), "
                  f"or shrink --channels/--blocks, or lower --sims.")
            if sec_per_iter > 1800:
                raise RuntimeError(
                    f"CPU workers would take ~{sec_per_iter/3600:.1f}h/iter. Aborting. "
                    f"Run with --workers 1 instead.")

        ctx         = tmp.get_context("spawn")
        weight_qs   = [ctx.Queue(maxsize=1) for _ in range(n_workers)]
        sample_q    = ctx.Queue()
        cpu_sd      = lambda: {k: v.cpu() for k, v in net.state_dict().items()}

        processes = [
            ctx.Process(
                target=_worker_fn, daemon=True,
                args=(rank, weight_qs[rank], sample_q,
                      size, sims, parallel, GAMES_PER_ITER, channels, blocks)
            )
            for rank in range(n_workers)
        ]
        for p in processes: p.start()
        print(f"[AlphaZero] Spawned {n_workers} self-play workers (CPU inference)")

        # Dispatch first batch of self-play to all workers
        sd = cpu_sd()
        for q in weight_qs: q.put(sd)

        for it in range(start_it, iterations):
            # Train on what's already in the buffer while workers generate data
            pi_l, vf_l = _do_train_step()
            if pi_l is not None:
                pi_losses.append(pi_l); vf_losses.append(vf_l)

            # Collect completed samples from all workers
            for _ in range(n_workers):
                buf.push(_maybe_augment(sample_q.get()))

            # Dispatch next batch of self-play (workers start immediately)
            if it + 1 < iterations:
                sd = cpu_sd()
                for q in weight_qs: q.put(sd)

            scheduler.step()
            if (it + 1) % EVAL_EVERY == 0:
                _do_eval_and_log(it)

        # Shutdown workers
        for q in weight_qs: q.put(None)
        for p in processes: p.join()

    net.eval()
    _network, _trained_size = net, size
    save(net, size)
    return {"pi_losses": pi_losses, "vf_losses": vf_losses,
            "win_rates": win_history, "size": size}


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_metrics(metrics: dict, smoothing: int = 10) -> None:
    """Plot training curves from the dict returned by train()."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[AlphaZero] pip install matplotlib numpy  to enable plotting")
        return

    pi_losses = metrics.get("pi_losses", [])
    vf_losses = metrics.get("vf_losses", [])
    win_rates = metrics.get("win_rates", [])
    sz        = metrics.get("size", "?")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    fig.suptitle(f"AlphaZero training — board {sz}×{sz}")

    def _smooth(xs, w):
        return np.convolve(xs, np.ones(w) / w, mode="valid")

    for ax, data, ylabel, title, color in [
        (axes[0], pi_losses, "Policy loss",  "Cross-entropy(p_net, π_mcts)  — lower = policy matches MCTS", "steelblue"),
        (axes[1], vf_losses, "Value MSE",    "MSE(v_net, z)  — lower = better win-probability estimates",    "tomato"),
    ]:
        if data:
            ax.plot(data, alpha=0.3, color=color, linewidth=0.8)
            if len(data) >= smoothing:
                ax.plot(range(smoothing - 1, len(data)), _smooth(data, smoothing),
                        color=color, linewidth=2, label=f"smoothed (w={smoothing})")
                ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax = axes[2]
    if win_rates:
        iters, wrs = zip(*win_rates)
        wrs_pct = [w * 100 for w in wrs]
        ax.plot(iters, wrs_pct, marker="o", color="darkorange", linewidth=2)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="random baseline")
        ax.fill_between(iters, 50, wrs_pct,
                        where=[w >= 50 for w in wrs_pct], alpha=0.15, color="green", label="above random")
        ax.fill_between(iters, 50, wrs_pct,
                        where=[w < 50  for w in wrs_pct], alpha=0.15, color="red",   label="below random")
        ax.set_ylim(0, 105)
        ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate vs random opponent (MCTS greedy, 20 games)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


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
    """
    AlphaZero agent.  Loads a saved model if available, trains from scratch otherwise.
    Uses INFERENCE_SIMS MCTS simulations per move (more than during training)
    so the agent thinks harder when playing against a human.
    """
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

    visits = _run_mcts(game, _network, INFERENCE_SIMS, device, dirichlet=False)
    best   = max(visits, key=visits.get)
    return best
