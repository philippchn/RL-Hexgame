# Hex Game — Reinforcement Learning Assignment (FHTW SS26)

A Hex board game engine and a collection of reference AI agents built with reinforcement learning. Students implement their own agent in `submission/facade.py` and test it against the provided engine.

---

## What is Hex?

Hex is a two-player connection game played on an N×N rhombus board of hexagonal cells.

- **RED** wins by connecting the **left edge to the right edge** (column 0 → column N−1).
- **BLUE** wins by connecting the **top edge to the bottom edge** (row 0 → row N−1).
- Players alternate turns; RED moves first.
- The game **cannot end in a draw** — exactly one player always wins.

The default board size is **7×7** (valid range: 2–26).

---

## Project structure

```
hex_engine.py          — game engine + Pygame GUI (hexPosition, HexPygameApp)
play_pygame.py         — interactive Pygame launcher
play_test_hex.py       — quick terminal test: your agent (Blue) vs random (Red)
train.py               — headless training script (all methods)
setup_mcts.py          — build script for the C++ MCTS extension
hex_mcts.cpp           — C++ pybind11 MCTS tree (optional, ~10-50× speedup for AlphaZero)

submission/
  facade.py            — YOUR AGENT (only file students modify)
  facade_random.py     — random baseline
  facade_mcts.py       — pure Monte Carlo Tree Search (UCT)
  facade_alphazero.py  — AlphaZero: ResNet + MCTS self-play
  facade_dqn.py        — Deep Q-Network (DQN)
  facade_dqn_minimax.py— DQN + Minimax search
  facade_ppo.py        — Proximal Policy Optimization (PPO)
  facade_reinforce.py  — REINFORCE (Monte Carlo policy gradient)
  facade_qlearning.py  — Tabular Q-learning

models/                — mirror of submission/ for side-by-side comparison
paper/                 — research paper source (LaTeX + PDF)
replays/               — saved game pickles
```

---

## Requirements

```bash
pip install pygame          # GUI only
pip install torch numpy     # neural network agents (AlphaZero, DQN, PPO, REINFORCE)
pip install pybind11        # optional C++ MCTS extension
```

No build step is required for the core engine or the pure-Python agents.

---

## Running the game

### Terminal test (no dependencies beyond the engine)

```bash
python play_test_hex.py
```

Runs one game: your agent (`submission/facade.py`) plays **Blue** against a **random Red** opponent and prints the result.

### Pygame GUI

```bash
python play_pygame.py
```

Prompts for board size, number of games, and play mode:

| Mode | Description |
|------|-------------|
| 1 | Human vs Human |
| 2 | Human (Red) vs Agent (Blue) |
| 3 | Agent (Red) vs Human (Blue) |
| 4 | Agent vs Agent |

**Keyboard shortcuts in the GUI:**

| Key | Action |
|-----|--------|
| `R` | Restart the current series |
| `N` | Skip to the next game |
| `Esc` | Quit |

---

## Student interface

The only file you modify is **`submission/facade.py`**. Your agent must be a callable with this exact signature:

```python
def agent(board, action_set) -> (row, col):
    ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `board` | `list[list[int]]` | Current board state. `EMPTY=0`, `RED=1`, `BLUE=-1` |
| `action_set` | `list[(row, col)]` | All legal moves (empty cells) |
| return value | `(row, col)` | One coordinate from `action_set` |

If your agent returns an invalid move, the engine silently falls back to a random move.

**Minimal working example:**

```python
from random import choice

def agent(board, action_set):
    return choice(action_set)
```

---

## Engine API (`hexPosition`)

```python
from hex_engine import hexPosition, EMPTY, RED, BLUE

game = hexPosition(size=7)   # create a fresh board
```

| Method / Attribute | Description |
|--------------------|-------------|
| `game.board` | 2D list; values are `EMPTY=0`, `RED=1`, `BLUE=-1` |
| `game.player` | Whose turn it is (`RED` or `BLUE`) |
| `game.winner` | `EMPTY` until the game ends, then `RED` or `BLUE` |
| `game.size` | Board dimension |
| `game.move((row, col))` | Place a stone and advance the game state |
| `game.get_action_space()` | Return list of empty cells `[(row, col), ...]` |
| `game.reset()` | Clear the board and restart |
| `game.machine_vs_machine(m1, m2)` | Run a full game; `m1` plays Red, `m2` plays Blue. Returns winner. |
| `game.recode_blue_as_red()` | Flip board so Blue's perspective looks like Red's (diagonal transpose + color swap) |
| `game.recode_coordinates(coord)` | Transform a coordinate under the same flip |
| `game.coordinate_to_scalar((row, col))` | Convert board coordinate → flat integer index |
| `game.scalar_to_coordinates(n)` | Reverse of above |
| `game.print()` | Pretty-print the board in the terminal |
| `game.save(path)` | Pickle the game object to a file |
| `game.replay_history()` | Step through move history in the terminal |

### Board geometry

```
Columns →  A   B   C   D   E   F   G
          _   _   _   _   _   _   _
         / \_/ \_/ \_/ \_/ \_/ \_/ \
    1   |   |   |   |   |   |   |   | 1
         \_/ \_/ \_/ \_/ \_/ \_/ \_/ \
    2   |   |   |   |   |   |   |   | 2
         ...
```

Each cell has **6 neighbours**: up, down, left, right, up-right, down-left.

### Running a match programmatically

```python
from hex_engine import hexPosition
from submission.facade import agent as my_agent

game = hexPosition(size=7)
winner = game.machine_vs_machine(machine1=None, machine2=my_agent)
# machine1=None → random Red opponent
print("Winner:", winner)  # 1=RED, -1=BLUE
```

---

## Symmetric agent view (`recode_blue_as_red`)

Because RED connects left↔right and BLUE connects top↔bottom, the board geometry is asymmetric. `recode_blue_as_red()` resolves this by transposing the board along the SW–NE diagonal and swapping colors, so **Blue's problem looks identical to Red's problem**. An agent trained to play Red can also play Blue using this transformation:

```python
def agent(board, action_set):
    game = build_game(board)

    if game.player == BLUE:
        flipped_board = game.recode_blue_as_red()
        chosen_in_perspective = my_red_policy(flipped_board)
        return game.recode_coordinates(chosen_in_perspective)
    else:
        return my_red_policy(board)
```

All neural network agents in this repo use this trick internally so a single network plays both colors.

---

## Reference agents

| File | Algorithm | Key idea |
|------|-----------|----------|
| `facade_random.py` | Random | Picks a uniformly random legal move |
| `facade_mcts.py` | MCTS (UCT) | 800 random rollouts per move; fast BFS win check |
| `facade_alphazero.py` | AlphaZero | ResNet actor-critic + PUCT MCTS, self-play training |
| `facade_dqn.py` | DQN | CNN Q-network, experience replay, target network |
| `facade_dqn_minimax.py` | DQN + Minimax | DQN Q-values used inside a depth-limited minimax |
| `facade_ppo.py` | PPO | Shared actor-critic, clipped surrogate, entropy annealing |
| `facade_reinforce.py` | REINFORCE | Monte Carlo policy gradient, no value baseline |
| `facade_qlearning.py` | Q-learning | Tabular, perspective-invariant state encoding |

### AlphaZero details

The AlphaZero agent (`facade_alphazero.py`) is the strongest reference implementation. Key components:

- **Network**: ResNet with 5 residual blocks × 128 channels (7×7); 8 blocks × 256 channels (≥11×11).  
  Input: `(B, 2, size, size)` — one channel per player in the perspective frame.  
  Output: policy logits `(B, size²)` + scalar value `(B,)`.
- **MCTS**: PUCT selection, batched parallel leaf expansion, virtual loss, Dirichlet root noise.
- **Self-play loop**: generate games → fill replay buffer → train on mini-batches (AdamW + cosine LR).
- **Enhancements**: tree reuse, 180° symmetry augmentation, resignation detection, optional C++ tree extension.
- **Scaling**: simulation budget, network size, and replay buffer auto-scale with board size.

---

## Training

Use `train.py` to train any of the reference agents headlessly:

```bash
# AlphaZero on 7×7 (default)
python train.py

# AlphaZero with explicit settings
python train.py --method alphazero --size 7 --iterations 500 --sims 100 --parallel 16

# Resume from checkpoint
python train.py --method alphazero --size 7 --resume-from 60

# Multi-worker self-play (pipeline CPU workers with GPU training)
python train.py --method alphazero --size 7 --workers 4

# Other methods
python train.py --method ppo       --size 7 --iterations 2000
python train.py --method reinforce --size 5 --episodes 15000
python train.py --method dqn       --size 5 --episodes 10000
python train.py --method dqn_minimax --size 5 --episodes 10000
python train.py --method qlearning --size 5 --episodes 100000

# Plot training curves after (or instead of) training
python train.py --plot
python train.py --save-metrics metrics.json
```

### AlphaZero training options

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | 7 | Board size |
| `--iterations` | 200 | Training iterations |
| `--sims` | auto | MCTS simulations per move (auto-scales with board size: 7→150, 9→300, …) |
| `--parallel` | 8 | Parallel MCTS leaves per batch (must be ≤ sims/7) |
| `--resume-from` | 0 | Resume from checkpoint iteration N |
| `--channels` | auto | Residual feature channels (128 for ≤9×9, 256 for ≥11×11) |
| `--blocks` | auto | Residual blocks (5 for ≤9×9, 8 for ≥11×11) |
| `--workers` | 1 | CPU self-play workers (set to `cpu_count - 1` for pipelining) |

Saved models: `submission/alphazero_{size}x{size}.pt`  
Checkpoints: `submission/alphazero_{size}x{size}_ckpt{iter:04d}.pt`

---

## C++ MCTS extension (optional, AlphaZero only)

A pybind11 C++ extension (`hex_mcts.cpp`) provides 10–50× faster tree traversal for AlphaZero:

```bash
pip install pybind11
python setup_mcts.py build_ext --inplace
```

This produces `hex_mcts.cpython-*.so` in the project root. `facade_alphazero.py` auto-detects it; if absent it falls back to pure Python transparently.

---

## Board conventions summary

| Constant | Value | Meaning |
|----------|-------|---------|
| `EMPTY` | 0 | Empty cell |
| `RED` | 1 | Red stone — connects **left ↔ right** |
| `BLUE` | -1 | Blue stone — connects **top ↔ bottom** |

- `machine_vs_machine(m1, m2)`: `m1` plays Red, `m2` plays Blue.
- An invalid move returned by an agent is silently replaced with a random legal move.
- Board size is clamped to [2, 26].
