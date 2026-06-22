/**
 * hex_mcts.cpp — fast MCTS tree for Hex, exposed to Python via pybind11.
 *
 * Build:
 *   pip install pybind11
 *   python setup_mcts.py build_ext --inplace
 *
 * Replaces the three Python bottlenecks in facade_alphazero.py:
 *   1. _copy_game    → C++ vector copy (10-50× faster)
 *   2. hexPosition.move + BFS winner check  → tight C++ BFS
 *   3. _select_leaf PUCT loop → C++ while loop, no Python object overhead
 *
 * Interface (see MCTSTree class below):
 *   tree = hex_mcts.MCTSTree(size, c_puct, virtual_loss)
 *   tree.set_root(board_flat_int32, player)
 *   tree.expand_root(priors_float32, dir_alpha, dir_eps)
 *   states, leaf_ids = tree.select_leaves(parallel)
 *   tree.expand_and_backup(leaf_ids, priors_2d, values_1d)
 *   visits = tree.get_visit_counts()   # {scalar_real_coord: visit_count}
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ── hex adjacency (6 neighbours) ──────────────────────────────────────────────
static const int DR[6] = {-1,  1,  0,  0, -1,  1};
static const int DC[6] = { 0,  0, -1,  1,  1, -1};

// ── win detection ─────────────────────────────────────────────────────────────
// Only checks the player who just moved (the other cannot have won this turn).

static bool red_wins(const std::vector<int8_t>& b, int sz) {
    // RED(1): connect col 0 → col sz-1
    std::vector<bool> vis(sz * sz, false);
    std::queue<int> q;
    for (int r = 0; r < sz; ++r) {
        int p = r * sz;
        if (b[p] == 1) { vis[p] = true; q.push(p); }
    }
    while (!q.empty()) {
        int p = q.front(); q.pop();
        if (p % sz == sz - 1) return true;
        int r = p / sz, c = p % sz;
        for (int i = 0; i < 6; ++i) {
            int nr = r + DR[i], nc = c + DC[i];
            if (nr < 0 || nr >= sz || nc < 0 || nc >= sz) continue;
            int np = nr * sz + nc;
            if (!vis[np] && b[np] == 1) { vis[np] = true; q.push(np); }
        }
    }
    return false;
}

static bool blue_wins(const std::vector<int8_t>& b, int sz) {
    // BLUE(-1): connect row 0 → row sz-1
    std::vector<bool> vis(sz * sz, false);
    std::queue<int> q;
    for (int c = 0; c < sz; ++c) {
        int p = c;
        if (b[p] == -1) { vis[p] = true; q.push(p); }
    }
    while (!q.empty()) {
        int p = q.front(); q.pop();
        if (p / sz == sz - 1) return true;
        int r = p / sz, c = p % sz;
        for (int i = 0; i < 6; ++i) {
            int nr = r + DR[i], nc = c + DC[i];
            if (nr < 0 || nr >= sz || nc < 0 || nc >= sz) continue;
            int np = nr * sz + nc;
            if (!vis[np] && b[np] == -1) { vis[np] = true; q.push(np); }
        }
    }
    return false;
}

// ── MCTS node ─────────────────────────────────────────────────────────────────

struct Node {
    std::vector<int8_t> board;   // flattened sz×sz board
    int8_t  player;              // whose turn it is at this node (1=RED, -1=BLUE)
    int8_t  winner;              // 0=ongoing, 1=RED won, -1=BLUE won
    int     parent;              // index in pool (-1 for root)
    int     move;                // real-coordinate flat index that created this node
    std::unordered_map<int,int> children;  // move_scalar → pool index
    int     N;
    float   W, Q, P;
    bool    expanded;

    Node(std::vector<int8_t> b, int8_t pl, int8_t win,
         int par, int mv, float prior)
        : board(std::move(b)), player(pl), winner(win),
          parent(par), move(mv),
          N(0), W(0.f), Q(0.f), P(prior), expanded(false) {}
};

// ── MCTS tree ─────────────────────────────────────────────────────────────────

class MCTSTree {
    int   sz;
    float cpuct, vl;

    std::vector<Node> pool;
    int   root_idx = 0;          // pool index of the current root; moves on advance_root()

    // Between select_leaves() and expand_and_backup():
    //   pending holds all (leaf_index, path) pairs that need backup.
    struct Pending { int leaf; std::vector<int> path; };
    std::vector<Pending> pending;

    // ── perspective state tensor ───────────────────────────────────────────────
    // Layout: [ch0: sz×sz current-player stones][ch1: sz×sz opponent stones]
    // For BLUE's turn we apply recode_blue_as_red (SW-NE diagonal flip + color swap).
    void fill_state(int idx, float* buf) const {
        const Node& n = pool[idx];
        float* ch0 = buf;
        float* ch1 = buf + sz * sz;
        if (n.player == 1) {                    // RED: identity
            for (int i = 0; i < sz * sz; ++i) {
                ch0[i] = (n.board[i] ==  1) ? 1.f : 0.f;
                ch1[i] = (n.board[i] == -1) ? 1.f : 0.f;
            }
        } else {                                // BLUE: recode_blue_as_red
            for (int r = 0; r < sz; ++r)
                for (int c = 0; c < sz; ++c) {
                    int src = (sz-1-c) * sz + (sz-1-r);
                    int8_t v = -n.board[src];   // swap RED↔BLUE
                    ch0[r*sz+c] = (v ==  1) ? 1.f : 0.f;
                    ch1[r*sz+c] = (v == -1) ? 1.f : 0.f;
                }
        }
    }

    // ── PUCT score ────────────────────────────────────────────────────────────
    float puct(int idx) const {
        const Node& n   = pool[idx];
        const Node& par = pool[n.parent];
        return -n.Q + cpuct * n.P * std::sqrt((float)par.N) / (1.f + n.N);
    }

    int best_child(int parent_idx) const {
        float best_s = -1e9f;
        int   best_i = -1;
        for (auto& [mv, ci] : pool[parent_idx].children) {
            float s = puct(ci);
            if (s > best_s) { best_s = s; best_i = ci; }
        }
        return best_i;
    }

    // ── PUCT descent with virtual loss ────────────────────────────────────────
    int walk_to_leaf(std::vector<int>& path) {
        int idx = root_idx;
        while (pool[idx].expanded && pool[idx].winner == 0) {
            pool[idx].W += vl;
            ++pool[idx].N;
            pool[idx].Q = pool[idx].W / pool[idx].N;
            path.push_back(idx);
            idx = best_child(idx);
        }
        pool[idx].W += vl;
        ++pool[idx].N;
        pool[idx].Q = pool[idx].W / pool[idx].N;
        path.push_back(idx);
        return idx;
    }

    // ── negamax backup (undoes virtual loss) ──────────────────────────────────
    void do_backup(const std::vector<int>& path, float v) {
        for (int i = (int)path.size() - 1; i >= 0; --i) {
            int idx = path[i];
            pool[idx].W += v - vl;
            pool[idx].Q  = pool[idx].W / pool[idx].N;
            v = -v;
        }
    }

    // ── expand a node given a priors array (perspective-space, sz*sz) ─────────
    // NOTE: accesses pool[idx] by index only — safe across pool reallocation.
    void expand_node(int idx, const float* priors) {
        if (pool[idx].expanded) return;
        int8_t pl = pool[idx].player;

        for (int pos = 0; pos < sz * sz; ++pos) {
            if (pool[idx].board[pos] != 0) continue;
            int r = pos / sz, c = pos % sz;

            // Index into priors array (perspective space)
            int pscalar = (pl == 1) ? pos : (sz-1-c) * sz + (sz-1-r);

            // Build child board
            std::vector<int8_t> cb = pool[idx].board;  // copy before emplace_back
            cb[pos] = pl;
            int8_t cwin = (pl == 1) ? (red_wins(cb, sz)  ? (int8_t) 1 : (int8_t)0)
                                    : (blue_wins(cb, sz) ? (int8_t)-1 : (int8_t)0);

            int ci = (int)pool.size();
            pool[idx].children[pos] = ci;               // modify before emplace_back
            pool.emplace_back(std::move(cb), (int8_t)-pl, cwin, idx, pos, priors[pscalar]);
            // pool[idx] still valid by index after potential reallocation
        }
        pool[idx].expanded = true;
    }

public:
    MCTSTree(int size, float c_puct, float virtual_loss)
        : sz(size), cpuct(c_puct), vl(virtual_loss) {
        pool.reserve(1 << 18);   // 256 k nodes — avoids reallocation during training
    }

    // ── set_root ──────────────────────────────────────────────────────────────
    // board_arr: 1-D int32 numpy array of length sz*sz (row-major)
    void set_root(py::array_t<int32_t> board_arr, int player) {
        pool.clear();
        pending.clear();
        root_idx = 0;
        auto b = board_arr.unchecked<1>();
        std::vector<int8_t> board(sz * sz);
        for (int i = 0; i < sz * sz; ++i) board[i] = (int8_t)b(i);
        pool.emplace_back(std::move(board), (int8_t)player, (int8_t)0, -1, -1, 0.f);
    }

    bool root_terminal() const { return pool[root_idx].winner != 0; }
    bool root_expanded() const { return pool[root_idx].expanded; }

    // Returns (2, sz, sz) float32 numpy array — root's perspective state.
    py::array_t<float> root_state() const {
        py::array_t<float> arr({2, sz, sz});
        fill_state(root_idx, arr.mutable_data());
        return arr;
    }

    // Root's Q value from the perspective of the player to move at root.
    // Useful for resignation: Q < −threshold ⇒ current player is losing.
    float root_value() const {
        return pool[root_idx].N == 0 ? 0.f : pool[root_idx].Q;
    }

    // Adds Dirichlet(α) noise to root's children's priors (used after advance_root).
    void add_root_dirichlet(float dir_alpha, float dir_eps) {
        if (dir_alpha <= 0.f || dir_eps <= 0.f) return;
        if (pool[root_idx].children.empty()) return;
        int n = (int)pool[root_idx].children.size();
        static std::mt19937 rng{std::random_device{}()};
        std::gamma_distribution<float> gamma(dir_alpha, 1.f);
        std::vector<float> noise(n);
        float sum = 0.f;
        for (auto& x : noise) { x = gamma(rng); sum += x; }
        for (auto& x : noise) x /= sum;
        int i = 0;
        for (auto& [mv, ci] : pool[root_idx].children)
            pool[ci].P = (1.f - dir_eps) * pool[ci].P + dir_eps * noise[i++];
    }

    // Tree reuse: make the child reached by `move_scalar` (real-coordinate flat
    // index) the new root. Returns false if no such child exists (caller should
    // fall back to a fresh set_root + expand_root). Old root and its other
    // subtrees stay in the pool as garbage — they're never visited again.
    bool advance_root(int move_scalar) {
        auto it = pool[root_idx].children.find(move_scalar);
        if (it == pool[root_idx].children.end()) return false;
        root_idx = it->second;
        pool[root_idx].parent = -1;
        return true;
    }

    // Expand root with given priors; optionally add Dirichlet noise.
    // priors: 1-D float32 numpy array of length sz*sz (perspective-space softmax output)
    // dir_alpha=0 or dir_eps=0 → no noise
    void expand_root(py::array_t<float> priors_arr, float dir_alpha, float dir_eps) {
        expand_node(root_idx, priors_arr.data());

        if (dir_alpha > 0.f && dir_eps > 0.f && !pool[root_idx].children.empty()) {
            int n = (int)pool[root_idx].children.size();
            static std::mt19937 rng{std::random_device{}()};
            std::gamma_distribution<float> gamma(dir_alpha, 1.f);
            std::vector<float> noise(n);
            float sum = 0.f;
            for (auto& x : noise) { x = gamma(rng); sum += x; }
            for (auto& x : noise) x /= sum;
            int i = 0;
            for (auto& [mv, ci] : pool[root_idx].children)
                pool[ci].P = (1.f - dir_eps) * pool[ci].P + dir_eps * noise[i++];
        }
    }

    // ── select_leaves ─────────────────────────────────────────────────────────
    // Run `parallel` PUCT selections with virtual loss.
    // Terminal leaves → backed up immediately with v = -1.
    // Already-expanded leaves → backed up with v = 0.
    // Unexpanded, non-terminal leaves → collected for NN evaluation.
    //
    // Returns:
    //   states   : (B, 2, sz, sz) float32 — perspective states of unique leaves
    //   leaf_ids : list[int] of length B — pool indices
    //
    // Stores all pending paths internally; call expand_and_backup() next.
    py::tuple select_leaves(int parallel) {
        pending.clear();
        std::unordered_map<int,int> eval_map;   // leaf_idx → output slot
        std::vector<int> eval_ids;

        for (int i = 0; i < parallel; ++i) {
            std::vector<int> path;
            int leaf = walk_to_leaf(path);

            if (pool[leaf].winner != 0) {
                do_backup(path, -1.f);
            } else if (pool[leaf].expanded) {
                do_backup(path, 0.f);
            } else {
                if (eval_map.find(leaf) == eval_map.end()) {
                    eval_map[leaf] = (int)eval_ids.size();
                    eval_ids.push_back(leaf);
                }
                pending.push_back({leaf, std::move(path)});
            }
        }

        int B = (int)eval_ids.size();
        py::array_t<float> states({B, 2, sz, sz});
        float* ptr = states.mutable_data();
        for (int b = 0; b < B; ++b)
            fill_state(eval_ids[b], ptr + b * 2 * sz * sz);

        return py::make_tuple(states, eval_ids);
    }

    // ── expand_and_backup ─────────────────────────────────────────────────────
    // leaf_ids : list returned by select_leaves
    // priors   : (B, sz*sz) float32 — perspective-space masked softmax output
    // values   : (B,) float32 — NN value estimates
    void expand_and_backup(const std::vector<int>& leaf_ids,
                           py::array_t<float> priors_arr,
                           py::array_t<float> values_arr) {
        const float* pr_ptr = priors_arr.data();
        const float* vl_ptr = values_arr.data();
        int B = (int)leaf_ids.size();

        std::unordered_map<int,float> vmap;
        for (int b = 0; b < B; ++b) {
            int lidx = leaf_ids[b];
            if (!pool[lidx].expanded)
                expand_node(lidx, pr_ptr + (ptrdiff_t)b * sz * sz);
            vmap[lidx] = vl_ptr[b];
        }
        for (auto& pend : pending) {
            float v = vmap.count(pend.leaf) ? vmap[pend.leaf] : 0.f;
            do_backup(pend.path, v);
        }
        pending.clear();
    }

    // Returns {real_coord_scalar: visit_count} for root's children.
    std::unordered_map<int,int> get_visit_counts() const {
        std::unordered_map<int,int> out;
        for (auto& [mv, ci] : pool[root_idx].children)
            out[mv] = pool[ci].N;
        return out;
    }

    int pool_size() const { return (int)pool.size(); }
};

// ── pybind11 binding ──────────────────────────────────────────────────────────

PYBIND11_MODULE(hex_mcts, m) {
    m.doc() = "Fast C++ MCTS tree for the Hex game";
    py::class_<MCTSTree>(m, "MCTSTree")
        .def(py::init<int, float, float>(),
             py::arg("size"), py::arg("c_puct"), py::arg("virtual_loss"))
        .def("set_root",            &MCTSTree::set_root)
        .def("root_terminal",       &MCTSTree::root_terminal)
        .def("root_expanded",       &MCTSTree::root_expanded)
        .def("root_state",          &MCTSTree::root_state)
        .def("root_value",          &MCTSTree::root_value)
        .def("expand_root",         &MCTSTree::expand_root,
             py::arg("priors"), py::arg("dir_alpha"), py::arg("dir_eps"))
        .def("add_root_dirichlet",  &MCTSTree::add_root_dirichlet,
             py::arg("dir_alpha"), py::arg("dir_eps"))
        .def("advance_root",        &MCTSTree::advance_root,
             py::arg("move_scalar"))
        .def("select_leaves",       &MCTSTree::select_leaves)
        .def("expand_and_backup",   &MCTSTree::expand_and_backup)
        .def("get_visit_counts",    &MCTSTree::get_visit_counts)
        .def("pool_size",           &MCTSTree::pool_size);
}
