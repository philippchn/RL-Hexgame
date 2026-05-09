# submission/greedy_agent.py

from collections import deque

EMPTY =  0
RED   =  1
BLUE  = -1


def _bfs_shortest_path(board, size, player):
    """
    Returns the minimum number of EMPTY cells still needed
    to complete a winning path for `player`.

    RED  must connect left (col 0) → right (col size-1).
    BLUE must connect top  (row 0) → bottom (row size-1).

    Traversal costs:
        own stone  → 0  (already placed, free)
        empty cell → 1  (would need to place here)
        opponent   → blocked (never cross)

    Uses 0-1 BFS (deque): cost-0 edges go to the front,
    cost-1 edges go to the back. Gives exact shortest path.
    """
    INF = size * size + 1
    dist = [[INF] * size for _ in range(size)]
    dq = deque()

    # Seed the BFS from the player's starting edge
    if player == RED:
        start_cells = [(r, 0) for r in range(size) if board[r][0] != BLUE]
    else:
        start_cells = [(0, c) for c in range(size) if board[0][c] != RED]

    for (r, c) in start_cells:
        cost = 0 if board[r][c] == player else 1
        dist[r][c] = cost
        if cost == 0:
            dq.appendleft((r, c))
        else:
            dq.append((r, c))

    # Hex neighbours (6 directions)
    def neighbours(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc

    while dq:
        r, c = dq.popleft()
        d = dist[r][c]

        # Check if we reached the goal edge
        if player == RED and c == size - 1:
            return d
        if player == BLUE and r == size - 1:
            return d

        for nr, nc in neighbours(r, c):
            if board[nr][nc] == -player:   # blocked
                continue
            cost = 0 if board[nr][nc] == player else 1
            nd = d + cost
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                if cost == 0:
                    dq.appendleft((nr, nc))
                else:
                    dq.append((nr, nc))

    return INF   # no path exists (shouldn't happen in Hex)


def greedy_agent(board, action_set):
    """
    Greedy Hex agent. Matches the exact signature your engine expects:
        agent(board, action_set) -> (row, col)

    For each legal move it simulates placing the stone, measures how
    much shorter its own path got, and picks the best one. A small
    blocking bonus rewards moves on the opponent's critical path.
    """
    size   = len(board)
    red_n  = sum(c == RED  for row in board for c in row)
    blue_n = sum(c == BLUE for row in board for c in row)
    player = RED if red_n == blue_n else BLUE

    alpha = 0.5   # blocking weight — tune between 0 (pure attack) and 1 (pure block)

    opp_path_now = _bfs_shortest_path(board, size, -player)

    best_score  = float("-inf")
    best_action = action_set[0]

    for (r, c) in action_set:
        board[r][c] = player                              # simulate
        my_path = _bfs_shortest_path(board, size, player)
        board[r][c] = EMPTY                               # undo

        score = -my_path + alpha * opp_path_now
        if score > best_score:
            best_score  = score
            best_action = (r, c)

    return best_action