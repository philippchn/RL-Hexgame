# hex_engine.py

from copy import deepcopy
from random import choice
import math


EMPTY = 0
RED = 1      # Red connects left to right
BLUE = -1    # Blue connects top to bottom


class hexPosition:
    """
    Hex game engine.

    Board encoding:
        0  = empty
        1  = red
        -1 = blue

    Red connects left to right.
    Blue connects top to bottom.

    This class keeps the original public API:
        game = hexPosition()
        game.move((row, col))
        game.machine_vs_machine(machine1, machine2)

    Agents must have the signature:
        agent(board, action_set) -> move
    """

    def __init__(self, size=7):
        # Keep original size bounds.
        size = max(2, min(size, 26))

        self.size = size
        self.board = [[EMPTY for _ in range(size)] for _ in range(size)]

        self.player = RED
        self.winner = EMPTY

        self.history = [self.board]

    def reset(self):
        """
        Reset the board.

        """
        self.board = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.player = RED
        self.winner = EMPTY
        self.history = []

    def move(self, coordinates):
        """
        Enact one move.

        This keeps the following move order:
            1. place current player's stone
            2. switch active player
            3. evaluate board
            4. append board to history
        """
        row, col = coordinates

        assert self.winner == EMPTY, "The game is already won."
        assert self.board[row][col] == EMPTY, "These coordinates already contain a stone."

        self.board[row][col] = self.player

        # Original behavior: switch player before evaluation.
        self.player *= -1

        self.evaluate()

        self.history.append(deepcopy(self.board))

    def print(self, invert_colors=True):
        """
        Print the board in the terminal.

        Red is represented by ●.
        Blue is represented by ○.
        """
        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0

        headings = " " * 5 + (" " * 3).join(names[:self.size])
        print(headings)

        tops = " " * 5 + (" " * 3).join("_" * self.size)
        print(tops)

        roof = " " * 4 + "/ \\" + "_/ \\" * (self.size - 1)
        print(roof)

        if invert_colors:
            color_mapping = lambda value: (
                " " if value == EMPTY else
                "\u25CB" if value == BLUE else
                "\u25CF"
            )
        else:
            color_mapping = lambda value: (
                " " if value == EMPTY else
                "\u25CF" if value == BLUE else
                "\u25CB"
            )

        for row in range(self.size):
            row_mid = " " * indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping, self.board[row]))
            row_mid += f" | {row + 1} "
            print(row_mid)

            row_bottom = " " * indent
            row_bottom += " " * 3 + " \\_/" * self.size

            if row < self.size - 1:
                row_bottom += " \\"

            print(row_bottom)
            indent += 2

        headings = " " * (indent - 2) + headings
        print(headings)

    def _get_adjacent(self, coordinates):
        """
        Return adjacent hex cells.

        """
        row, col = coordinates

        up = (row - 1, col)
        down = (row + 1, col)
        right = (row, col - 1)
        left = (row, col + 1)
        up_right = (row - 1, col + 1)
        down_left = (row + 1, col - 1)

        candidates = [up, down, right, left, up_right, down_left]

        return [
            pair
            for pair in candidates
            if max(pair[0], pair[1]) <= self.size - 1
            and min(pair[0], pair[1]) >= 0
        ]

    def get_action_space(self, recode_blue_as_red=False, recode_black_as_white=False):
        """
        Return all empty cells.

        
        """
        actions = []

        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == EMPTY:
                    actions.append((row, col))

        if recode_blue_as_red or recode_black_as_white:
            return [self.recode_coordinates(action) for action in actions]

        return actions

    def _random_move(self):
        """Play one uniformly random valid move."""
        chosen = choice(self.get_action_space())
        self.move(chosen)

    def _random_match(self):
        """Play a full random game."""
        while self.winner == EMPTY:
            self._random_move()

    def _prolong_path(self, path):
        """
        Extend a path through stones of the same color.

        """
        player = self.board[path[-1][0]][path[-1][1]]
        candidates = self._get_adjacent(path[-1])

        # Prevent loops.
        candidates = [candidate for candidate in candidates if candidate not in path]

        # Only continue through stones of the same player.
        candidates = [
            candidate
            for candidate in candidates
            if self.board[candidate[0]][candidate[1]] == player
        ]

        return [path + [candidate] for candidate in candidates]

    def evaluate(self, verbose=False):
        """
        Evaluate whether red or blue has won.

        Order:
            first evaluate red,
            then evaluate blue.
        """
        self._evaluate_red(verbose=verbose)
        self._evaluate_blue(verbose=verbose)

    def _evaluate_red(self, verbose=False):
        """
        Check whether red has connected left to right.

        """
        paths = []
        visited = []

        for row in range(self.size):
            if self.board[row][0] == RED:
                paths.append([(row, 0)])
                visited.append([(row, 0)])

        while True:
            if len(paths) == 0:
                return False

            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)

                for new_path in prolongations:
                    if new_path[-1][1] == self.size - 1:
                        if verbose:
                            print("A winning path for red ('1'):\n", new_path)

                        self.winner = RED
                        return True

                    if new_path[-1] not in visited:
                        paths.append(new_path)
                        visited.append(new_path[-1])

    def _evaluate_blue(self, verbose=False):
        """
        Check whether blue has connected top to bottom.

        """
        paths = []
        visited = []

        for col in range(self.size):
            if self.board[0][col] == BLUE:
                paths.append([(0, col)])
                visited.append([(0, col)])

        while True:
            if len(paths) == 0:
                return False

            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)

                for new_path in prolongations:
                    if new_path[-1][0] == self.size - 1:
                        if verbose:
                            print("A winning path for blue ('-1'):\n", new_path)

                        self.winner = BLUE
                        return True

                    if new_path[-1] not in visited:
                        paths.append(new_path)
                        visited.append(new_path[-1])

    def machine_vs_machine(self, machine1=None, machine2=None):
        """
        Let two agents play.

        machine1 plays red.
        machine2 plays blue.

        If one machine is None, it plays randomly.
        """
        if machine1 is None:
            def machine1(board, action_list):
                return choice(action_list)

        if machine2 is None:
            def machine2(board, action_list):
                return choice(action_list)

        self.reset()

        while self.winner == EMPTY:
            if self.player == RED:
                chosen = machine1(self.board, self.get_action_space())

            if self.player == BLUE:
                chosen = machine2(self.board, self.get_action_space())

            # Safer for external agents.
            if chosen not in self.get_action_space():
                chosen = choice(self.get_action_space())

            self.move(chosen)

            if self.winner == RED:
                self._evaluate_red(verbose=False)

            if self.winner == BLUE:
                self._evaluate_blue(verbose=False)

        return self.winner

    def recode_blue_as_red(self, print=False, invert_colors=True):
        """
        Return a board where blue is recoded as red.

        This corresponds to flipping the board along the south-west
        to north-east diagonal and swapping colors.
        """
        flipped_board = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                original_value = self.board[self.size - 1 - col][self.size - 1 - row]

                if original_value == RED:
                    flipped_board[row][col] = BLUE

                if original_value == BLUE:
                    flipped_board[row][col] = RED

        return flipped_board

    # Compatibility alias with the original code.
    def recode_black_as_white(self, print=False, invert_colors=True):
        return self.recode_blue_as_red(print=print, invert_colors=invert_colors)

    def recode_coordinates(self, coordinates):
        """Transform coordinates according to recode_blue_as_red."""
        row, col = coordinates

        assert 0 <= row <= self.size - 1, "There is something wrong with the first coordinate."
        assert 0 <= col <= self.size - 1, "There is something wrong with the second coordinate."

        return self.size - 1 - col, self.size - 1 - row

    def coordinate_to_scalar(self, coordinates):
        """Convert coordinates to a scalar action."""
        row, col = coordinates

        assert 0 <= row <= self.size - 1, "There is something wrong with the first coordinate."
        assert 0 <= col <= self.size - 1, "There is something wrong with the second coordinate."

        return row * self.size + col

    def scalar_to_coordinates(self, scalar):
        """Convert a scalar action back to board coordinates."""
        row = int(scalar / self.size)
        col = scalar - row * self.size

        assert 0 <= row <= self.size - 1, "The scalar input is invalid."
        assert 0 <= col <= self.size - 1, "The scalar input is invalid."

        return row, col

    def replay_history(self):
        """Print the game history."""
        for board in self.history:
            temp = hexPosition(size=self.size)
            temp.board = board
            temp.print()
            input("Press ENTER to continue.")

    def save(self, path):
        """Serialize the game object."""
        import pickle

        with open(path, "ab") as file:
            pickle.dump(self, file)


class HexPygameApp:
    """
    Pygame interface for Hex.

    Direction:
        RED  connects LEFT ↔ RIGHT
        BLUE connects TOP ↕ BOTTOM
    """

    WIDTH = 1000
    HEIGHT = 700
    FPS = 60

    HEX_RADIUS = 32
    HEX_WIDTH = math.sqrt(3) * HEX_RADIUS
    HEX_VERTICAL_STEP = 1.5 * HEX_RADIUS

    BOARD_TOP = 95
    BOARD_LEFT = 145

    AGENT_DELAY_MS = 250

    COLORS = {
        "background": (245, 245, 245),
        "cell": (220, 220, 220),
        "hover": (255, 245, 170),
        "grid": (25, 25, 25),
        "red": (245, 20, 20),
        "blue": (0, 120, 255),
        "text": (30, 30, 30),
        "white_text": (255, 255, 255),
    }

    def __init__(self, size=7, red_agent=None, blue_agent=None, games_to_play=1):
        import pygame

        self.pygame = pygame
        self.game = hexPosition(size)

        self.red_agent = red_agent
        self.blue_agent = blue_agent

        self.games_to_play = games_to_play
        self.current_game_number = 1

        self.last_agent_move_time = 0
        self.waiting_after_game = False
        self.game_over_time = None

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Hex Game")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("arial", 18, bold=True)

        self.hex_centers = self._calculate_hex_centers()

    def _calculate_hex_centers(self):
        """

        Each next column moves right by HEX_WIDTH.
        Each next row moves down by 1.5 * radius.
        Each row is shifted right by half a hex width.
        """
        centers = {}

        for row in range(self.game.size):
            for col in range(self.game.size):
                x = (
                    self.BOARD_LEFT
                    + col * self.HEX_WIDTH
                    + row * self.HEX_WIDTH / 2
                )
                y = self.BOARD_TOP + row * self.HEX_VERTICAL_STEP
                centers[(row, col)] = (x, y)

        return centers

    def _hex_corners(self, center):
        """
        Return corners for a flat-top hexagon.
        """
        x, y = center
        corners = []

        for i in range(6):
            angle = math.radians(60 * i + 30)
            px = x + self.HEX_RADIUS * math.cos(angle)
            py = y + self.HEX_RADIUS * math.sin(angle)
            corners.append((px, py))

        return corners

    def _current_agent(self):
        if self.game.player == RED:
            return self.red_agent
        if self.game.player == BLUE:
            return self.blue_agent
        return None

    def _agent_step(self):
        if self.game.winner != EMPTY:
            return

        agent = self._current_agent()
        if agent is None:
            return

        now = self.pygame.time.get_ticks()

        if now - self.last_agent_move_time < self.AGENT_DELAY_MS:
            return

        actions = self.game.get_action_space()
        move = agent(self.game.board, actions)

        if move not in actions:
            move = choice(actions)

        self.game.move(move)
        self.last_agent_move_time = now

    def run(self):
        pygame = self.pygame
        running = True

        while running:
            self.clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._start_new_series()
                    elif event.key == pygame.K_n:
                        self._start_next_game()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (
                        event.button == 1
                        and self._current_agent() is None
                        and self.game.winner == EMPTY
                    ):
                        self._handle_click(event.pos)

            self._agent_step()
            self._handle_game_over_wait()
            self._draw()

        pygame.quit()

    def _handle_game_over_wait(self):
        """Keep the final board visible before moving to the next game."""
        if self.game.winner == EMPTY:
            return

        if not self.waiting_after_game:
            self.waiting_after_game = True
            self.game_over_time = self.pygame.time.get_ticks()
            return

        if self.current_game_number >= self.games_to_play:
            return

        elapsed = self.pygame.time.get_ticks() - self.game_over_time

        if elapsed >= 2000:
            self._start_next_game()

    def _start_new_series(self):
        self.current_game_number = 1
        self.game.reset()
        self.waiting_after_game = False
        self.game_over_time = None
        self.last_agent_move_time = 0

    def _start_next_game(self):
        if self.current_game_number < self.games_to_play:
            self.current_game_number += 1

        self.game.reset()
        self.waiting_after_game = False
        self.game_over_time = None
        self.last_agent_move_time = 0

    def _handle_click(self, position):
        cell = self._mouse_to_cell(position)

        if cell is not None:
            self.game.move(cell)

    def _mouse_to_cell(self, position):
        mouse_x, mouse_y = position

        closest_cell = None
        closest_distance = float("inf")

        for cell, center in self.hex_centers.items():
            x, y = center
            distance = math.hypot(mouse_x - x, mouse_y - y)

            if distance < closest_distance:
                closest_distance = distance
                closest_cell = cell

        if closest_distance <= self.HEX_RADIUS:
            return closest_cell

        return None

    def _draw(self):
        pygame = self.pygame
        self.screen.fill(self.COLORS["background"])

        self._draw_goal_edges()
        self._draw_direction_labels()
        self._draw_board()
        self._draw_status_text()

        pygame.display.flip()

    def _draw_board(self):
        for row in range(self.game.size):
            for col in range(self.game.size):
                self._draw_cell(row, col)

    def _draw_goal_edges(self):
        """
        Draw colored goal sides.

        RED goal sides:
            left and right board sides

        BLUE goal sides:
            top and bottom board sides
        """
        pygame = self.pygame
        size = self.game.size

        top_left = self.hex_centers[(0, 0)]
        top_right = self.hex_centers[(0, size - 1)]
        bottom_left = self.hex_centers[(size - 1, 0)]
        bottom_right = self.hex_centers[(size - 1, size - 1)]

        pygame.draw.line(
            self.screen,
            self.COLORS["blue"],
            top_left,
            top_right,
            18,
        )

        pygame.draw.line(
            self.screen,
            self.COLORS["blue"],
            bottom_left,
            bottom_right,
            18,
        )

        pygame.draw.line(
            self.screen,
            self.COLORS["red"],
            top_left,
            bottom_left,
            18,
        )

        pygame.draw.line(
            self.screen,
            self.COLORS["red"],
            top_right,
            bottom_right,
            18,
        )

    def _draw_direction_labels(self):
        """
        Direction comments shown on the interface.
        """
        red_text = "RED: connect LEFT ↔ RIGHT"
        blue_text = "BLUE: connect TOP ↕ BOTTOM"

        red_surface = self.small_font.render(red_text, True, self.COLORS["red"])
        blue_surface = self.small_font.render(blue_text, True, self.COLORS["blue"])

        self.screen.blit(red_surface, (30, 20))
        self.screen.blit(blue_surface, (30, 45))

    def _draw_status_text(self):
        if self.game.winner == RED:
            status = "RED wins! Final board shown. Press N for next game or R to restart."
        elif self.game.winner == BLUE:
            status = "BLUE wins! Final board shown. Press N for next game or R to restart."
        else:
            current = "RED" if self.game.player == RED else "BLUE"
            status = f"{current}'s turn"

        game_counter = f"Game {self.current_game_number} / {self.games_to_play}"
        help_text = "R: restart series | N: next game | Esc: quit"

        self.screen.blit(
            self.font.render(status, True, self.COLORS["text"]),
            (30, self.HEIGHT - 85),
        )
        self.screen.blit(
            self.small_font.render(game_counter, True, self.COLORS["text"]),
            (30, self.HEIGHT - 55),
        )
        self.screen.blit(
            self.small_font.render(help_text, True, self.COLORS["text"]),
            (30, self.HEIGHT - 30),
        )

    def _draw_cell(self, row, col):
        pygame = self.pygame

        center = self.hex_centers[(row, col)]
        x, y = center
        corners = self._hex_corners(center)

        fill = self.COLORS["cell"]

        if (
            self._mouse_to_cell(pygame.mouse.get_pos()) == (row, col)
            and self.game.board[row][col] == EMPTY
            and self.game.winner == EMPTY
            and self._current_agent() is None
        ):
            fill = self.COLORS["hover"]

        pygame.draw.polygon(self.screen, fill, corners)
        pygame.draw.polygon(self.screen, self.COLORS["grid"], corners, 2)

        value = self.game.board[row][col]

        if value == RED:
            pygame.draw.circle(
                self.screen,
                self.COLORS["red"],
                (int(x), int(y)),
                int(self.HEX_RADIUS * 0.68),
            )

        elif value == BLUE:
            pygame.draw.circle(
                self.screen,
                self.COLORS["blue"],
                (int(x), int(y)),
                int(self.HEX_RADIUS * 0.68),
            )