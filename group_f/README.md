# Hexgame with Reinforcement Learning
By Group F

## Members
- Chorn Philipp
- Hromada Stefan
- Wang Li Wen
- Wirawan Cahya

## Agent's strategy: Alphazero
AlphaZero MCTS in Hex pairs a neural network with Monte Carlo Tree Search. The network reads the board (from the side-to-move's view — handy in Hex since recode_blue_as_red() lets Blue reuse a Red-trained net) and outputs a policy (move priors) and a value (who's winning). MCTS then builds a search tree, descending via the PUCT rule that balances the average value of a move against its network prior and visit count; at each leaf it uses the network's value estimate rather than a random playout, expands the children, and backpropagates the value with a sign flip at each level (zero-sum). After a fixed number of simulations, it plays the most-visited move. The network is trained by self-play, where the search's visit counts become policy targets and game outcomes become value targets, feeding stronger searches in a self-improving loop — a great fit for Hex since it never draws and is symmetric across colors.

## The Model and code
The directory contains only the model for the board size 5, 7 and 9
due to the maximum upload in Moodle.
Please download the completed models at 
https://github.com/philippchn/RL-Hexgame/raw/refs/heads/master/group_f.tgz

## Notes
The MCTS code is written in C++, therefore please compile it before running the test as follow:
    pip install pybind11
    python setup_mcts.py build_ext --inplace 
