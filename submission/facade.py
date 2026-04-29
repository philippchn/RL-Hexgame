from .facade_dqn import agent as agent_dqn
    
def agent (board, action_set):
    return agent_dqn(board, action_set)
