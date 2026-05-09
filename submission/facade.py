from .facade_dqn    import agent as agent_dqn
from .greedy_agent  import greedy_agent

def agent(board, action_set):
   return agent_dqn(board, action_set)
  #  return greedy_agent(board, action_set)