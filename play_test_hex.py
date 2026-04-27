import hex_engine as engine
from submission.facade import agent


game = engine.hexPosition()

game.machine_vs_machine(
    machine1=None,
    machine2=agent
)

print("Winner:", game.winner)
game.print()