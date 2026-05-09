from hex_engine import HexPygameApp
from submission.facade import agent
from submission.greedy_agent import greedy_agent

import os

def ask_mode():
    print("\nChoose mode:")
    print("1 - Human vs Human")
    print("2 - Human Red vs Agent Blue")
    print("3 - Agent Red vs Human Blue")
    print("4 - Agent vs Agent")
    print("5 - Greedy Red vs DQN Blue")
    print("6 - DQN Red vs Greedy Blue")

    choice = input("Mode: ").strip()

    if choice == "2":
        return None, agent
    if choice == "3":
        return agent, None
    if choice == "4":
        return agent, agent
    if choice == "5":
        return greedy_agent, agent
    if choice == "6":
        return agent, greedy_agent

    return None, None



if __name__ == "__main__":
    size = int(input("Board size: "))

    model_path = os.path.join("submission", f"dqn_{size}x{size}.pt")

    if not os.path.exists(model_path):
        print(f"⚠️  No trained model found at {model_path}")
        print(f"    Run train.py first with size={size}")
        exit()

    games = int(input("Number of games to play: "))

    red_agent, blue_agent = ask_mode()

    app = HexPygameApp(
        size=size,
        red_agent=red_agent,
        blue_agent=blue_agent,
        games_to_play=games,
    )


    app.run()