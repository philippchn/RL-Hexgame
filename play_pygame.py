from hex_engine import HexPygameApp
from submission.facade import agent


def ask_mode():
    print("\nChoose mode:")
    print("1 - Human vs Human")
    print("2 - Human Red vs Agent Blue")
    print("3 - Agent Red vs Human Blue")
    print("4 - Agent vs Agent")

    choice = input("Mode: ").strip()

    if choice == "2":
        return None, agent
    if choice == "3":
        return agent, None
    if choice == "4":
        return agent, agent

    return None, None


if __name__ == "__main__":
    size = int(input("Board size: "))
    games = int(input("Number of games to play: "))

    red_agent, blue_agent = ask_mode()

    app = HexPygameApp(
        size=size,
        red_agent=red_agent,
        blue_agent=blue_agent,
        games_to_play=games,
    )

    app.run()