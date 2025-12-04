import requests, time
from bs4 import BeautifulSoup
import pandas as pd


TEAMS_BASE  = "https://www.iplt20.com/teams"

team_names = [
    "chennai-super-kings",
    "delhi-capitals",
    "gujarat-titans",
    "kolkata-knight-riders",
    "lucknow-super-giants",
    "mumbai-indians",
    "punjab-kings",
    "rajasthan-royals",
    "royal-challengers-bengaluru",
    "sunrisers-hyderabad",
]

def create_current_squad_objects():
    """This function creates the current squad objects based on the players names
       The columns include player_name,team_name, player_short_name
    """


    players = []
    for team_name in team_names:
        url = TEAMS_BASE + "/" + team_name
        html = requests.get(url, headers={}, timeout=20).text
        tsoup = BeautifulSoup(html, "html.parser")
        # team name
        batsmen = [s for s in tsoup.select_one("#identifiercls0").getText().split("\n") if s]
        for i in range(0, len(batsmen), 2):
            batsman = {"name": batsmen[i], "player_role": batsmen[i + 1], "team_name": team_name}
            players.append(batsman)

        all_rounders = [s for s in tsoup.select_one("#identifiercls1").getText().split("\n") if s]
        for i in range(0,len(all_rounders), 2):
            all_rounder = {"name": all_rounders[i], "player_role": all_rounders[i + 1], "team_name": team_name}
            players.append(all_rounder)

        bowlers = [s for s in tsoup.select_one("#identifiercls2").getText().split("\n") if s]
        for i in range(0, len(bowlers), 2):
            bowler = {"name":bowlers[i], "player_role": bowlers[i + 1], "team_name": team_name}
            players.append(bowler)

    df = pd.DataFrame(players).drop_duplicates()
    df.to_csv("ipl_2025_squads.csv", index=False)
    print(df.head(), "\nSaved to ipl_2025_squads.csv")


def main():
    create_current_squad_objects()

if __name__ == '__main__':
    main()