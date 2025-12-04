import json
import os

import pandas as pd

from constants import DATA_IPL_JSON_PATH

def get_bowling_team_name(batting_team_name, teams: list[str]) -> str | None:
    for team in teams:
        if batting_team_name != team:
            return team
    return None


def calculate_bowler_runs(delivery)-> int:
    ravi: dict = {}
    total = delivery['runs']['batter']
    if 'extras' in delivery:
        for extra_key, extra_value in delivery['extras'].items():
            if extra_key != 'legbyes' or extra_key == 'byes':
                total += extra_value
    return total

def main():
    ipl_file_names = os.listdir(DATA_IPL_JSON_PATH)
    ball_by_ball_data = []
    for ipl_file_name in ipl_file_names:
        if not ipl_file_name.endswith('.json'):
            continue
        with open(os.path.join(DATA_IPL_JSON_PATH, ipl_file_name), 'r') as ipl_file:
            match_obj = json.load(ipl_file)
            innings = match_obj['innings']
            team_names = [i['team'] for i in innings]
            for inning in innings:
                team_name = inning['team']
                bowling_team_name = get_bowling_team_name(inning['team'], team_names)
                for over in inning['overs']:
                    over_number = over['over'] + 1
                    ball_number = 1
                    for delivery in over['deliveries']:
                        ball_data = {}
                        ball_data['delivery'] = ball_number
                        ball_data['batter'] = delivery['batter']
                        ball_data['bowler'] = delivery['bowler']
                        ball_data['total_runs'] = delivery['runs']['total']
                        ball_data['runs_scored_by_batsman'] = delivery['runs']['batter']
                        ball_data['runs_conceded_by_bowler'] = calculate_bowler_runs(delivery)
                        ball_data['over_number'] = over_number
                        ball_data['batting_team'] = team_name
                        ball_data['bowling_team'] = bowling_team_name
                        ball_data['match_id'] = ipl_file_name.replace('.json', '')
                        if not (delivery['runs']['extras'] and isinstance(delivery['runs']['batter'], dict) and ("wides" in delivery['runs']['extras'] or "noballs" in delivery['runs']['extras'])):
                            ball_number += 1
                        ball_data['name_of_the_player_got_out'] = delivery['wickets'][0]['player_out'] if 'wickets' in delivery else ''
                        ball_data['the_way_batter_got_out'] = delivery['wickets'][0]['kind'] if 'wickets' in delivery else ''
                        ball_by_ball_data.append(ball_data)
    with open("output/ball_by_ball_data_new.csv", "w") as new_file:
        pd.DataFrame(ball_by_ball_data).to_csv(new_file, index=False)

if __name__ == '__main__':
    main()