import json
import os
from typing import Any

import pandas as pd

from constants import DATA_IPL_JSON_PATH


def is_night_match(match, all_matches_df: pd.DataFrame) -> bool:
    matches_on_same_day = all_matches_df.loc[all_matches_df['date'] == match['date']]
    matches = matches_on_same_day.loc[matches_on_same_day['match_number'] > match['match_number']]
    return not len(matches) > 0


def main():
    match_df = extract_match_data()
    match_df.to_csv("output/matches_table.csv", index=False)


def extract_loser(winner, teams):
    for team in teams:
        if winner != team:
            return team
    return None


def add_teams(match_data: dict, teams: list[str]):
    for i, team in enumerate(teams):
        match_data["team_" + str(i+1)+"_team_name"] = team


def extract_match_data() -> pd.DataFrame:
    # global matches_df
    ipl_match_files = os.listdir((DATA_IPL_JSON_PATH))
    all_matches = []
    for ipl_match_file_name in ipl_match_files:
        match_data = {}
        try:
            if not ipl_match_file_name.endswith(".json"):
                continue
            with open(os.path.join(DATA_IPL_JSON_PATH, ipl_match_file_name), 'r') as ipl_match_file:
                match_obj = json.load(ipl_match_file)
                info = match_obj.get('info', {})
                teams = list(info.get("players").keys())
                add_teams(match_data, teams)
                match_data['winner_team_name'] = info.get('outcome').get("winner") if "winner" in info.get('outcome') else None
                match_data["loser_team_name"] = extract_loser(match_data["winner_team_name"],
                                                    teams) if "by" in info.get("outcome") else None
                match_data["win_by_type"] = list(info.get("outcome").get("by").keys())[0] if info.get(
                    "outcome") and info.get("outcome").get("by") else None
                match_data["win_by"] = info.get("outcome").get("by").get(match_data["win_by_type"]) if info.get(
                    "outcome") and info.get("outcome").get("by") else None
                match_data['match_id'] = ipl_match_file_name.replace(".json", "")
                match_data['date'] = info['dates'][0]
                match_data['match_date'] = match_data.get('date')
                match_data['venue'] = info['venue']
                match_data['season'] = info['season']
                match_data['city'] = info['city'] if 'city' in info else ''
                match_data['match_number'] = info['event']['match_number'] if 'match_number' in info['event'] else \
                    info['event']['stage']
                all_matches.append(match_data)
        except Exception as e:
            print(ipl_match_file_name)

    matches_df = pd.DataFrame(all_matches)
    with open('data/stadiums.json', 'r') as f:
        venue_mappings = json.load(f)
        matches_df['venue'] = matches_df['venue'].apply(lambda x: venue_mappings[x] if x in venue_mappings else x)
        matches_df['city'] = matches_df.apply(
            lambda row: 'Dubai' if row['venue'].startswith('Dubai') or row['venue'].startswith('Sharjah') else row[
                'city'], axis=1)
        matches_df['is_night_match'] = matches_df.apply(lambda row: is_night_match(row, matches_df), axis=1)
    return matches_df


if __name__ == '__main__':
    main()
