import json
import os
from constants import DATA_IPL_JSON_PATH
import pandas as pd

PLAYER_NAMES_FILE_PATH = 'output/player_names.csv'

def get_short_name(name, player_names_df: pd.DataFrame):
    parts = name.split(" ")
    player_names = player_names_df.loc[player_names_df['name'].str.endswith(parts[-1]) & player_names_df['name'].str.startswith(parts[0][0])]
    if len(player_names) == 1:
        return player_names["name"].iloc[0]


    player_names_new = player_names.loc[player_names['name']==name]
    if len(player_names_new) > 0 or len(player_names) == 0:
        return name
    print(f"duplicate names found for {name}")
    for n in player_names['name']:
        print(n, end=" ")
    print()
    return "_"

def create_player_name_mapping_table():
    player_names_df = pd.read_csv(PLAYER_NAMES_FILE_PATH)
    ipl_player_names = pd.read_csv("ipl_2025_squads.csv")
    ipl_player_names["short_name"] = ipl_player_names["name"].apply(lambda name: get_short_name(name, player_names_df))
    ipl_player_names.rename(columns={"name":"full_name"}, inplace=True)
    ipl_player_names.rename(columns={"short_name":"player_name"}, inplace=True)
    ipl_player_names.to_csv(PLAYER_NAMES_FILE_PATH, index=False)

def main():
    create_player_names_table()
    create_player_name_mapping_table()

def create_player_names_table():
    ipl_file_names = os.listdir(DATA_IPL_JSON_PATH)
    matches = []
    player_names = set()
    for ipl_file_name in ipl_file_names:
        if not ipl_file_name.endswith('.json'):
            continue
        with open(os.path.join(DATA_IPL_JSON_PATH, ipl_file_name)) as ipl_file:
            match_obj = json.load(ipl_file)
            matches.append(match_obj)
            player_names.update(*match_obj['info']['players'].values())
    pd.DataFrame({"name": list(player_names)}).to_csv(PLAYER_NAMES_FILE_PATH, index=False)


if __name__ == '__main__':
    main()