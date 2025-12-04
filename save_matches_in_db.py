import pandas as pd
import sqlite3

def main():

    conn = sqlite3.connect('db/cricket.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                                        match_id VARCHAR NOT NULL,
                                        match_number INTEGER NOT NULL, 
                                        match_date timestamp,
                                        venue VARCHAR NOT NULL, 
                                        season VARCHAR NOT NULL, 
                                        city VARCHAR NOT NULL, 
                                        is_night_match boolean NOT NULL,
                                        winner_team_name VARCHAR, 
                                        loser_team_name VARCHAR, 
                                        team_1_team_name VARCHAR NOT NULL, 
                                        team_2_team_name VARCHAR NOT NULL, 
                                        win_by VARCHAR, 
                                        win_by_type VARCHAR
    )
    """)

    with open("output/matches_table.csv") as matches_file:
        pd.read_csv(matches_file).to_sql('matches', conn, if_exists='replace')


if __name__ == '__main__':
    main()