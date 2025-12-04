import sqlite3

import pandas as pd


def main():

    conn = sqlite3.connect('db/cricket.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ball_to_ball_data (id INTEGER AUTO_INCREMENT PRIMARY KEY NOT NULL, match_id VARCHAR NOT NULL, 
                                                  delivery INT NOT NULL, batter VARCHAR NOT NULL, bowler VARCHAR NOT NULL, 
        total_runs INT NOT NULL, batter_runs INT NOT NULL, bowler_runs INT NOT NULL, over_number INT NOT NULL, batting_team VARCHAR NOT NULL, bowling_team VARCHAR NOT NULL, batter_got_out VARCHAR, wicket_type VARCHAR 
    )
    """)

    with open("./output/ball_by_ball_data_new.csv") as file:
        ball_data_df = pd.read_csv(file)
        ball_data_df.to_sql("ball_to_ball_data", conn, if_exists="replace")

    cursor.close()

if __name__ == '__main__':
    main()