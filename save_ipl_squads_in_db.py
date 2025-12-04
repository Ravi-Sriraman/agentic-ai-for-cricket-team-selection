import pandas as pd
import sqlite3

def main():
    conn = sqlite3.connect("db/cricket.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ipl_squads (id INTEGER PRIMARY KEY AUTOINCREMENT, player_name varchar(30) not null, role varchar(30) not null, team varchar(40) not null, short_name varchar(40) not null)
    """)

    with open("output/player_names.csv") as player_names_file:
        player_names = pd.read_csv(player_names_file)
        player_names.to_sql("ipl_squads", conn, if_exists="replace")
    cursor.close()

if __name__ == '__main__':
    main()