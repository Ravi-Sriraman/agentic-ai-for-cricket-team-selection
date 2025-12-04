import create_matches_table
import create_ball_to_ball_table
import create_current_squad_table
import save_matches_in_db
import save_ball_to_ball_table_in_db
import save_ipl_squads_in_db

def main():
    create_matches_table.main()
    create_ball_to_ball_table.main()
    create_current_squad_table.main()
    save_matches_in_db.main()
    save_ball_to_ball_table_in_db.main()
    save_ipl_squads_in_db.main()

if __name__ == '__main__':
    main()