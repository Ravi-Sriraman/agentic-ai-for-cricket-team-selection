pip install -r requirements.txt
python setup_data.py
sqlite update_team_names.sql
python evaluate_team_selector.py
python evaluate_cricket_analyst.py