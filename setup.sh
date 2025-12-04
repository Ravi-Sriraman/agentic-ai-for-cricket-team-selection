sudo apt update -y
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y python3-virtualenv
sudo virtualenv venv
source ./venv/bin/activate
./venv/bin/pip install -r requirements.txt
sudo apt install -y sqlite3
export DATABASE_URL=sqlite:///db/cricket.db
./venv/bin/python3 setup_data.py
sqlite update_team_names.sql
./venv/bin/python3 evaluate_team_selector.py
./venv/bin/python3 evaluate_cricket_analyst.py