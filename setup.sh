sudo apt update -y
sudo apt install -y unzip
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y python3-virtualenv
virtualenv venv
source ./venv/bin/activate
./venv/bin/pip install -r requirements.txt
sudo apt install -y sqlite3
export DATABASE_URL=sqlite:///db/cricket.db
./venv/bin/python3 download_dataset.py
unzip ./data/ipl_json.zip -d ./data/ipl_json
./venv/bin/python3 setup_data.py
sqlite3 db/cricket.db < update_team_names.sql
./venv/bin/python3 evaluate_team_selector.py
./venv/bin/python3 evaluate_cricket_analyst.py