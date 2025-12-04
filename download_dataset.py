import os

import requests


def main():
    response = requests.get("https://cricsheet.org/downloads/ipl_json.zip")
    if not os.path.exists("./data/ipl_json"):
        os.mkdir("data/ipl_json")
    with open("./data/ipl_json.zip", "wb") as f:
        f.write(response.content)

if __name__ == '__main__':
    main()