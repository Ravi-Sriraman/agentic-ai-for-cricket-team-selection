import requests


def main():
    response = requests.get("https://cricsheet.org/downloads/ipl_json.zip")
    with open("data/ipl_json.zip", "rb") as f:
        f.write(response.content)

if __name__ == '__main__':
    main()