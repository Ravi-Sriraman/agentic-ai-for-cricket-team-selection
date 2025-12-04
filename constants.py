import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ['DATABASE_URL']
LLM_MODEL = "gpt-4o"
LLM_PROVIDER = "openai"

DATA_IPL_JSON_PATH = './data/ipl_json'