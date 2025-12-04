import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ['DATABASE_URL']
LLM_MODEL = "gpt-4o"
LLM_PROVIDER = "openai"

