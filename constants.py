import os

from dotenv import load_dotenv
from langchain_community.document_loaders.notiondb import DATABASE_URL

load_dotenv()

DATABASE_URL = os.environ['DATABASE_URL']
LLM_MODEL = "gpt-4.1"
LLM_PROVIDER = "openai"

