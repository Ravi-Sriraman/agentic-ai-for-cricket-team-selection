from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.graph import MessagesState


def create_fetch_current_squad_agent():
    db = SQLDatabase.from_uri("sqlite:///db/cricket.db")
    system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. 
        
        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.
        
        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.
        
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.
        
        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.
        
        DO NOT EXECUTE QUERIES ON ANY TABLE EXCEPT THE TABLE ipl_squads.
        
        """.format(dialect=db.dialect,top_k=5)

    llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = [*toolkit.get_tools(), sort_players_based_on_performance]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, name="fetch_current_squad_agent", state_schema=MessagesState)

@tool
def sort_players_based_on_performance(players_performance: List[dict[str, object]], sort_by:List[str]) -> List[dict[str, object]]:
    """
    Get a list of players' statistics and field names by which they are to be sorted.

    Returns the sorted list
    """

    return sorted(players_performance, key=lambda x: tuple([x[k] for k in sort_by]))

def main():
    load_dotenv()
    agent = create_fetch_current_squad_agent()
    question = "Who has the highest batting average in KKR team?"
    for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()

if __name__ == '__main__':
    main()
