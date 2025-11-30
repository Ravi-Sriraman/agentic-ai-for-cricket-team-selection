from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.graph import MessagesState


def create_player_statistics_agent():
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
        
        DO NOT EXECUTE QUERIES ON ANY TABLE EXCEPT THE TABLE ball_to_ball_data.
        
        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.
        
        ALWAYS CALCULATE ONE PLAYER STATISTIC AT A TIME by USING WHERE CLAUSE like where player_name='V Kohli'
        
        ALWAYS PLAN and BREAK the query into multiple simple queries. execute the simple queries and combine the outputs from the simple queries
        to answer the final question/query. For example:
        
       
        """.format(dialect=db.dialect,top_k=5)

    llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = [*toolkit.get_tools(), sort_players_based_on_performance]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, name="fetch_one_player_statistics_agent", state_schema=MessagesState)

@tool
def sort_players_based_on_performance(players_performance: List[dict[str, object]], sort_by:List[str]) -> List[dict[str, object]]:
    """
    Get a list of players' statistics and field names by which they are to be sorted.

    Returns the sorted list
    """

    return sorted(players_performance, key=lambda x: tuple([x[k] for k in sort_by]))

def main():
    load_dotenv()
    agent = create_player_statistics_agent()
    question = "What is V Kohli's batting average?"
    for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()

if __name__ == '__main__':
    main()
