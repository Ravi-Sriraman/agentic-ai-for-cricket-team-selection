from typing import List, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph

from constants import LLM_MODEL, LLM_PROVIDER


def create_cricket_analyst_agent():
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
        
        ALWAYS PLAN and BREAK the query into multiple simple queries. execute the simple queries and combine the outputs from the simple queries
        to answer the final question/query. For example: 
        
         Example : What is SP Narine's batting average against RCB?.
                Steps:
                1. Find number of matches SP Narine got out against RCB (select  count(*) from ball_to_ball_data where name_of_the_player_got_out LIKE "%Narine"
                and bowling_team = 'RCB')
                2. Find total number of runs SP Narine scored against RCB (select sum(runs_scored_by_batsman) as total_runs from ball_to_ball_data where batter LIKE "%Narine"
                and bowling_team = 'RCB')
                3. Divide number of runs SP Narine scored against RCB by number of matches he got out against RCB. (333/11) = 30.2
                
         Example : Rank RCB Batsmen based on their striker rate?.
                Steps:
                1. Find batsmen in the current RCB squad by running a query like (select player_name from ipl_squads where team_name='RCB')
                2. Fetch number of balls faced and number of scored by each batsmen in the current squad by running a query like (select batter, SUM(runs_scored_by_batsman) as total_runs, COUNT(*) as balls_faced  from ball_to_ball_data where batter IN ('V Kohli', 'DM Karthick') group by batter 
                3. Compute strike rate of each batsman by using balls_faced total_runs fields.
                4. Sort the results based on the request.
                5. return the list in JSON format with the keys "player_name", "strike_rate", "balls_faced" "total_runs"
        
        """.format(dialect=db.dialect,top_k=5)

    llm = init_chat_model(model=LLM_MODEL, model_provider=LLM_PROVIDER, temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = [*toolkit.get_tools(), sort_players_based_on_performance]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, name="cricket_analyst_agent", state_schema=MessagesState)

@tool
def sort_players_based_on_performance(players_performance: List[dict[str, object]], sort_by:List[str]) -> List[dict[str, object]]:
    """
    Get a list of players' statistics and field names by which they are to be sorted.

    Returns the sorted list
    """
    return sorted(players_performance, key=lambda x: tuple([x[k] for k in sort_by]))

def main():
    load_dotenv()
    current_squad_agent = create_cricket_analyst_agent()
    # player_statistics_agent = create_player_statistics_agent()
    question = "Who has scored the most runs for CSK during 2024 and 2025?"
    for step in current_squad_agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()



if __name__ == '__main__':
    main()
