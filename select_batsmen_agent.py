from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from constants import LLM_MODEL, LLM_PROVIDER
from ranker import compute_batting_performance, rank_list_of_players_by_field


def create_pick_top_batsmen_agent():
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
                
        Example: Who are the best batsmen in SRH?. 
        Steps:
        1. Find the list of batsmen in the current SRH squad by running a query like (select player_name from ipl_squads where team_name='SRH' and player_role='Batter' or player_role='WK-Batter'). 
        2. Fetch number of balls faced and number of runs scored by each batsman in the current squad by running a query like (select batter, SUM(runs_scored_by_batsman) as total_runs, COUNT(*) as balls_faced  from ball_to_ball_data where batter IN ('Ishan Kishan', 'Klaasen') group by batter 
        3. Compute strike rate of each batsman by using balls_faced total_runs fields.
        4. Fetch number of times each wicket keeper got out by running a query.
        5. Compute batting average by dividing total_runs by no of times the batsman got out.
        6. FORMAT the player's performance in JSON format with fields player_name, batting_average, batting_strike_rate
        7. Compute batting_performance_index and pick the top N batsmen based one the query 
        8. Format data in JSON format and rank the players by batting_performance_index in descending order.
        9. Format the player's performance in JSON format with fields player_name, average, strike_rate, performance_index (By default N is 6)..
        """.format(dialect=db.dialect,top_k=5)

    llm = init_chat_model(model=LLM_MODEL, model_provider=LLM_PROVIDER)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = [*toolkit.get_tools(), compute_batting_performance, rank_list_of_players_by_field]
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, name="select_batsmen_agent")


def main():
    load_dotenv()
    agent = create_pick_top_batsmen_agent()
    question = "Find all batsmen in DC and their performance in JSON format"
    for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()

if __name__ == '__main__':
    main()