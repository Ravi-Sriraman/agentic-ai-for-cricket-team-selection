from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

def create_pick_top_bowlers_agent():
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
                
        Example: Who are the best batsmen in RCB?. 
        Steps:
        1. Find the list of bowlers in the current RCB squad by running a query like (select player_name from ipl_squads where team_name='RCB' and player_role='Bowler'). 
        2. Fetch number of balls bowled and number of runs conceded by each bowler in the current squad by running a query like (select batter, SUM(runs_conceded_by_bowler) as runs_conceded, COUNT(*) as balls_bowled  from ball_to_ball_data where bowler IN ('Hazlewood', 'B Kumar') group by bowler 
        3. Count number of wickets took by each bowler by running a sql query like (select bowler, count(name_of_the_player_got_out) as number_of_wickets from ball_to_ball_data group where name_of_the_player_got_out is not null by bowler order by bowler). 
        4. Compute number of overs bowled by diving number of balls bowled by 6.
        5. Compute bowler's strike rate and bowler's average.
        6. return the top 10 result in JSON format with fields player_name, average, strike_rate, economy, balls_bowled, runs_conceded.
        """.format(dialect=db.dialect,top_k=5)

    llm = init_chat_model(model="gpt-4o", model_provider="openai")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt, name="pick_top_bowlers_agent")


def main():
    load_dotenv()
    agent = create_pick_top_bowlers_agent()
    question = "Find top 10 bowlers in RR and return their performance metrics in JSON format"
    for step in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()

if __name__ == '__main__':
    main()