from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor
from fetch_current_squad_agent import create_fetch_current_squad_agent
from fetch_player_statistics import create_player_statistics_agent
from select_batsmen_agent import create_pick_top_batsmen_agent
from select_bowlers_agent import create_pick_top_bowlers_agent
from select_wicket_keeper_agent import create_best_wicket_keeper_agent
from select_all_rounders_agent import create_pick_top_all_rounders_agent

def create_supervisor_agent():
    llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
    fetch_current_squad_agent = create_fetch_current_squad_agent()
    fetch_player_statistics_agent = create_player_statistics_agent()
    select_wicket_keeper_agent = create_best_wicket_keeper_agent()
    select_bowlers_agent = create_pick_top_bowlers_agent()
    select_all_rounders_agent = create_pick_top_all_rounders_agent()
    select_batsmen_agent = create_pick_top_batsmen_agent()
    return create_supervisor(
        agents=[select_wicket_keeper_agent, select_bowlers_agent, select_batsmen_agent, select_all_rounders_agent],
        model=llm,
        prompt="""You coordinate specialist agents for to build the best playing 11 for a cricket team.
        
       ALWAYS PLAN and BREAK the query into multiple simple queries. execute the simple steps and combine the outputs from the simple steps
       to answer the final question/query. For example: 
                
        Example: Select a playing 11 for CSK by using the performance metrics in JSON format?. 
        Steps:
        1. Call select_wicket_keeper_agent with a user query like (Select best wicket keeper batsman in CSK team and return in JSON format). 
        2. Call select_all_rounders_agent with a user query like (Find all All-Rounders performance of CSK team in JSON format).
        3. Call select_bowlers_agent with a user query like (Find top 10 bowlers in CSK and return their performance metrics in JSON format). 
        4. Call select_batsmen_agent with a user query like (Find all batsmen in CSK and their performance in JSON format).
        5. Come up with a plan on selecting number of all rounders, pure batsmen, pure bowlers based on the location if it is provided in the user's query else a generic plan. for example (bowlers: 4, batsmen: 4, wicket_keeper: 1, all-rounders: 2).
        6. Pick top performers in each role and return a 11 member squad with their performance metrics. for example (batters & wicket_keeper: name, batting_strike_rate, batting_average), (bowlers: name, economy, bowling_strike_rate, bowling_average), (all_rounders: batting_average, batting_strike_rate, bowling_average, bowling_strike_rate, bowling_economy)
        """
    ).compile()

a = """
        MAKE plan and call specialist agents to fetch required data, Using the data received from the specialist agents answer the user's question. DO NOT let the specialist agents solve everything.
        To answer the 1st type question like player statistics, team statistics, or venue statistics make use of generic_sql_agent.
        To handle the 2nd type request use specialist agents such as select_bowlers_agent, select_batsmen_agent, select_wicket_keeper_agent, and select_bowlers_keeper_agent.
"""

def pretty_print_message(message, indent=False):
    """Pretty print a message."""
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    """Pretty print messages from agent updates."""
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"\nUpdate from subgraph {graph_id}:")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"\nUpdate from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print()


def main():
    load_dotenv()
    agent = create_supervisor_agent()
    question = "Select a playing 11 for CSK by using the performance metrics in JSON format"
    for chunk in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="updates"):
        pretty_print_messages(chunk)

if __name__ == '__main__':
    main()