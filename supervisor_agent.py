from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor

from constants import LLM_MODEL, LLM_PROVIDER
from select_batsmen_agent import create_pick_top_batsmen_agent
from select_bowlers_agent import create_pick_top_bowlers_agent
from select_wicket_keeper_agent import create_best_wicket_keeper_agent
from select_all_rounders_agent import create_pick_top_all_rounders_agent

def create_supervisor_agent():
    llm = init_chat_model(model=LLM_MODEL, model_provider=LLM_PROVIDER, temperature=0)
    select_wicket_keeper_agent = create_best_wicket_keeper_agent()
    select_bowlers_agent = create_pick_top_bowlers_agent()
    select_all_rounders_agent = create_pick_top_all_rounders_agent()
    select_batsmen_agent = create_pick_top_batsmen_agent()
    return create_supervisor(
        agents=[select_wicket_keeper_agent, select_bowlers_agent, select_batsmen_agent, select_all_rounders_agent],
        model=llm,
        prompt="""You coordinate specialist agents to pick the best players.
        
       ALWAYS STRICTLY Use the following plan: 
       
        Example: Select a playing 11 for CSK by using the performance metrics in JSON format?. 
        Steps:
        1. Come up with a plan on selecting number of all rounders, pure batsmen, pure bowlers based on the location if it is provided in the user's query else a generic plan. for example (bowlers: 4, batsmen: 4, wicket_keeper: 1, all-rounders: 2).
        2. Call select_wicket_keeper_agent with a user query like (Select best wicket keeper batsman in CSK team and return in JSON format with fields player_name, batting_strike_rate, batting_average, performance_index). 
        3. Call select_all_rounders_agent with a user query like (Find all All-Rounders performance of CSK team in JSON format with fields player_name, batting_strike_rate, batting_average, bowling_average, bowling_economy, bowling_strike_rate, performance_index).
        4. Call select_bowlers_agent with a user query like (Find top 10 bowlers in CSK and return their performance metrics in JSON format with fields player_name, bowling_average, bowling_economy, bowling_strike_rate, performance_index). 
        5. Call select_batsmen_agent with a user query like (Find all batsmen in CSK and their performance in JSON format with fields player_name, batting_strike_rate, batting_average, performance_index).
        6. Compose a 11 member team from the above players(the current squad) WHEN ALL TOOLS ARE CALLED AT LEAST ONCE and then return their performance metrics with the required fields. for example (batters & wicket_keeper: name, batting_strike_rate, batting_average), (bowlers: name, economy, bowling_strike_rate, bowling_average), (all_rounders: batting_average, batting_strike_rate, bowling_average, bowling_strike_rate, bowling_economy)
        
        DO NOT ASK QUESTIONS.
        """
    ).compile()


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
    question = "Pick a playing 11 for RCB from current squad based on the performance metrics."
    for chunk in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="updates"):
        pretty_print_messages(chunk)

if __name__ == '__main__':
    main()