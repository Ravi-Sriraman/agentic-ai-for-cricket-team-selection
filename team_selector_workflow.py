import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from operator import add

from constants import LLM_MODEL, LLM_PROVIDER
from cricket_analyst_agent import create_cricket_analyst_agent
from select_batsmen_agent import create_pick_top_batsmen_agent
from select_bowlers_agent import create_pick_top_bowlers_agent
from select_wicket_keeper_agent import create_best_wicket_keeper_agent
from select_all_rounders_agent import create_pick_top_all_rounders_agent

from config import NUMBER_OF_BOWLERS, NUMBER_OF_BATSMEN, NUMBER_OF_ALL_ROUNDERS

class SearchQuery(BaseModel):
    should_select_team: str = Field(None, description="Is the user requesting to select a team? return True only if Pick a team or Select a team is mentioned in the query, otherwise return False")
    team_name: str = Field(None, description="What team name is mentioned in the query if any?")


# Pydantic models for structured output
class WicketKeeperData(BaseModel):
    """Structured data for a wicket keeper."""
    player_name: str = Field(description="Name of the wicket keeper")
    batting_average: float = Field(description="Batting average")
    batting_strike_rate: float = Field(description="Batting strike rate")
    performance_index: float = Field(description="Overall performance index")


class BatsmanData(BaseModel):
    """Structured data for a batsman."""
    player_name: str = Field(description="Name of the batsman")
    batting_average: float = Field(description="Batting average")
    batting_strike_rate: float = Field(description="Batting strike rate")
    performance_index: float = Field(description="Overall performance index")


class BowlerData(BaseModel):
    """Structured data for a bowler."""
    player_name: str = Field(description="Name of the bowler")
    bowling_average: float = Field(description="Bowling average")
    bowling_strike_rate: float = Field(description="Bowling strike rate")
    bowling_economy: float = Field(description="Bowling economy rate")
    performance_index: float = Field(description="Overall performance index")


class AllRounderData(BaseModel):
    """Structured data for an all-rounder."""
    player_name: str = Field(description="Name of the all-rounder")
    batting_average: float = Field(description="Batting average")
    batting_strike_rate: float = Field(description="Batting strike rate")
    bowling_average: float = Field(description="Bowling average")
    bowling_strike_rate: float = Field(description="Bowling strike rate")
    bowling_economy: float = Field(description="Bowling economy rate")
    performance_index: float = Field(description="Overall performance index")


class WicketKeeperList(BaseModel):
    """List of wicket keepers."""
    players: List[WicketKeeperData] = Field(description="List of wicket keeper players")

class CricketAnalystResult(BaseModel):
    """Structured data for a cricket analyst result."""
    result: str = Field(description="Final result of the query.")


class BatsmanList(BaseModel):
    """List of batsmen."""
    players: List[BatsmanData] = Field(description="List of batsman players")


class BowlerList(BaseModel):
    """List of bowlers."""
    players: List[BowlerData] = Field(description="List of bowler players")


class AllRounderList(BaseModel):
    """List of all-rounders."""
    players: List[AllRounderData] = Field(description="List of all-rounder players")


# State schema
class AgentState(TypedDict):
    messages: Annotated[Sequence, add]
    team_name: str
    question: str
    answer: str
    should_select_team: str
    wicket_keepers: List[Dict[str, Any]]
    all_rounders: List[Dict[str, Any]]
    bowlers: List[Dict[str, Any]]
    batsmen: List[Dict[str, Any]]
    final_team: Dict[str, Any]


# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def extract_team_name(query: str) -> str:
    """Extract team name from user query."""
    teams = ["CSK", "RCB", "MI", "KKR", "DC", "RR", "PBKS", "SRH", "GT", "LSG"]
    for team in teams:
        if team in query.upper():
            return team
    return "Unknown"


def save_json_to_file(data: List[Dict[str, Any]], filename: str):
    """Save structured data to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved to {filepath}")


# Create agents once at module level
_wicket_keeper_agent = None
_all_rounders_agent = None
_bowlers_agent = None
_batsmen_agent = None
_structured_llm = None


def get_agents():
    """Lazily initialize and return all agents."""
    global _wicket_keeper_agent, _all_rounders_agent, _bowlers_agent, _batsmen_agent

    if _wicket_keeper_agent is None:
        print("Initializing agents...")
        _wicket_keeper_agent = create_best_wicket_keeper_agent()
        _all_rounders_agent = create_pick_top_all_rounders_agent()
        _bowlers_agent = create_pick_top_bowlers_agent()
        _batsmen_agent = create_pick_top_batsmen_agent()

    return _wicket_keeper_agent, _all_rounders_agent, _bowlers_agent, _batsmen_agent


def get_structured_llm():
    """Get LLM configured for structured output parsing."""
    global _structured_llm
    if _structured_llm is None:
        _structured_llm = init_chat_model(model=LLM_MODEL, model_provider=LLM_PROVIDER, temperature=0)
    return _structured_llm


def extract_structured_data(content: str, schema: type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Use LLM with structured output to extract and validate data.

    Args:
        content: Raw text response from agent
        schema: Pydantic model class for structured output

    Returns:
        List of player dictionaries
    """
    llm = get_structured_llm()
    structured_llm = llm.with_structured_output(schema)

    prompt = f"""Extract the player performance data from the following text and return it in the required structured format.
If the data is already in JSON format, parse and validate it. If it's in plain text, extract the relevant information.

Text:
{content}
"""

    try:
        result = structured_llm.invoke(prompt)
        # Convert Pydantic models to dicts
        players = [player.model_dump() for player in result.players]
        return players
    except Exception as e:
        print(f" Failed to extract structured data: {e}")
        return []


# Node functions
def wicket_keeper_node(state: AgentState):
    """Select wicket keeper and extract structured data."""
    team_name = state.get("team_name", "Unknown")
    agent, _, _, _ = get_agents()
    query = f"Select best wicket keeper batsman in {team_name} team and return in JSON format with fields player_name, batting_strike_rate, batting_average, performance_index"

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = result["messages"][-1].content

    # Extract structured data using Pydantic
    wicket_keepers = extract_structured_data(content, WicketKeeperList)

    # Save to file
    filename = f"{team_name}_wicket_keepers.json"
    save_json_to_file(wicket_keepers, filename)

    return {
        "messages": [AIMessage(content=f" Wicket keepers analyzed ({len(wicket_keepers)} found)")],
        "wicket_keepers": wicket_keepers
    }


def all_rounders_node(state: AgentState):
    """Select all-rounders and extract structured data."""
    team_name = state.get("team_name", "Unknown")
    _, agent, _, _ = get_agents()
    query = f"Find all All-Rounders performance of {team_name} team in JSON format with fields player_name, batting_strike_rate, batting_average, bowling_average, bowling_economy, bowling_strike_rate, performance_index"

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = result["messages"][-1].content

    # Extract structured data using Pydantic
    all_rounders = extract_structured_data(content, AllRounderList)

    # Save to file
    filename = f"{team_name}_all_rounders.json"
    save_json_to_file(all_rounders, filename)

    return {
        "messages": [AIMessage(content=f" All-rounders analyzed ({len(all_rounders)} found)")],
        "all_rounders": all_rounders
    }


def bowlers_node(state: AgentState):
    """Select bowlers and extract structured data."""
    team_name = state.get("team_name", "Unknown")
    _, _, agent, _ = get_agents()
    query = f"Find top 10 bowlers in {team_name} and return their performance metrics in JSON format with fields player_name, bowling_average, bowling_economy, bowling_strike_rate, performance_index"

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = result["messages"][-1].content

    # Extract structured data using Pydantic
    bowlers = extract_structured_data(content, BowlerList)

    # Save to file
    filename = f"{team_name}_bowlers.json"
    save_json_to_file(bowlers, filename)

    return {
        "messages": [AIMessage(content=f"✓ Bowlers analyzed ({len(bowlers)} found)")],
        "bowlers": bowlers
    }


def batsmen_node(state: AgentState):
    """Select batsmen and extract structured data."""
    team_name = state.get("team_name", "Unknown")
    _, _, _, agent = get_agents()
    query = f"Find all batsmen in {team_name} and their performance in JSON format with fields player_name, batting_strike_rate, batting_average, performance_index"

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = result["messages"][-1].content

    # Extract structured data using Pydantic
    batsmen = extract_structured_data(content, BatsmanList)

    # Save to file
    filename = f"{team_name}_batsmen.json"
    save_json_to_file(batsmen, filename)

    return {
        "messages": [AIMessage(content=f"Batsmen analyzed ({len(batsmen)} found)")],
        "batsmen": batsmen
    }


def compose_team_node(state: AgentState):
    """
    Compose final team using deterministic selection based on performance_index.
    Team composition: 1 WK + 4 Batsmen + 2 All-rounders + 4 Bowlers = 11 players
    """
    team_name = state.get("team_name", "Unknown")

    # Get data from state
    wicket_keepers = state.get("wicket_keepers", [])
    all_rounders = state.get("all_rounders", [])
    bowlers = state.get("bowlers", [])
    batsmen = state.get("batsmen", [])

    # Sort by performance_index
    wicket_keepers_sorted = sorted(
        wicket_keepers,
        key=lambda x: float(x.get("performance_index", 0)),
        reverse=True
    )
    all_rounders_sorted = sorted(
        all_rounders,
        key=lambda x: float(x.get("performance_index", 0)),
        reverse=True
    )
    bowlers_sorted = sorted(
        bowlers,
        key=lambda x: float(x.get("performance_index", 0)),
        reverse=False  # Lower is better for bowlers
    )
    batsmen_sorted = sorted(
        batsmen,
        key=lambda x: float(x.get("performance_index", 0)),
        reverse=True
    )

    batsmen_sorted = list(filter(lambda batsman: batsman["player_name"] != wicket_keepers_sorted[0]["player_name"], batsmen_sorted))
    # Select players deterministically
    final_team = {
        "team_name": team_name,
        "wicket_keeper": wicket_keepers_sorted[0] if wicket_keepers_sorted else None,
        "batsmen": batsmen_sorted[:NUMBER_OF_BATSMEN],
        "all_rounders": all_rounders_sorted[:NUMBER_OF_ALL_ROUNDERS],
        "bowlers": bowlers_sorted[:NUMBER_OF_BOWLERS],
        "total_players": 11
    }

    # Save final team to file
    filename = f"{team_name}_final_team.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(final_team, f, indent=2)
    print(f"  → Final team saved to {filepath}")

    # Count selected players
    selected_count = (
        (1 if final_team["wicket_keeper"] else 0) +
        len(final_team["batsmen"]) +
        len(final_team["all_rounders"]) +
        len(final_team["bowlers"])
    )

    return {
        "messages": [AIMessage(content=f" Final team composed ({selected_count} players)")],
        "final_team": final_team
    }

def router_node(state: AgentState):
    """routes the requests to the right node"""
    llm = ChatOpenAI(model="gpt-4o")
    # Augment the LLM with schema for structured output
    structured_llm = llm.with_structured_output(SearchQuery)

    # Invoke the augmented LLM
    result = structured_llm.invoke(state["question"])
    return {
        "messages": [AIMessage(content=f" We should {'select a team' if result.should_select_team == 'True' else 'answer a cricket question'}")],
        "team_name": result.team_name,
        "should_select_team": result.should_select_team
    }


def cricket_analyst_node(state: AgentState):
    """Answers cricket statistics questions."""

    question = state.get("question", "Unknown")
    agent = create_cricket_analyst_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": f"{question}, return the response in JSON format with one field result"}]})
    content = result["messages"][-1].content
    llm = get_structured_llm()
    output = llm.with_structured_output(CricketAnalystResult).invoke(f"extract only the result field from json output. {content}")
    return {
        "messages": [AIMessage(content=f"Final answer is {output.result}")],
        "answer": output.result
    }


def should_should_team(state: AgentState):
    """Check whether we should select a team"""

    return state["should_select_team"]


def create_supervisor_agent():
    """
    Create supervisor agent with structured output workflow.

    Features:
    - Uses Pydantic models with with_structured_output() for data extraction
    - No manual JSON parsing needed
    - Type-safe and validated data
    - Saves results to files
    - Deterministic selection by performance_index
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("wicket_keeper", wicket_keeper_node)
    workflow.add_node("all_rounders", all_rounders_node)
    workflow.add_node("bowlers", bowlers_node)
    workflow.add_node("batsmen", batsmen_node)
    workflow.add_node("compose_team", compose_team_node)
    workflow.add_node("cricket_analyst", cricket_analyst_node)

    # Add edges - sequential workflow
    workflow.add_edge(START, "router")
    workflow.add_edge("wicket_keeper", "batsmen")
    workflow.add_edge("batsmen", "bowlers")
    workflow.add_edge("bowlers", "all_rounders")
    workflow.add_edge("all_rounders", "compose_team")
    workflow.add_edge("compose_team", END)
    workflow.add_edge("cricket_analyst", END)

    workflow.add_conditional_edges(
        "router", should_should_team, {"True": "wicket_keeper", "False": "cricket_analyst"}
    )

    return workflow.compile()


def print_final_team(final_team: Dict[str, Any]):
    """Pretty print the final team."""
    print("\n" + "="*80)
    print(f"FINAL PLAYING XI FOR {final_team['team_name']}")
    print("="*80)

    # Wicket Keeper
    print("\n🧤 WICKET KEEPER:")
    if final_team["wicket_keeper"]:
        wk = final_team["wicket_keeper"]
        print(f"  1. {wk.get('player_name', 'N/A')}")
        print(f"     Average: {wk.get('batting_average', 'N/A')}, "
              f"Strike Rate: {wk.get('batting_strike_rate', 'N/A')}, "
              f"Performance: {wk.get('performance_index', 'N/A')}")

    # Batsmen
    print("\n🏏 BATSMEN:")
    for i, bat in enumerate(final_team["batsmen"], 2):
        print(f"  {i}. {bat.get('player_name', 'N/A')}")
        print(f"     Average: {bat.get('batting_average', 'N/A')}, "
              f"Strike Rate: {bat.get('batting_strike_rate', 'N/A')}, "
              f"Performance: {bat.get('performance_index', 'N/A')}")

    # All-Rounders
    print("\n🏏🎾 ALL-ROUNDERS:")
    for i, ar in enumerate(final_team["all_rounders"], 6):
        print(f"  {i}. {ar.get('player_name', 'N/A')}")
        print(f"     Batting - Avg: {ar.get('batting_average', 'N/A')}, SR: {ar.get('batting_strike_rate', 'N/A')}")
        print(f"     Bowling - Avg: {ar.get('bowling_average', 'N/A')}, Econ: {ar.get('bowling_economy', 'N/A')}")
        print(f"     Performance: {ar.get('performance_index', 'N/A')}")

    # Bowlers
    print("\n🎾 BOWLERS:")
    for i, bowl in enumerate(final_team["bowlers"], 8):
        print(f"  {i}. {bowl.get('player_name', 'N/A')}")
        print(f"     Average: {bowl.get('bowling_average', 'N/A')}, "
              f"Economy: {bowl.get('bowling_economy', 'N/A')}, "
              f"Strike Rate: {bowl.get('bowling_strike_rate', 'N/A')}, "
              f"Performance: {bowl.get('performance_index', 'N/A')}")

    print("\n" + "="*80)


def main():
    load_dotenv()
    agent = create_supervisor_agent()
    question = "Pick a playing 11 for PBKS from current squad based on the performance metrics."
    # question = "What is V Kohli's batting average against PBKS"

    # team_name = extract_team_name(question)

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "answer": "",
        "question": question,
        "team_name": "",
        "wicket_keepers": [],
        "all_rounders": [],
        "bowlers": [],
        "batsmen": [],
        "final_team": {},
        "should_select_team": False,
    }

    print(f"\n{'='*80}")
    print(f"CRICKET CHAT WORKFLOW")
    print(f"{'='*80}\n")

    # Run workflow
    final_state = None
    for step in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in step.items():
            print(f"[{node_name}]")
            if "messages" in node_update and node_update["messages"]:
                for msg in node_update["messages"]:
                    print(f"  {msg.content}")

            # Capture final state
            if "final_team" in node_update and node_update.get("final_team"):
                final_state = node_update
            elif "answer" in node_update and node_update["answer"]:
                print(f"  {node_update['answer']}")

    # Display final team
    if final_state and "final_team" in final_state:
        print_final_team(final_state["final_team"])


if __name__ == '__main__':
    main()