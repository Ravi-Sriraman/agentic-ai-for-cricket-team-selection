from pydantic import BaseModel, Field

from dataclasses import dataclass, field

from langchain_core.tools import tool

from typing import List, Dict, Any
from pydantic import BaseModel

class SortInput(BaseModel):
    """Input schema for ranking a list of players by a particular numeric field."""
    players: List[Dict[str, Any]] = Field(
        description="List of player performance objects to rank."
    )
    by: str = Field(
        description="Field name to rank by, e.g. 'bowling_performance_index' or 'batting_performance_index'."
    )
    descending: bool = Field(
        default=True,
        description="If true, highest value first. If false, lowest value first."
    )


@dataclass
class PlayerStats:
    player_name: str
    batting_average: float = field(default=0.0)
    batting_strike_rate: float = field(default=0.0)
    bowling_average: float = field(default=0.0)
    bowling_strike_rate: float = field(default=0.0)
    bowling_economy: float = field(default=0.0)
    batting_performance_index: float = field(default=0.0)
    bowling_performance_index: float = field(default=0.0)
    all_round_performance_index: float = field(default=0.0)

@tool
def compute_batting_performance(batting_strike_rate, batting_average, player_name) -> float:
    """Computes batting_performance_index given batting_strike_rate, batting_average, player_name"""
    return 0.6 * batting_strike_rate + 0.4 * batting_average

@tool
def compute_bowling_performance(bowling_average, bowling_economy, player_name) -> float:
    """given bowling_economy, player_name, and bowling average computes bowling_performance_index"""
    return 0.14 * bowling_average + 0.86 * bowling_economy

@tool
def compute_all_round_performance(batting_strike_rate, batting_average, player_name, bowling_average, bowling_economy) -> float:
    """Given batting_strike_rate, batting_average, player_name, bowling_economy, player_name, and bowling average, computes all_round_performance_index """
    return 0.5 * (0.6 * batting_strike_rate + 0.4 * batting_average) + 0.5 * (0.14 * bowling_average + 0.86 * bowling_economy)

@tool("rank_players", args_schema=SortInput)
def rank_list_of_objects_based_on_by_field(players :List[Dict[str, Any]], by: str) -> List[Dict[str, Any]]:
    """Ranks a list of players based on a field and returns the list of players ranked based on that field"""
    return sorted(players, key=lambda obj: obj[by], reverse=True)

@tool("rank_players", args_schema=SortInput)
def rank_list_of_players_based_on_by_field(
        players: List[Dict[str, Any]],
        by: str,
        descending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ranks a list of player dicts based on the specified numeric field.

    - 'players' is a list of JSON-like dicts with numeric fields such as
      'bowling_performance_index', 'batting_performance_index', etc.
    - 'by' is the field name to rank by.
    - 'descending' controls sort order: True = highest to lowest.
    """

    def key_fn(player: Dict[str, Any]):
        # Use .get to avoid KeyError; missing values get pushed to the end.
        value = player.get(by, None)
        if value is None:
            # Missing values are treated as worst
            return float("-inf") if descending else float("inf")
        return value

    ranked = sorted(players, key=key_fn, reverse=descending)
    return ranked


def main():
    batters_stats = [
        PlayerStats(
            player_name="V Kohli",
            batting_average=38,
            batting_strike_rate=136,
        ),
        PlayerStats(
            player_name="RG Sharma",
            batting_average=39,
            batting_strike_rate=140,
        ),
        PlayerStats(
            player_name="A Rahane",
            batting_average=32,
            batting_strike_rate=121,
        )
    ]

    # print(rank_batsmen(batters_stats))

    bowler_stats = [
        PlayerStats(
            player_name="B Kumar",
            bowling_average=30,
            bowling_strike_rate=25,
            bowling_economy=6.7,
        ),
        PlayerStats(
            player_name="U Yadav",
            bowling_average=32,
            bowling_strike_rate=28,
            bowling_economy=7.7,
        ),
        PlayerStats(
            player_name="YZ Chahal",
            bowling_average=37,
            bowling_strike_rate=42,
            bowling_economy=5.4,
        )
    ]

    # print(rank_bowlers(bowler_stats))

    all_rounder_stats = [
        PlayerStats(
            player_name="Livingstone",
            bowling_average=30,
            bowling_strike_rate=25,
            bowling_economy=6.7,
            batting_average=38,
            batting_strike_rate=136,
        ),
        PlayerStats(
            player_name="Shephard",
            bowling_average=32,
            bowling_strike_rate=28,
            bowling_economy=7.7,
            batting_average=39,
            batting_strike_rate=140,
        ),
        PlayerStats(
            player_name="Krunal Pandya",
            bowling_average=37,
            bowling_strike_rate=42,
            bowling_economy=5.4,
            batting_average=32,
            batting_strike_rate=121,
        )
    ]

    # print(rank_all_rounders(all_rounder_stats))




if __name__ == '__main__':
    main()