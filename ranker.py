from pydantic import BaseModel, Field

from dataclasses import dataclass, field

from langchain_core.tools import tool

from typing import List, Dict, Any
from pydantic import BaseModel
from config import BATTING_WEIGHTS, BOWLING_WEIGHTS

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
def compute_batting_performance(batting_strike_rate, batting_average, player_name, alpha: float = 0.5) -> float:
    """
    Computes Batting Performance Factor (BPF) using the formula:
    BPF = (Batting Strike Rate)^α × (Batting Average)^(1-α)

    Reference: "A Balanced Squad for Indian Premier League Using Modified NSGA-II"

    Args:
        batting_strike_rate: Player's batting strike rate
        batting_average: Player's batting average
        player_name: Name of the player
        alpha: Weight factor between 0 and 1 (default: 0.5 for equal weighting)

    Returns:
        BPF value (higher is better)
    """
    if batting_strike_rate and batting_average and batting_strike_rate > 0 and batting_average > 0:
        # BPF = (SR)^α × (Avg)^(1-α)
        bpf = (batting_strike_rate ** alpha) * (batting_average ** (1 - alpha))
        return bpf
    return 0

@tool
def compute_bowling_performance(bowling_average, bowling_economy, bowling_strike_rate, player_name) -> float:
    """
    Computes Combined Bowling Rate (CBR) using the formula:
    CBR = 1 / (1/Bowling_Average + 1/Economy_Rate + 1/Strike_Rate)

    Reference: "A Balanced Squad for Indian Premier League Using Modified NSGA-II"

    Args:
        bowling_average: Player's bowling average (runs per wicket)
        bowling_economy: Player's economy rate (runs per over)
        bowling_strike_rate: Player's bowling strike rate (balls per wicket)
        player_name: Name of the player

    Returns:
        CBR value (lower is better for bowlers)
    """
    if bowling_economy and bowling_average and bowling_strike_rate:
        if bowling_economy > 0 and bowling_average > 0 and bowling_strike_rate > 0:
            try:
                # CBR = 1 / (1/A + 1/E + 1/S)
                cbr = 1.0 / ((1.0/bowling_average) + (1.0/bowling_economy) + (1.0/bowling_strike_rate))
                return cbr
            except ZeroDivisionError:
                return 0
    return 0

@tool
def compute_all_round_performance(batting_strike_rate, batting_average, player_name,
                                  bowling_average, bowling_economy, bowling_strike_rate,
                                  alpha: float = 0.5) -> float:
    """
    Computes all-rounder performance index using BPF and CBR.

    For all-rounders, combines both batting and bowling performances:
    - BPF = (Batting Strike Rate)^α × (Batting Average)^(1-α)
    - CBR = 1 / (1/Bowling_Average + 1/Economy_Rate + 1/Strike_Rate)

    All-rounder Index = (BPF + 1/CBR) / 2

    Note: We use 1/CBR for bowling since lower CBR is better, but we want
    higher values to indicate better performance for consistency.

    Args:
        batting_strike_rate: Player's batting strike rate
        batting_average: Player's batting average
        player_name: Name of the player
        bowling_average: Player's bowling average
        bowling_economy: Player's economy rate
        bowling_strike_rate: Player's bowling strike rate
        alpha: Weight factor for BPF (default: 0.5)

    Returns:
        All-rounder performance index
    """
    has_batting = batting_strike_rate and batting_average and batting_strike_rate > 0 and batting_average > 0
    has_bowling = bowling_average and bowling_economy and bowling_strike_rate and \
                  bowling_average > 0 and bowling_economy > 0 and bowling_strike_rate > 0

    if not has_batting and not has_bowling:
        return 0.0

    # Calculate BPF
    bpf = 0
    if has_batting:
        bpf = (batting_strike_rate ** alpha) * (batting_average ** (1 - alpha))

    # Calculate CBR
    cbr = 0
    if has_bowling:
        cbr = 1.0 / ((1.0/bowling_average) + (1.0/bowling_economy) + (1.0/bowling_strike_rate))

    # If only batting or only bowling, return respective performance
    if not has_batting:
        return 1.0 / cbr if cbr > 0 else 0  # Inverse of CBR for consistency
    elif not has_bowling:
        return bpf

    # For true all-rounders, combine both (equal weighting)
    # Using 1/CBR since lower CBR is better
    bowling_component = 1.0 / cbr if cbr > 0 else 0
    all_rounder_index = 0.5 * bpf + 0.5 * bowling_component

    return all_rounder_index

@tool("rank_players", args_schema=SortInput)
def rank_list_of_players_by_field(
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