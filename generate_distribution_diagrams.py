"""
Generate statistical distribution diagrams for the research methodology section:
1. Player Role Distribution by Team - Shows composition diversity across IPL teams
2. Match Distribution Over Seasons - Temporal coverage of the dataset
3. Performance Metrics Distribution - Statistical spread of key cricket metrics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sqlite3
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Connect to database
db_path = 'db/cricket.db'

def create_player_role_distribution():
    """Create stacked bar chart showing player role distribution by team"""
    conn = sqlite3.connect(db_path)

    # Get role distribution by team
    query = """
    SELECT team_name, player_role, COUNT(*) as count
    FROM ipl_squads
    WHERE team_name IN ('RCB', 'MI', 'CSK', 'KKR', 'DC', 'PBKS', 'RR', 'SRH')
    GROUP BY team_name, player_role
    ORDER BY team_name, player_role
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Pivot data for stacked bar chart
    pivot_df = df.pivot(index='team_name', columns='player_role', values='count').fillna(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each role
    colors = {
        'Batter': '#F4B183',
        'WK-Batter': '#C9A0DC',
        'All-Rounder': '#A9D18E',
        'Bowler': '#5B9BD5',
        'Opener-Batter': '#FFD966'
    }

    # Get all roles that exist in the data
    roles = [col for col in pivot_df.columns if col in colors]

    # Create stacked bars
    bottom = np.zeros(len(pivot_df))
    bars = []

    for role in roles:
        if role in pivot_df.columns:
            bar = ax.bar(pivot_df.index, pivot_df[role], bottom=bottom,
                        label=role, color=colors[role], edgecolor='white', linewidth=1.5)
            bars.append(bar)
            bottom += pivot_df[role]

    # Customize chart
    ax.set_xlabel('IPL Team', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
    ax.set_title('Player Role Distribution Across IPL Teams',
                fontsize=14, fontweight='bold', pad=20)

    # Add value labels on bars
    for i, team in enumerate(pivot_df.index):
        total = bottom[i]
        ax.text(i, total + 0.5, f'{int(total)}', ha='center', va='bottom',
               fontweight='bold', fontsize=10)

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('./images/player_role_distribution.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated: player_role_distribution.png")
    plt.close()


def create_match_season_distribution():
    """Create combined chart showing matches per season and cumulative coverage"""
    conn = sqlite3.connect(db_path)

    # Get match count by season
    query = """
    SELECT season, COUNT(*) as match_count
    FROM matches
    WHERE season IS NOT NULL AND season != ''
    GROUP BY season
    ORDER BY season
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert season to int for proper ordering (handle "2007/08" format)
    df['season'] = df['season'].apply(lambda x: int(str(x).split('/')[0]) if '/' in str(x) else int(x))
    df = df.sort_values('season')

    # Calculate cumulative matches
    df['cumulative'] = df['match_count'].cumsum()

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart for matches per season
    bars = ax1.bar(df['season'], df['match_count'], color='#5B9BD5',
                   alpha=0.7, edgecolor='#2E5C8A', linewidth=1.5, label='Matches per Season')

    # Add value labels on bars
    for i, (season, count) in enumerate(zip(df['season'], df['match_count'])):
        ax1.text(season, count + 2, str(int(count)), ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax1.set_xlabel('IPL Season', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Matches per Season', fontsize=12, fontweight='bold', color='#2E5C8A')
    ax1.tick_params(axis='y', labelcolor='#2E5C8A')
    ax1.set_ylim(0, max(df['match_count']) * 1.15)

    # Line chart for cumulative matches
    ax2 = ax1.twinx()
    line = ax2.plot(df['season'], df['cumulative'], color='#C55A11',
                    linewidth=3, marker='o', markersize=6, label='Cumulative Matches')

    ax2.set_ylabel('Cumulative Matches', fontsize=12, fontweight='bold', color='#C55A11')
    ax2.tick_params(axis='y', labelcolor='#C55A11')
    ax2.set_ylim(0, max(df['cumulative']) * 1.1)

    # Add final cumulative count annotation
    final_season = df['season'].iloc[-1]
    final_cum = df['cumulative'].iloc[-1]
    ax2.annotate(f'Total: {int(final_cum)} matches',
                xy=(final_season, final_cum),
                xytext=(final_season - 3, final_cum - 100),
                fontsize=11, fontweight='bold', color='#C55A11',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCE4D6', edgecolor='#C55A11', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='#C55A11', lw=2))

    # Title
    ax1.set_title('IPL Match Distribution Across Seasons (2008-2024)',
                 fontsize=14, fontweight='bold', pad=20)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Grid
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    # X-axis formatting
    ax1.set_xticks(df['season'])
    ax1.set_xticklabels(df['season'], rotation=45, ha='right')

    # Styling
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('images/match_season_distribution.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated: match_season_distribution.png")
    plt.close()


def create_performance_metrics_distribution():
    """Create multi-panel histogram showing distribution of key performance metrics"""
    conn = sqlite3.connect(db_path)

    # Get player statistics
    query = """
    SELECT
        batting_avg,
        economy,
        runs,
        wickets,
        innings
    FROM player_statistics
    WHERE innings >= 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate derived metrics
    df['runs_per_match'] = df['runs'] / df['innings']
    df['wickets_per_match'] = df['wickets'] / df['innings']

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Key Performance Metrics (Players with ≥10 Innings)',
                fontsize=14, fontweight='bold', y=0.995)

    # Define metrics to plot
    metrics = [
        ('runs', 'Total Runs Scored', '#F4B183', 100),
        ('runs_per_match', 'Runs per Innings', '#5B9BD5', 5),
        ('batting_avg', 'Batting Average', '#A9D18E', 5),
        ('wickets', 'Total Wickets Taken', '#C9A0DC', 10),
        ('economy', 'Bowling Economy Rate', '#FFD966', 0.5),
        ('wickets_per_match', 'Wickets per Innings', '#C55A11', 0.5)
    ]

    for idx, (metric, title, color, bin_width) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        # Filter out zeros and NaNs
        data = df[metric].dropna()
        data = data[data > 0]

        if len(data) > 0:
            # Calculate bins
            if metric in ['runs', 'wickets']:
                bins = np.arange(0, data.max() + bin_width, bin_width)
            else:
                bins = 30

            # Create histogram
            n, bins_edges, patches = ax.hist(data, bins=bins, color=color,
                                             alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()

            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}', alpha=0.8)
            ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2,
                      label=f'Median: {median_val:.1f}', alpha=0.8)

            # Labels and title
            ax.set_xlabel(title, fontsize=10, fontweight='bold')
            ax.set_ylabel('Number of Players', fontsize=10, fontweight='bold')

            # Legend
            ax.legend(fontsize=8, loc='upper right')

            # Grid
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='--', alpha=0.5)

            # Add sample size annotation
            ax.text(0.02, 0.98, f'n = {len(data)}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('Ravi Sriraman Research/images/performance_metrics_distribution.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated: performance_metrics_distribution.png")
    plt.close()


if __name__ == '__main__':
    import os

    # Create images directory if it doesn't exist
    os.makedirs('Ravi Sriraman Research/images', exist_ok=True)

    print("Generating distribution diagrams...")
    print("-" * 50)

    try:
        create_player_role_distribution()
        create_match_season_distribution()
        # create_performance_metrics_distribution()

        print("-" * 50)
        print("✓ All distribution diagrams generated successfully!")
        print("\nGenerated files:")
        print("  1. player_role_distribution.png - Player roles across IPL teams")
        print("  2. match_season_distribution.png - Temporal coverage of dataset")
        print("  3. performance_metrics_distribution.png - Statistical distributions")

    except Exception as e:
        print(f"\n✗ Error generating diagrams: {e}")
        import traceback
        traceback.print_exc()