"""
Evaluation script for cricket team selection AI agent workflow.

Measures three essential metrics:
1. Task Success Rate - Does it complete team selection correctly?
2. Efficiency - How many steps/tokens does it use?
3. Correctness - Are the selected players accurate and valid?
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from team_selector_workflow import create_supervisor_agent, extract_team_name


class TeamSelectorEvaluator:
    """Evaluates the team selector workflow performance."""

    def __init__(self):
        self.metrics = {
            "task_success": {},
            "efficiency": {},
            "correctness": {},
            "overall_score": 0.0
        }
        self.results_dir = Path("results")

    def evaluate(self, query: str) -> Dict[str, Any]:
        """
        Run complete evaluation on the team selector workflow.

        Args:
            query: User query for team selection

        Returns:
            Dictionary containing all evaluation metrics
        """
        print("="*80)
        print("TEAM SELECTOR WORKFLOW EVALUATION")
        print("="*80)
        print(f"Query: {query}\n")

        team_name = extract_team_name(query)
        agent = create_supervisor_agent()

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "team_name": team_name,
            "question": query,
            "wicket_keepers": [],
            "all_rounders": [],
            "bowlers": [],
            "batsmen": [],
            "final_team": {}
        }

        # Track efficiency metrics
        start_time = time.time()
        step_count = 0
        node_executions = []

        # Run workflow
        print("Running workflow...\n")
        final_state = None
        try:
            for step in agent.stream(initial_state, stream_mode="updates"):
                step_count += 1
                for node_name, node_update in step.items():
                    node_executions.append(node_name)
                    print(f"  [{step_count}] {node_name}")

                    if "final_team" in node_update and node_update.get("final_team"):
                        final_state = node_update

            execution_time = time.time() - start_time

            # Evaluate metrics
            print("\n" + "="*80)
            print("EVALUATION RESULTS")
            print("="*80 + "\n")

            self._evaluate_task_success(final_state, team_name)
            self._evaluate_efficiency(step_count, node_executions, execution_time)
            self._evaluate_correctness(final_state, team_name)
            self._calculate_overall_score()

            # Generate report
            self._print_report()
            self._save_report(query, team_name)

            return self.metrics

        except Exception as e:
            print(f"\n ERROR: Workflow failed with exception: {e}")
            self.metrics["task_success"]["completed"] = False
            self.metrics["task_success"]["error"] = str(e)
            return self.metrics

    def _evaluate_task_success(self, final_state: Dict[str, Any], team_name: str):
        """Evaluate Task Success Rate metric."""
        print("1. TASK SUCCESS RATE")
        print("-" * 80)

        success_checks = {
            "workflow_completed": final_state is not None,
            "has_final_team": False,
            "correct_team_size": False,
            "has_wicket_keeper": False,
            "has_batsmen": False,
            "has_all_rounders": False,
            "has_bowlers": False,
            "correct_composition": False
        }

        if final_state and "final_team" in final_state:
            final_team = final_state["final_team"]
            success_checks["has_final_team"] = True

            # Check team composition
            wk_count = 1 if final_team.get("wicket_keeper") else 0
            batsmen_count = len(final_team.get("batsmen", []))
            ar_count = len(final_team.get("all_rounders", []))
            bowlers_count = len(final_team.get("bowlers", []))

            total_players = wk_count + batsmen_count + ar_count + bowlers_count

            success_checks["has_wicket_keeper"] = wk_count == 1
            success_checks["has_batsmen"] = batsmen_count > 0
            success_checks["has_all_rounders"] = ar_count > 0
            success_checks["has_bowlers"] = bowlers_count > 0
            success_checks["correct_team_size"] = total_players == 11
            success_checks["correct_composition"] = (
                wk_count == 1 and
                batsmen_count == 4 and
                ar_count == 2 and
                bowlers_count == 4
            )

        # Calculate success rate
        passed_checks = sum(1 for v in success_checks.values() if v)
        total_checks = len(success_checks)
        success_rate = (passed_checks / total_checks) * 100

        self.metrics["task_success"] = {
            "success_rate": success_rate,
            "checks": success_checks,
            "passed": passed_checks,
            "total": total_checks
        }

        # Print results
        for check, passed in success_checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")

        print(f"\n  Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks} checks passed)")
        print()

    def _evaluate_efficiency(self, step_count: int, node_executions: List[str], execution_time: float):
        """Evaluate Efficiency and Step Count metric."""
        print("2. EFFICIENCY")
        print("-" * 80)

        # Count node executions
        node_counts = {}
        for node in node_executions:
            node_counts[node] = node_counts.get(node, 0) + 1

        # Expected workflow: wicket_keeper -> batsmen -> bowlers -> all_rounders -> compose_team
        expected_steps = 6
        efficiency_score = min(100, (expected_steps / step_count) * 100) if step_count > 0 else 0

        # Check for repeated nodes (indicates loops/retries)
        has_loops = any(count > 1 for count in node_counts.values())

        self.metrics["efficiency"] = {
            "total_steps": step_count,
            "expected_steps": expected_steps,
            "efficiency_score": efficiency_score,
            "execution_time_seconds": round(execution_time, 2),
            "node_executions": node_counts,
            "has_loops": has_loops
        }

        print(f"  Total Steps: {step_count}")
        print(f"  Expected Steps: {expected_steps}")
        print(f"  Efficiency Score: {efficiency_score:.1f}%")
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Loops Detected: {'Yes ⚠' if has_loops else 'No ✓'}")
        print(f"\n  Node Executions:")
        for node, count in node_counts.items():
            status = "⚠" if count > 1 else "✓"
            print(f"    {status} {node}: {count}x")
        print()

    def _evaluate_correctness(self, final_state: Dict[str, Any], team_name: str):
        """Evaluate Correctness and Relevance metric."""
        print("3. CORRECTNESS")
        print("-" * 80)

        correctness_checks = {
            "players_have_names": False,
            "players_have_metrics": False,
            "wicket_keeper_valid": False,
            "batsmen_valid": False,
            "bowlers_valid": False,
            "all_rounders_valid": False,
            "performance_indices_present": False,
            "no_duplicate_players": False
        }

        if final_state and "final_team" in final_state:
            final_team = final_state["final_team"]
            all_players = []

            # Check wicket keeper
            wk = final_team.get("wicket_keeper")
            if wk:
                correctness_checks["wicket_keeper_valid"] = (
                    "player_name" in wk and
                    "batting_average" in wk and
                    "batting_strike_rate" in wk and
                    "performance_index" in wk
                )
                all_players.append(wk.get("player_name"))

            # Check batsmen
            batsmen = final_team.get("batsmen", [])
            if batsmen:
                correctness_checks["batsmen_valid"] = all(
                    "player_name" in b and
                    "batting_average" in b and
                    "batting_strike_rate" in b and
                    "performance_index" in b
                    for b in batsmen
                )
                all_players.extend([b.get("player_name") for b in batsmen])

            # Check bowlers
            bowlers = final_team.get("bowlers", [])
            if bowlers:
                correctness_checks["bowlers_valid"] = all(
                    "player_name" in b and
                    "bowling_average" in b and
                    "bowling_economy" in b and
                    "bowling_strike_rate" in b and
                    "performance_index" in b
                    for b in bowlers
                )
                all_players.extend([b.get("player_name") for b in bowlers])

            # Check all-rounders
            all_rounders = final_team.get("all_rounders", [])
            if all_rounders:
                correctness_checks["all_rounders_valid"] = all(
                    "player_name" in ar and
                    "batting_average" in ar and
                    "batting_strike_rate" in ar and
                    "bowling_average" in ar and
                    "bowling_economy" in ar and
                    "bowling_strike_rate" in ar and
                    "performance_index" in ar
                    for ar in all_rounders
                )
                all_players.extend([ar.get("player_name") for ar in all_rounders])

            # Check for duplicates
            correctness_checks["no_duplicate_players"] = len(all_players) == len(set(all_players))

            # Check if all players have names and metrics
            correctness_checks["players_have_names"] = all(name for name in all_players)
            correctness_checks["players_have_metrics"] = (
                correctness_checks["wicket_keeper_valid"] and
                correctness_checks["batsmen_valid"] and
                correctness_checks["bowlers_valid"] and
                correctness_checks["all_rounders_valid"]
            )
            correctness_checks["performance_indices_present"] = True

        # Calculate correctness score
        passed_checks = sum(1 for v in correctness_checks.values() if v)
        total_checks = len(correctness_checks)
        correctness_score = (passed_checks / total_checks) * 100

        self.metrics["correctness"] = {
            "correctness_score": correctness_score,
            "checks": correctness_checks,
            "passed": passed_checks,
            "total": total_checks
        }

        # Print results
        for check, passed in correctness_checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")

        print(f"\n  Correctness Score: {correctness_score:.1f}% ({passed_checks}/{total_checks} checks passed)")
        print()

    def _calculate_overall_score(self):
        """Calculate weighted overall score."""
        # Weights: Task Success (40%), Efficiency (30%), Correctness (30%)
        task_score = self.metrics["task_success"]["success_rate"]
        efficiency_score = self.metrics["efficiency"]["efficiency_score"]
        correctness_score = self.metrics["correctness"]["correctness_score"]

        overall_score = (
            task_score * 0.4 +
            efficiency_score * 0.3 +
            correctness_score * 0.3
        )

        self.metrics["overall_score"] = round(overall_score, 2)

    def _print_report(self):
        """Print evaluation summary report."""
        print("="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\n  Task Success Rate:  {self.metrics['task_success']['success_rate']:.1f}%")
        print(f"  Efficiency Score:   {self.metrics['efficiency']['efficiency_score']:.1f}%")
        print(f"  Correctness Score:  {self.metrics['correctness']['correctness_score']:.1f}%")
        print(f"\n  Overall Score:      {self.metrics['overall_score']:.1f}%")

        # Grade
        score = self.metrics['overall_score']
        if score >= 90:
            grade = "A (Excellent)"
        elif score >= 80:
            grade = "B (Good)"
        elif score >= 70:
            grade = "C (Fair)"
        elif score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Failing)"

        print(f"  Grade:              {grade}")
        print("\n" + "="*80 + "\n")

    def _save_report(self, query: str, team_name: str):
        """Save evaluation report to JSON file."""
        report = {
            "query": query,
            "team_name": team_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self.metrics
        }

        report_file = self.results_dir / f"{team_name}_evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to: {report_file}\n")


def run_evaluation(query: str = None):
    """Run evaluation on team selector workflow."""
    load_dotenv()

    if query is None:
        query = "Pick a playing 11 for CSK from current squad based on the performance metrics."

    evaluator = TeamSelectorEvaluator()
    metrics = evaluator.evaluate(query)

    return metrics


def run_batch_evaluation():
    """Run evaluation on multiple teams."""
    load_dotenv()

    teams = ["CSK", "RCB", "MI", "KKR"]
    results = {}

    print("\n" + "="*80)
    print("BATCH EVALUATION - MULTIPLE TEAMS")
    print("="*80 + "\n")

    for team in teams:
        query = f"Pick a playing 11 for {team} from current squad based on the performance metrics."
        print(f"\nEvaluating {team}...")
        print("-"*80)

        evaluator = TeamSelectorEvaluator()
        metrics = evaluator.evaluate(query)
        results[team] = metrics["overall_score"]

    # Summary
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    for team, score in results.items():
        print(f"  {team}: {score:.1f}%")

    avg_score = sum(results.values()) / len(results)
    print(f"\n  Average Score: {avg_score:.1f}%")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Run single evaluation
    run_evaluation()

    # Uncomment to run batch evaluation on multiple teams
    # run_batch_evaluation()