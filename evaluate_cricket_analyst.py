"""
Evaluation script for cricket_analyst_agent.py

Tests the general-purpose cricket analyst agent on various query types:
- Player statistics (batting/bowling)
- Team comparisons
- Historical data analysis
- Rankings and sorting
- Complex multi-step queries

Measures:
1. Task Success Rate - Does it answer questions correctly?
2. Efficiency - How many SQL queries/steps does it use?
3. Correctness - Are SQL queries valid and answers accurate?
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from cricket_analyst_agent import create_cricket_analyst_agent


class CricketAnalystEvaluator:
    """Evaluates the cricket analyst agent."""

    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def create_test_cases(self) -> List[Dict[str, Any]]:
        """
        Create test cases for different query types.

        Returns:
            List of test case dictionaries with query and expected characteristics
        """
        test_cases = [
            {
                "name": "Player Batting Statistics",
                "query": "What is V Kohli's batting average and strike rate?",
                "category": "player_stats",
                "expected_keywords": ["kohli", "average", "strike rate"],
                "expected_sql_queries": 2,  # Expected minimum SQL queries
            },
            {
                "name": "Team Top Scorer",
                "query": "Who has scored the most runs for CSK during 2024 and 2025?",
                "category": "team_analysis",
                "expected_keywords": ["csk", "runs", "2024", "2025"],
                "expected_sql_queries": 1,
            },
            {
                "name": "Player vs Team Stats",
                "query": "What is MS Dhoni's batting average against RCB?",
                "category": "head_to_head",
                "expected_keywords": ["dhoni", "rcb", "average"],
                "expected_sql_queries": 2,
            },
            {
                "name": "Team Ranking",
                "query": "Rank CSK batsmen based on their strike rate",
                "category": "ranking",
                "expected_keywords": ["csk", "batsmen", "strike rate", "rank"],
                "expected_sql_queries": 2,
            },
            {
                "name": "Bowler Statistics",
                "query": "What is JJ Bumrah's bowling average and economy rate?",
                "category": "bowling_stats",
                "expected_keywords": ["bumrah", "average", "economy"],
                "expected_sql_queries": 2,
            },
            {
                "name": "Complex Multi-Step Query",
                "query": "Compare the batting averages of RG Sharma and V Kohli in IPL 2024",
                "category": "comparison",
                "expected_keywords": ["rohit", "kohli", "2024", "average"],
                "expected_sql_queries": 3,
            }
        ]

        return test_cases

    def evaluate_single_query(self, query: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            query: The question to ask the agent
            test_case: Test case metadata

        Returns:
            Evaluation metrics for this query
        """
        print(f"\n{'='*80}")
        print(f"TEST: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Category: {test_case['category']}\n")

        metrics = {
            "test_name": test_case["name"],
            "query": query,
            "category": test_case["category"],
            "task_success": {},
            "efficiency": {},
            "correctness": {},
            "overall_score": 0.0
        }

        # Create agent
        start_time = time.time()
        try:
            agent = create_cricket_analyst_agent()
            agent_creation_time = time.time() - start_time
        except Exception as e:
            print(f"❌ ERROR: Failed to create agent: {e}")
            metrics["task_success"]["error"] = str(e)
            return metrics

        # Run query
        print("Running query...\n")
        step_count = 0
        all_messages = []
        workflow_failed = False
        error_message = None
        sql_queries_detected = 0

        execution_start = time.time()
        try:
            for step in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
                step_count += 1
                if "messages" in step:
                    all_messages = step["messages"]

                    # Try to detect SQL queries
                    last_message = all_messages[-1]
                    content = last_message.content if hasattr(last_message, 'content') else str(last_message)

                    # Count SQL queries
                    sql_patterns = [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b']
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in sql_patterns):
                        sql_queries_detected += 1

                    print(f"  Step {step_count}: Processing...")

        except KeyboardInterrupt:
            print(f"\n\n⚠ INTERRUPTED: Evaluation interrupted by user")
            workflow_failed = True
            error_message = "Interrupted by user"
        except Exception as e:
            print(f"\n\n❌ ERROR: Query execution failed: {e}")
            import traceback
            traceback.print_exc()
            workflow_failed = True
            error_message = str(e)

        execution_time = time.time() - execution_start
        total_time = time.time() - start_time

        # Extract final response
        final_response = None
        if all_messages:
            final_response = all_messages[-1].content

        # Evaluate metrics
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")

        if workflow_failed:
            print(f"⚠ Note: Query failed - evaluating partial results\n")

        try:
            self._evaluate_task_success(metrics, workflow_failed, error_message,
                                       all_messages, final_response, test_case)
        except Exception as e:
            print(f"❌ Error in task success evaluation: {e}")

        try:
            self._evaluate_efficiency(metrics, step_count, execution_time,
                                     agent_creation_time, total_time,
                                     sql_queries_detected, test_case)
        except Exception as e:
            print(f"❌ Error in efficiency evaluation: {e}")

        try:
            self._evaluate_correctness(metrics, final_response, test_case, all_messages)
        except Exception as e:
            print(f"❌ Error in correctness evaluation: {e}")

        try:
            self._calculate_overall_score(metrics)
        except Exception as e:
            print(f"❌ Error calculating overall score: {e}")

        # Print summary
        try:
            self._print_query_summary(metrics)
        except Exception as e:
            print(f"❌ Error printing summary: {e}")

        return metrics

    def _evaluate_task_success(self, metrics: Dict, workflow_failed: bool,
                               error_message: str, messages: List,
                               final_response: str, test_case: Dict):
        """Evaluate Task Success Rate metric."""
        print("1. TASK SUCCESS RATE")
        print("-" * 80)

        success_checks = {
            "query_executed": not workflow_failed,
            "generated_response": len(messages) > 0,
            "response_has_content": False,
            "response_is_substantial": False,
            "mentions_relevant_keywords": False,
        }

        if workflow_failed and error_message:
            print(f"  ⚠ Query failed: {error_message}\n")

        if final_response:
            success_checks["response_has_content"] = len(final_response) > 20
            success_checks["response_is_substantial"] = len(final_response) > 100

            # Check if response mentions expected keywords
            response_lower = final_response.lower()
            keywords_found = sum(
                1 for keyword in test_case.get("expected_keywords", [])
                if keyword.lower() in response_lower
            )
            success_checks["mentions_relevant_keywords"] = keywords_found >= 1

        # Calculate success rate
        passed_checks = sum(1 for v in success_checks.values() if v)
        total_checks = len(success_checks)
        success_rate = (passed_checks / total_checks) * 100

        metrics["task_success"] = {
            "success_rate": success_rate,
            "checks": success_checks,
            "passed": passed_checks,
            "total": total_checks,
        }

        # Print results
        for check, passed in success_checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")

        print(f"\n  Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks} checks passed)")
        print()

    def _evaluate_efficiency(self, metrics: Dict, step_count: int, execution_time: float,
                            agent_creation_time: float, total_time: float,
                            sql_queries_detected: int, test_case: Dict):
        """Evaluate Efficiency metric."""
        print("2. EFFICIENCY")
        print("-" * 80)

        expected_sql = test_case.get("expected_sql_queries", 2)

        # Calculate efficiency score
        # Good if: steps <= 10, sql queries close to expected, time < 10s
        step_score = 100 if step_count <= 10 else (100 - (step_count - 10) * 5)
        step_score = max(0, step_score)

        sql_score = 100 if abs(sql_queries_detected - expected_sql) <= 2 else 70

        time_score = 100 if execution_time < 10 else (100 - (execution_time - 10) * 2)
        time_score = max(0, time_score)

        efficiency_score = (step_score * 0.4 + sql_score * 0.3 + time_score * 0.3)

        metrics["efficiency"] = {
            "total_steps": step_count,
            "expected_sql_queries": expected_sql,
            "sql_queries_detected": sql_queries_detected,
            "efficiency_score": round(efficiency_score, 2),
            "execution_time_seconds": round(execution_time, 2),
            "agent_creation_time_seconds": round(agent_creation_time, 2),
            "total_time_seconds": round(total_time, 2)
        }

        print(f"  Total Steps: {step_count}")
        print(f"  SQL Queries Detected: {sql_queries_detected} (Expected: ~{expected_sql})")
        print(f"  Efficiency Score: {efficiency_score:.1f}%")
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        print()

    def _evaluate_correctness(self, metrics: Dict, final_response: str,
                             test_case: Dict, all_messages: List):
        """Evaluate Correctness metric."""
        print("3. CORRECTNESS & ANSWER QUALITY")
        print("-" * 80)

        correctness_checks = {
            "has_answer": final_response is not None and len(final_response) > 0,
            "contains_data": False,
            "has_numerical_results": False,
            "mentions_context": False,
            "no_error_messages": True,
            "coherent_response": False,
        }

        if final_response:
            response_lower = final_response.lower()

            # Check for data/numbers
            has_numbers = bool(re.search(r'\d+\.?\d*', final_response))
            correctness_checks["has_numerical_results"] = has_numbers
            correctness_checks["contains_data"] = has_numbers or "no data" not in response_lower

            # Check for context (mentions team/player names)
            keywords = test_case.get("expected_keywords", [])
            mentions = sum(1 for k in keywords if k.lower() in response_lower)
            correctness_checks["mentions_context"] = mentions >= 1

            # Check for error messages
            error_indicators = ["error", "failed", "unable", "could not", "exception"]
            correctness_checks["no_error_messages"] = not any(
                err in response_lower for err in error_indicators
            )

            # Check coherence (reasonable length and structure)
            correctness_checks["coherent_response"] = (
                len(final_response) >= 50 and
                not final_response.startswith("Error")
            )

        # Calculate correctness score
        passed_checks = sum(1 for v in correctness_checks.values() if v)
        total_checks = len(correctness_checks)
        correctness_score = (passed_checks / total_checks) * 100

        metrics["correctness"] = {
            "correctness_score": correctness_score,
            "checks": correctness_checks,
            "passed": passed_checks,
            "total": total_checks,
            "final_answer": final_response[:200] if final_response else None  # Save truncated answer
        }

        # Print results
        for check, passed in correctness_checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")

        print(f"\n  Correctness Score: {correctness_score:.1f}% ({passed_checks}/{total_checks} checks passed)")

        if final_response:
            print(f"\n  Final Answer Preview:")
            print(f"  {final_response[:200]}{'...' if len(final_response) > 200 else ''}")
        print()

    def _calculate_overall_score(self, metrics: Dict):
        """Calculate weighted overall score."""
        # Weights: Task Success (40%), Efficiency (30%), Correctness (30%)
        task_score = metrics["task_success"]["success_rate"]
        efficiency_score = metrics["efficiency"]["efficiency_score"]
        correctness_score = metrics["correctness"]["correctness_score"]

        overall_score = (
            task_score * 0.4 +
            efficiency_score * 0.3 +
            correctness_score * 0.3
        )

        metrics["overall_score"] = round(overall_score, 2)

    def _print_query_summary(self, metrics: Dict):
        """Print summary for a single query."""
        print("QUERY SUMMARY")
        print("-" * 80)
        print(f"  Test: {metrics['test_name']}")
        print(f"  Category: {metrics['category']}")
        print(f"  Overall Score: {metrics['overall_score']:.1f}%")

        score = metrics['overall_score']
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        print(f"  Grade: {grade}")
        print()

    def evaluate_all_queries(self) -> Dict[str, Any]:
        """Evaluate agent on all test cases."""
        load_dotenv()

        print("\n" + "="*80)
        print("CRICKET ANALYST AGENT EVALUATION")
        print("="*80 + "\n")

        test_cases = self.create_test_cases()
        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'#'*80}")
            print(f"# TEST {i}/{len(test_cases)}")
            print(f"{'#'*80}\n")

            metrics = self.evaluate_single_query(test_case["query"], test_case)
            results.append(metrics)

            print(f"\n{'-'*80}\n")
            time.sleep(1)  # Brief pause between queries

        # Generate overall report
        self._generate_overall_report(results, test_cases)

        return {
            "test_cases": test_cases,
            "results": results
        }

    def _generate_overall_report(self, results: List[Dict], test_cases: List[Dict]):
        """Generate overall evaluation report."""
        print("\n" + "="*80)
        print("OVERALL EVALUATION REPORT")
        print("="*80 + "\n")

        # Summary table
        print(f"{'Test Name':<35} {'Category':<18} {'Success %':<12} {'Efficiency %':<14} {'Correct %':<12} {'Overall %':<12} {'Grade'}")
        print("-"*115)

        for metrics in results:
            test_name = metrics["test_name"][:33]
            category = metrics["category"][:16]
            success = metrics["task_success"]["success_rate"]
            efficiency = metrics["efficiency"]["efficiency_score"]
            correctness = metrics["correctness"]["correctness_score"]
            overall = metrics["overall_score"]

            if overall >= 90:
                grade = "A"
            elif overall >= 80:
                grade = "B"
            elif overall >= 70:
                grade = "C"
            elif overall >= 60:
                grade = "D"
            else:
                grade = "F"

            print(f"{test_name:<35} {category:<18} {success:<12.1f} {efficiency:<14.1f} {correctness:<12.1f} {overall:<12.1f} {grade}")

        # Calculate averages
        avg_success = sum(m["task_success"]["success_rate"] for m in results) / len(results)
        avg_efficiency = sum(m["efficiency"]["efficiency_score"] for m in results) / len(results)
        avg_correctness = sum(m["correctness"]["correctness_score"] for m in results) / len(results)
        avg_overall = sum(m["overall_score"] for m in results) / len(results)

        print("-"*115)
        print(f"{'AVERAGE':<35} {'':<18} {avg_success:<12.1f} {avg_efficiency:<14.1f} {avg_correctness:<12.1f} {avg_overall:<12.1f}")
        print("="*115 + "\n")

        # Category breakdown
        print("\nPERFORMANCE BY CATEGORY:")
        print("-"*80)

        categories = {}
        for metrics in results:
            cat = metrics["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(metrics["overall_score"])

        for category, scores in categories.items():
            avg_score = sum(scores) / len(scores)
            print(f"  {category:<25} Avg Score: {avg_score:.1f}%  ({len(scores)} tests)")

        print("\n" + "="*80 + "\n")

        # Save comprehensive report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(results),
            "results": results,
            "averages": {
                "success_rate": round(avg_success, 2),
                "efficiency": round(avg_efficiency, 2),
                "correctness": round(avg_correctness, 2),
                "overall": round(avg_overall, 2)
            },
            "category_breakdown": {
                cat: round(sum(scores) / len(scores), 2)
                for cat, scores in categories.items()
            }
        }

        report_file = self.results_dir / "cricket_analyst_evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Comprehensive evaluation report saved to: {report_file}\n")


def run_evaluation():
    """Run complete evaluation on cricket analyst agent."""
    evaluator = CricketAnalystEvaluator()
    return evaluator.evaluate_all_queries()


if __name__ == '__main__':
    run_evaluation()