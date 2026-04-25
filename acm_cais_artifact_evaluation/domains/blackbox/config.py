"""Configuration and setup utilities for gepa_blog blackbox optimization."""

import argparse
import os
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OptimizeAnything Blackbox Function Optimization"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-5.1",
        help="LLM model to use (default: openai/gpt-5.1)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="gepa_blog",
        help="Optional suffix for log and wandb run names",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory (if not provided, will be auto-generated)",
    )
    parser.add_argument(
        "--num-proposals",
        type=int,
        default=None,
        help="Number of candidate proposals (overrides optimization level)",
    )
    parser.add_argument(
        "--evaluation-budget",
        type=int,
        default=100,
        help="Evaluation budget per candidate (max number of evaluation calls per code candidate)",
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        default=None,
        help="Index of the problem in experiments/polynomial/problems.py (0-55).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--timeout-per-trial",
        type=float,
        default=1.5,
        help="Timeout per evaluation trial in seconds (default: 2)",
    )
    parser.add_argument(
        "--top-k-trajectory",
        type=int,
        default=1000,
        help="Number of top solutions to pass in trajectory (0 for all, default: 1000)",
    )
    return parser.parse_args()


def get_log_directory(args):
    """Get or create log directory."""
    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = time.strftime("%y%m%d_%H:%M:%S")
        model_name = args.llm_model.replace("openai/", "").replace("/", "_")
        log_dir = f"experiments/polynomial/outputs/{args.run_name}/problem_{args.problem_index}/{model_name}/{args.seed}/{timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir
