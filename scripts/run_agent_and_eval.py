#!/usr/bin/env python3
"""
Helper runner: runs the existing evaluation script with a chosen LLM backend and subsample,
captures and summarizes pass@1 and basic metrics.

Usage:
  python scripts/run_agent_and_eval.py --backend ollama --subsample 50 --save /tmp/hvf_results.json

This script assumes the repository's `scripts/evaluate_agent.py` exists and is runnable
with the current Python virtualenv. It sets `LLM_BACKEND` in the environment and
executes the evaluation script as a subprocess. Results are printed and (optionally) saved.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


def run_evaluation(backend: str, subsample: int, save: Optional[str], extra_args: List[str]):
    repo_py = sys.executable
    eval_script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_agent.py"
    if not eval_script.exists():
        raise SystemExit(f"evaluate_agent.py not found at {eval_script}")

    # Build command for evaluate_agent.py
    cmd = [repo_py, str(eval_script)]
    # Add model flag if backend is specified (evaluate_agent.py uses --model)
    if backend and backend not in ("mock", "none", "fake"):
        cmd += ["--model", backend]
    if subsample and subsample > 0:
        cmd += ["--subsample", str(subsample)]
    if save:
        cmd += ["--save", str(save)]
    if extra_args:
        cmd += extra_args

    env = os.environ.copy()

    print(f"Running evaluation: backend={backend}, subsample={subsample}, save={save}")
    print("Command:", shlex.join(cmd))

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("--- stdout ---")
    print(proc.stdout)
    print("--- stderr ---")
    print(proc.stderr)

    if save and Path(save).exists():
        with open(save, "r") as f:
            data = json.load(f)
        total = data.get("total", 0)
        correct = data.get("correct", 0)
        details = data.get("details", [])
        retrieved = sum(1 for x in details if x.get("final_code") and x.get("final_code").strip())
        precision = (correct / retrieved) if retrieved > 0 else 0.0
        recall = (correct / total) if total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        summary = {
            "total": total,
            "correct": correct,
            "retrieved": retrieved,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        print("=== Evaluation summary ===")
        print(json.dumps(summary, indent=2))
        return data
    else:
        # Try to parse printed JSON if script printed results
        print("No results file saved; returning raw process output.")
        return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ollama", help="LLM backend to use (e.g. ollama, qwen)")
    p.add_argument("--subsample", type=int, default=50, help="Number of examples to evaluate (subsample)")
    p.add_argument("--save", default="/tmp/hvf_results.json", help="Where to save the evaluation JSON (pass to evaluate_agent)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args forwarded to evaluate_agent.py")

    args = p.parse_args()
    extra = args.extra or []
    run_evaluation(args.backend, args.subsample, args.save, extra)


if __name__ == "__main__":
    main()
