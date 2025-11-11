"""Evaluate the agent on a HumanEvalFix-style JSON dataset and compute pass@1.

This script supports running on a small subsample for quick experiments and
can use either the MockLLM or a real Qwen-compatible server.

Dataset format expected by HumanEvalFixLoader:
  A JSON file where top-level is an object with key "examples" containing a list
  of example objects. Each example should contain at least:
    - "code": the buggy Python source (string)
    - optionally "expected_fixed_code" or "fixed_code": the reference fixed code
    - optionally "user_prompt": the human request/explanation

Usage:
  python3 scripts/evaluate_agent.py --dataset path/to/dataset.json --subsample 20
"""
import argparse
import json
import os
import random
import sys
import subprocess
import tempfile
from typing import List, Dict, Any

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation.humanevalfix_loader import HumanEvalFixLoader
from src.evaluation.metrics import Metrics
from src.agent.tools.llm_client import OllamaClient
from src.agent.agent import Agent

def load_examples(dataset_path: str) -> List[Dict[str, Any]]:
    """Load examples from a dataset file (JSON or JSONL).
    
    The HumanEvalFixLoader now supports both JSON and JSONL formats, as well as
    different structures (dict with 'examples' key, direct array, etc.).
    The humaneval-x dataset provides files in various formats, so we normalize
    the field names to match what the evaluation expects.
    
    For HumanEvalPack format, constructs complete functions from declaration + body.
    """
    # Use HumanEvalFixLoader which now supports JSONL, JSON arrays, and JSON objects
    loader = HumanEvalFixLoader(dataset_path)
    loader.load_dataset()
    raw_examples = loader.get_examples()
    
    # Normalize field names: humaneval-x uses different field names than expected
    # Map humaneval-x fields to our expected format
    examples = []
    for item in raw_examples:
        # For HumanEvalPack format: need to construct complete function from declaration + body
        declaration = item.get('declaration', '')
        buggy_body = item.get('buggy_solution', '')
        canonical_body = item.get('canonical_solution', '')
        
        # If we have declaration and body, construct complete function
        if declaration and buggy_body:
            # Construct complete buggy function: declaration + body
            buggy_code = declaration + buggy_body
        else:
            # Fallback: use code directly if available (for other formats)
            buggy_code = item.get('code') or buggy_body or item.get('canonical_solution') or ''
        
        # Construct canonical solution if available
        if declaration and canonical_body:
            expected_fixed_code = declaration + canonical_body
        elif 'expected_fixed_code' in item:
            expected_fixed_code = item['expected_fixed_code']
        elif 'canonical_solution' in item:
            expected_fixed_code = item['canonical_solution']
        else:
            expected_fixed_code = None
        
        # Extract test code
        test = item.get('test') or item.get('example_test')
        # Extract prompt/instruction (use prompt which includes the function signature for context)
        prompt = item.get('prompt') or item.get('instruction') or item.get('user_prompt') or 'Fix the code'
        
        # Create normalized example dict
        example = {
            'code': buggy_code,  # Complete function (declaration + body)
            'user_prompt': prompt,
            'test': test,
            'declaration': declaration,  # Keep for reference
        }
        
        # Preserve expected fixed code
        if expected_fixed_code:
            example['expected_fixed_code'] = expected_fixed_code
        
        # Preserve other useful fields
        if 'task_id' in item:
            example['task_id'] = item['task_id']
        if 'entry_point' in item:
            example['entry_point'] = item['entry_point']
        
        examples.append(example)
    
    return examples


def tiny_default_examples() -> List[Dict[str, Any]]:
    return [
        {
            'code': "def add(a,b):\n    return a - b\n",
            'expected_fixed_code': "def add(a,b):\n    return a + b\n",
            'user_prompt': 'Make add return the sum of two numbers'
        },
        {
            'code': "def is_even(n):\n    return n % 2 == 1\n",
            'expected_fixed_code': "def is_even(n):\n    return n % 2 == 0\n",
            'user_prompt': 'Fix is_even to return True for even numbers'
        }
    ]


def check_fixed_code_matches(expected: str, produced: str) -> bool:
    """Execution-based check: if a test is provided (in `expected` param when used that way),
    run the produced code combined with the test and return True if tests pass (exit code 0).

    This function remains a fallback for textual comparison when no tests are available.
    """
    # If expected contains test code (heuristic: contains 'assert' or 'pytest' or 'unittest'),
    # treat it as test content and run produced code against it.
    if expected and any(tok in expected for tok in ('assert ', 'pytest', 'unittest', 'def test_')):
        return run_code_with_test(produced, expected)

    # Fallback textual comparison
    if not expected:
        return False
    e = '\n'.join([line.rstrip() for line in expected.strip().splitlines() if line.strip()])
    p = '\n'.join([line.rstrip() for line in produced.strip().splitlines() if line.strip()])
    if e == p:
        return True
    if e in p:
        return True
    return False


def run_code_with_test(code: str, test_code: str, timeout: int = 8) -> bool:
    """Write code and test to a temporary Python file and execute it.

    Returns True if the process exits with code 0, False otherwise.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
        fname = tmp.name
        # Ensure the test runs the functions defined in code. Place code first, then test.
        tmp.write(code)
        tmp.write('\n\n')
        tmp.write(test_code)

    try:
        # Use the same Python interpreter to run the test
        proc = subprocess.run([sys.executable, fname], capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(fname)
        except Exception:
            pass


def evaluate(agent: Agent, examples: List[Dict[str, Any]], n_options: int = 4) -> Dict[str, Any]:
    total = len(examples)
    correct = 0
    details = []

    for ex in examples:
        buggy = ex.get('code', '')
        prompt = ex.get('user_prompt', 'Fix the code')
        declaration = ex.get('declaration', '')
        # Prefer an explicit test harness when available (HumanEval-X provides a
        # `test` field). If not present, fall back to expected_fixed_code / fixed_code.
        test_code = ex.get('test') or ex.get('example_test')
        expected_code = ex.get('expected_fixed_code') or ex.get('fixed_code') or ex.get('expected_output')

        out = agent.fix_code(buggy, prompt, n_options=n_options)
        final_code = out.get('final_code') if isinstance(out, dict) else ''
        
        # Ensure final_code is a complete function if we have a declaration
        # The agent might return just the function body, so we need to prepend the declaration
        if declaration and final_code:
            # Check if final_code already includes the declaration (has function def)
            if not final_code.strip().startswith('def ') and not final_code.strip().startswith('from '):
                # Agent returned just the body, prepend declaration
                final_code = declaration + final_code
            # If it does start with 'def', it should be complete, but verify it matches our declaration
            elif declaration and not final_code.startswith(declaration.strip()):
                # The agent might have changed the function signature, but we need the original
                # for the test to work. Try to extract just the body and prepend our declaration.
                # This is a heuristic: if the code has 'def' but doesn't match our declaration,
                # we assume the agent returned a modified function. We'll use it as-is.
                pass

        # If test_code exists, use execution-based matching (run produced code + test).
        if test_code:
            ok = check_fixed_code_matches(test_code, final_code or '')
        else:
            ok = check_fixed_code_matches(expected_code or '', final_code or '')
        if ok:
            correct += 1

        details.append({
            'buggy': buggy,
            'expected': test_code or expected_code,
            'final_code': final_code,
            'ok': ok,
        })

    score = Metrics.pass_at_1(correct, total)
    return {'total': total, 'correct': correct, 'pass@1': score, 'details': details}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default=None, help='Path to HumanEvalFix JSON dataset')
    p.add_argument('--subsample', type=int, default=0, help='Evaluate on a random subsample of this size (0 = all)')
    p.add_argument('--model', default='qwen3-0.6b', help='Model name to request from the server')
    p.add_argument('--save', type=str, default=None, help='Path to write JSON results')
    args = p.parse_args()

    llm = OllamaClient(model=args.model)
    agent = Agent(llm_client=llm)

    dataset_provided = bool(args.dataset)
    # Prefer JSONL format as it's complete (has all fields), fallback to JSON if JSONL doesn't exist
    default_humanevalx_jsonl = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation', 'create', 'humaneval-x', 'data', 'python', 'data', 'humanevalpack.jsonl')
    default_humanevalx_json = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation', 'create', 'humaneval-x', 'data', 'python', 'data', 'humanevalpack.json')
    
    if args.dataset:
        if not os.path.exists(args.dataset):
            print(f"Dataset path {args.dataset} not found.")
            return
        examples = load_examples(args.dataset)
    elif os.path.exists(os.path.normpath(default_humanevalx_jsonl)):
        print(f"Using humaneval-x python pack at {default_humanevalx_jsonl}")
        examples = load_examples(os.path.normpath(default_humanevalx_jsonl))
    elif os.path.exists(os.path.normpath(default_humanevalx_json)):
        print(f"Using humaneval-x python pack at {default_humanevalx_json} (note: JSON format has fewer fields than JSONL)")
        examples = load_examples(os.path.normpath(default_humanevalx_json))
    else:
        examples = tiny_default_examples()

    if args.subsample and args.subsample > 0 and args.subsample < len(examples):
        examples = random.sample(examples, args.subsample)

    print(f"Running evaluation on {len(examples)} examples)...")
    results = evaluate(agent, examples)

    print(f"pass@1 = {results['pass@1']:.3f} ({results['correct']}/{results['total']})")
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.save}")


if __name__ == '__main__':
    main()
