# Uncertainty-Based Code Fixing Agent

An LLM-based agent that uses uncertainty estimation to selectively apply code fixes, applying only changes the model is confident about.

## Research Motivation

Uncertainty-based methods are widely used in robotics for safe action selection (e.g., KnowNo, conformal prediction methods). The principle is to formulate the next robot task as a Multiple Choice Question Answering (MCQA) problem over possible actions and assign probabilities to each option using model logits (as in KnowNo, LAP, or more stable options like LofreeCP). Conformal Prediction (CP) is then applied: if only one option remains after CP, the robot executes the action; otherwise, it asks for help.

Recent studies (AmbiK, Ivanova et al. 2025) have shown that since LLM logits are often miscalibrated, sometimes the simplest binary approach outperforms complex CP-based methods. This suggests that labeling each option as "certain" or "uncertain" may be informative.

In the context of code fixing (unlike robot actions), multiple options from MCQA may be possible simultaneously—we can apply multiple fixes if the model is certain about them.

## Approach

This agent implements a three-stage uncertainty-based code fixing pipeline:

1. **Diagnosis**: The LLM analyzes the buggy code and identifies what needs to be fixed
2. **Action Generation**: The LLM generates multiple concrete fix options (MCQA)
3. **Binary Uncertainty Classification**: Each fix option is labeled as either:
   - **"certain"**: The model is confident this fix will correctly address the bug
   - **"uncertain"**: The model is not confident or the fix is risky
4. **Selective Application**: Only fixes marked as "certain" are applied to the code

This approach allows the agent to be conservative: it only makes changes when confident, reducing the risk of introducing new bugs.

## Implementation

### Core Components

- **Agent** (`src/agent/agent.py`): Main agent that orchestrates the fixing pipeline
- **LLM Client** (`src/agent/tools/llm_client.py`): Interface to Ollama for LLM inference
- **Code Interpreter** (`src/agent/tools/code_interpreter.py`): Safe execution environment for code
- **Dataset Loader** (`src/evaluation/humanevalfix_loader.py`): Loads HumanEvalFix dataset

### Key Features

- **Binary Uncertainty Classification**: Simple certain/uncertain labeling (inspired by AmbiK findings)
- **Multi-step Fixing**: Can apply multiple certain fixes sequentially
- **Safe Execution**: Code is executed in a sandboxed environment
- **HumanEvalFix Evaluation**: Evaluates on the HumanEvalFix benchmark

## Installation

```bash
pip install -r requirements.txt
```

**Note**: This agent requires Ollama to be running locally. Install Ollama from [ollama.ai](https://ollama.ai) and pull a model (e.g., `ollama pull qwen2.5:7b`).

## Usage

### Running Evaluation

Evaluate the agent on the HumanEvalFix dataset:

```bash
python scripts/evaluate_agent.py --dataset path/to/humanevalpack.jsonl --subsample 10 --model qwen3:0.6b
```

### Using the Agent Programmatically

```python
from agent.agent import Agent
from agent.tools.llm_client import OllamaClient

# Initialize agent with Ollama
llm = OllamaClient(model='qwen3:0.6b')
agent = Agent(llm_client=llm)

# Fix buggy code
buggy_code = """
def add(a, b):
    return a - b
"""

result = agent.fix_code(buggy_code, "Fix the add function to return the sum")
print(result['final_code'])
```

## Experimental Results and Limitations

**Current Limitations with Small Models**: 

When tested with `qwen3:0.6b` (0.6B parameters), the model tended to label all fix options as "uncertain", resulting in no fixes being applied. This suggests that:

1. The model may be too small to reliably assess fix correctness
2. Small models may be overly cautious when asked to express certainty
3. The binary classification approach may need larger, more capable models to be effective

**Hypothesis**: 

This uncertainty-based approach should be validated with larger LLMs (e.g., 7B+ parameters) that have better calibration and reasoning capabilities. The binary classification method may prove effective with models that can more reliably distinguish between certain and uncertain fixes.

## Dataset

The agent is evaluated on the HumanEvalFix dataset, which contains buggy Python code with corresponding tests. The dataset loader supports both JSON and JSONL formats and automatically constructs complete function definitions from declarations and bodies.

## Project Structure

```
humanevalfix-agent/
├── src/
│   ├── agent/
│   │   ├── agent.py              # Main agent implementation
│   │   ├── executor.py           # Code execution wrapper
│   │   └── tools/
│   │       ├── llm_client.py     # Ollama LLM client
│   │       └── code_interpreter.py  # Safe code execution
│   └── evaluation/
│       ├── humanevalfix_loader.py  # Dataset loader
│       └── metrics.py            # Evaluation metrics
├── scripts/
│   ├── evaluate_agent.py         # Evaluation script
│   └── run_agent_and_eval.py     # Batch evaluation runner
└── requirements.txt
```

## Future Work

- Test with larger LLMs (7B+ parameters) to validate the uncertainty-based approach
- Experiment with different uncertainty thresholds and calibration methods
- Compare binary classification against multi-level certainty and conformal prediction
- Investigate prompt engineering to improve certainty assessment in small models
- Add support for asking for human help when all fixes are uncertain

## References

- KnowNo: Uncertainty-based action selection in robotics
- Conformal Prediction (Vovk et al.)
- AmbiK, Ivanova et al. 2025: Binary uncertainty approaches for miscalibrated LLMs
- HumanEvalFix: Code fixing benchmark dataset

## License

MIT License
