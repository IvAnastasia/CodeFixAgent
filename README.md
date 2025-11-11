# Uncertainty-Based Code Fixing Agent

An LLM-based agent that uses uncertainty estimation to selectively apply code fixes, applying only changes the model is confident about.

## Motivation

In robotics, researchers use uncertainty to help robots decide when to act and when to ask for help. The basic approach (Ren et al., 2023; Quach et al., 2024) works like this:
1. Frame the robot's next action as a multiple choice question answering task (options A, B etc.)
2. Use the model's logits to assign probabilities to each option
3. Apply Conformal Prediction (Vovk et al., 1999; Angelopoulos & Bates, 2021) to filter uncertain actions
4. If only one option remains, do it; otherwise, ask a human for help

This is useful for safety-critical applications where you don't want the robot doing something it's not sure about.

Recent studies (Guo et al. 2017, Ivanova et al. 2025) have shown that since LLM logits are often miscalibrated, sometimes the simplest binary approach outperforms complex CP-based methods. This suggests that labeling each option as "certain" or "uncertain" may be informative.

In the context of code fixing (unlike robot actions), multiple options from MCQA may be possible simultaneously. We apply multiple fixes if the model is certain about them.

Note on ReAct (Yao et al., 2023) approach: 

ReAct is the agent architecture where the model iteratively reasons and acts. However, recent work like AutoGuide (Hu et al., 2024) has shown that ReAct is often outperformed by simpler approaches. ReAct's interleaved reasoning can actually hurt performance because the model gets distracted by its own reasoning traces or early reasoning mistakes propagate through iterations. So if I had more resources I would actually so something AutoGuide-style - using offline experience to compare trajectories and propose high-level guidelines for the similar situations (similar context (envireonment) state for robots, similar code problem type is our case).

## Approach

This agent implements a three-stage uncertainty-based code fixing pipeline:

1. **Diagnosis**: The LLM analyzes the buggy code and identifies what needs to be fixed
2. **Action Generation**: The LLM generates multiple concrete fix options (MCQA)
3. **Binary Uncertainty Classification**: Each fix option is labeled as either:
   - **"certain"**: The model is confident this fix will correctly address the bug
   - **"uncertain"**: The model is not confident and the fix is risky
4. **Fixing Code**: Only fixes marked as "certain" are applied to the code

This approach only makes changes when LLM is confident, reducing the risk of introducing new bugs.

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
│   │   ├── agent.py            
│   │   ├── executor.py          
│   │   └── tools/
│   │       ├── llm_client.py     
│   │       └── code_interpreter.py
│   └── evaluation/
│       ├── humanevalfix_loader.py  
│       └── metrics.py           
├── scripts/
│   ├── evaluate_agent.py         
│   └── run_agent_and_eval.py    
└── requirements.txt
```

## Future Work

- Test with larger LLMs (7B+ parameters) to validate the uncertainty-based approach
- Experiment with different uncertainty thresholds and calibration methods
- Compare binary classification against multi-level certainty and conformal prediction
- Investigate prompt engineering to improve certainty assessment in small models

## References

Ren, A. Z., Dixit, A., Bodreau, A., Singh, S., Agarwal, S., Bisk, Y., & Narasimhan, K. (2023). KnowNo: Safe Human-Robot Collaboration through Conformal Prediction. ICRA.
Quach, K. G., Xu, B., Tian, Y., & Anderson, P. (2024). LAP: Language-Conditioned Affordance Prediction with Conformal Prediction. arXiv preprint arXiv:2410.08223.

Vovk, V., Gammerman, A., & Shafer, G. (1999). Algorithmic Learning in a Random World. Springer.
Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv preprint arXiv:2107.07511.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.

Ivanova, A., Bakaeva, E., Volovikova, Z., Kovalev, A. K., and Panov, A. I. (2025). AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment. ACL 2025.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.

Hu, J., Chen, T., Huang, S., Lou, J.-G., Muennighoff, N., Nagappan, M., & Zhao, J. (2024). AutoGuide: Automated Generation and Selection of State-Aware Guidelines for Large Language Model Agents. arXiv preprint arXiv:2403.08978.

Muennighoff, N., Liu, Q., Zebaze, A., Zheng, Q., Hui, B., Zhuo, T. Y., ... & von Werra, L. (2023). OctoPack: Instruction Tuning Code Large Language Models.
