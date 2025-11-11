from .tools.llm_client import OllamaClient
from .tools.code_interpreter import CodeInterpreter
from .executor import Executor
import langgraph
import json


class Agent:
    """Agent that follows the multi-prompt certainty flow using Ollama + langgraph.
    
    Uses binary uncertainty classification: only applies code fixes marked as "certain".
    """

    def __init__(self, *args, **kwargs):
        """Initialize the agent with an LLM client and code executor.
        
        Args:
            llm_client: Optional LLM client (defaults to OllamaClient)
            executor: Optional executor for code execution (defaults to Executor with CodeInterpreter)
        """
        llm_client = kwargs.get('llm_client') or (args[0] if len(args) > 0 else OllamaClient())
        executor = kwargs.get('executor') or (args[1] if len(args) > 1 else None)

        self.llm_client = llm_client
        interp = CodeInterpreter()
        self.executor = executor or Executor(interp)

    def fix_code(self, buggy_code: str, user_prompt: str, n_options: int = 4, allowed_certainties=('certain',)):
        # 1) Get structured suggestions
        suggestions = self.llm_client.get_suggestions(buggy_code, user_prompt, n_options=n_options)

        # 2) Filter options by binary certainty (only apply "certain" fixes)
        options = suggestions.get('options', [])
        certs = suggestions.get('certainties', [])
        selected = [
            {'option': opt, 'certainty': cert}
            for opt, cert in zip(options, certs)
            if isinstance(cert, dict) and cert.get('certainty') == 'certain'
        ]

        # If no certain fixes found, log warning but continue (won't apply any fixes)
        if not selected:
            # Could add logging here: "No certain fixes found, skipping code modification"
            pass

        # 3) Apply code edits using langgraph (only for "certain" fixes)
        applied_edits = []
        current_code = buggy_code
        for sel in selected:
            prompt = (
                f"Apply this action to the code. Action: {sel['option']}\n"
                f"Original code:\n{current_code}\n\n"
                "Provide the full modified Python file contents only. Do not include explanations."
            )
            # Run via langgraph
            try:
                new_code = langgraph.run_node(prompt, llm=self.llm_client)
            except Exception:
                # fallback: direct LLM call
                new_code = self.llm_client.ask(prompt, temperature=0.0)

            # Strip markdown fences if present
            if new_code.startswith('```'):
                parts = new_code.split('```')
                if len(parts) >= 3:
                    new_code = parts[2].strip()

            applied_edits.append({'action': sel['option'], 'code': new_code})
            current_code = new_code

        # 4) Execute final code
        exec_result, exec_err = self.executor.execute_code(current_code)

        return {
            'suggestions': suggestions,
            'selected_actions': selected,
            'applied_edits': applied_edits,
            'final_code': current_code,
            'execution': exec_result,
            'execution_error': exec_err,
        }

    def evaluate_fix(self, fixed_code: str):
        """Evaluate whether fixed code executes successfully.
        
        Args:
            fixed_code: The fixed code to evaluate
            
        Returns:
            Dictionary with 'ran' (bool), 'details', and 'error' fields
        """
        res, err = self.executor.execute_code(fixed_code)
        return {
            'ran': res is not None and not res.get('timed_out', False) and (res.get('returncode') == 0),
            'details': res,
            'error': err
        }


def _demo():
    a = Agent()
    buggy = """def add(a,b):\n    return a - b\n"""
    out = a.fix_code(buggy, "Make add return the sum of two numbers")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    _demo()
