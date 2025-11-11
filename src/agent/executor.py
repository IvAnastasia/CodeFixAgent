class Executor:
    def __init__(self, code_interpreter):
        self.code_interpreter = code_interpreter

    def execute_code(self, code: str, timeout: int = 5, memory_limit_mb: int = 200):
        """
        Execute the provided Python code via the configured CodeInterpreter.

        Returns a tuple (result_dict, error_str). result_dict follows the shape
        returned by CodeInterpreter.execute/run.
        """
        try:
            # Prefer structured execution API when available
            if hasattr(self.code_interpreter, 'execute_structured'):
                result = self.code_interpreter.execute_structured(code, timeout=timeout, memory_limit_mb=memory_limit_mb)
                return result, None
            result = self.code_interpreter.execute(code, timeout=timeout, memory_limit_mb=memory_limit_mb)
            # backward compat: if interpreter returned a simple string, wrap
            if isinstance(result, str):
                return ({"returncode": 0, "stdout": result, "stderr": "", "timed_out": False}, None)
            return result, None
        except Exception as e:
            return None, str(e)

    def evaluate_code(self, code: str, timeout: int = 5, memory_limit_mb: int = 200):
        output, error = self.execute_code(code, timeout=timeout, memory_limit_mb=memory_limit_mb)
        return {
            "output": output,
            "error": error
        }