class Executor:
    def __init__(self, code_interpreter):
        self.code_interpreter = code_interpreter

    def execute_code(self, code: str, timeout: int = 5, memory_limit_mb: int = 200):
        """
        Execute Python code via the configured CodeInterpreter.
        Returns a tuple: (result_dict, error_str)
        """
        try:
            # Try structured execution first
            exec_fn = getattr(self.code_interpreter, 'execute_structured', None) or getattr(self.code_interpreter, 'execute')
            result = exec_fn(code, timeout=timeout, memory_limit_mb=memory_limit_mb)

            # Wrap string outputs in a dict for backward compatibility
            if isinstance(result, str):
                result = {"returncode": 0, "stdout": result, "stderr": "", "timed_out": False}

            return result, None
        except Exception as e:
            return None, str(e)

    def evaluate_code(self, code: str, timeout: int = 5, memory_limit_mb: int = 200):
        result, error = self.execute_code(code, timeout=timeout, memory_limit_mb=memory_limit_mb)
        return {"output": result, "error": error}
