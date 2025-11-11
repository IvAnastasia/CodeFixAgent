"""A small sandboxed code interpreter used by the agent.

This runs untrusted Python code in a separate process with a timeout and
basic resource limits. It captures stdout/stderr and returns structured
results. This is intentionally small and conservative — for production use
you should run code inside a proper sandbox (container, VM, or remote runner).
"""
import tempfile
import subprocess
import sys
import os
import textwrap
from typing import Dict, Any, Optional

try:
    import resource
except Exception:
    resource = None


class CodeInterpreter:
    def __init__(self):
        pass

    def _make_wrapper(self, user_code: str) -> str:
        # Wrap user code to ensure prints are flushed and errors are shown
        wrapper = textwrap.dedent(
            f"""
            import sys, traceback
            try:
                # user code starts
{textwrap.indent(user_code, '                ')}
                # user code ends
            except Exception:
                traceback.print_exc()
                sys.exit(1)
            """
        )
        return wrapper

    def execute(self, code: str, timeout: int = 5, memory_limit_mb: int = 200) -> str:
        """Execute `code` in a subprocess and return outputs.

        Args:
            code: Python source to execute.
            timeout: wall-clock timeout in seconds.
            memory_limit_mb: soft memory limit in megabytes.

        Returns:
            dict with keys: returncode, stdout, stderr, timed_out (bool)
        """
        wrapper = self._make_wrapper(code)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            fname = f.name

        cmd = [sys.executable, fname]

        def _preexec():
            # apply simple POSIX resource limits if available
            if resource is None:
                return
            try:
                # CPU time (seconds)
                resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
            except Exception:
                pass
            try:
                # address space limit (bytes)
                mem_bytes = memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except Exception:
                pass

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                preexec_fn=_preexec if os.name != "nt" else None,
                check=False,
                text=True,
            )
            result = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            # best-effort kill already done by subprocess
            result = {
                "returncode": None,
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + "\nExecution timed out",
                "timed_out": True,
            }
        finally:
            try:
                os.remove(fname)
            except Exception:
                pass


        # For backward compatibility we return a combined stdout+stderr string
        out = (result.get('stdout') or '') + (result.get('stderr') or '')
        return out

    # structured execution API for newer components
    def execute_structured(self, code: str, timeout: int = 5, memory_limit_mb: int = 200) -> Dict[str, Any]:
        """Return the structured dict (returncode, stdout, stderr, timed_out)."""
        wrapper = self._make_wrapper(code)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            fname = f.name

        cmd = [sys.executable, fname]

        def _preexec():
            if resource is None:
                return
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
            except Exception:
                pass
            try:
                mem_bytes = memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except Exception:
                pass

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                preexec_fn=_preexec if os.name != "nt" else None,
                check=False,
                text=True,
            )
            result = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            result = {
                "returncode": None,
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + "\nExecution timed out",
                "timed_out": True,
            }
        finally:
            try:
                os.remove(fname)
            except Exception:
                pass

        return result

    # small alias to keep earlier names working
    def run(self, code: str, timeout: int = 5, memory_limit_mb: int = 200):
        return self.execute_structured(code, timeout=timeout, memory_limit_mb=memory_limit_mb)


def quick_demo():
    ci = CodeInterpreter()
    return ci.execute("print('hello')")


if __name__ == "__main__":
    print(quick_demo())