import time
import json
import requests
from langchain_ollama import OllamaLLM as Ollama
import os

class OllamaClient:
    """Safe wrapper around langchain_ollama's OllamaLLM with connection checks and modern API."""

    def __init__(self, model: str = "qwen3:0.6b", base_url: str = "http://localhost:11434", timeout: int = 5):
        try:
            from langchain_ollama import OllamaLLM as Ollama  # type: ignore
        except Exception as exc:
            raise ImportError("langchain_ollama is required for OllamaClient: " + str(exc))

        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Wait for the Ollama server to become available before creating the client
        self._wait_for_server()

        # Initialize client after confirming the server is ready
        self._client = Ollama(model=model, base_url=base_url)

    def _wait_for_server(self):
        """Poll Ollama server until it responds or timeout reached."""
        start = time.time()
        url = f"{self.base_url}/api/tags"
        while time.time() - start < self.timeout:
            try:
                resp = requests.get(url, timeout=1)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.5)
        raise ConnectionError(f"Ollama server not reachable at {self.base_url}. "
                              f"Make sure it's running with `ollama serve`.")

    def ask(self, prompt: str, **kwargs) -> str:
        """Send a simple text prompt and return model output."""
        try:
            resp = self._client.invoke(prompt)  # ✅ new API replaces deprecated __call__
            if isinstance(resp, str):
                return resp
            # Handle structured LangChain return types
            if hasattr(resp, "generations"):
                gens = resp.generations
                if gens and len(gens) > 0:
                    first = gens[0][0] if isinstance(gens[0], list) else gens[0]
                    if hasattr(first, "text"):
                        return first.text
            if isinstance(resp, dict) and "text" in resp:
                return resp["text"]
            return str(resp)
        except Exception as exc:
            return f"<ollama-error> {exc}"

    def get_suggestions(self, buggy_code: str, user_prompt: str, n_options: int = 4, model: str = None) -> dict:
        """Implements the staged questioning flow (diagnosis → actions → certainties)."""
        model = model or self.model

        def _ask(messages):
            return self.ask("\n".join(m["content"] for m in messages))

        # 1️⃣ Diagnose
        system = ("You are a helpful Python bug-fixing assistant. Given buggy code and a short user prompt, "
                  "describe the concrete issues to fix. Be concise and numbered.")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Buggy code:\n{buggy_code}\n\nUser request: {user_prompt}\n\nWhat should be fixed?"}
        ]
        what_to_fix = _ask(messages)

        # 2️⃣ Generate action options
        messages = [
            {"role": "system", "content": "Convert the previous diagnostic into a short list of concise actions."},
            {"role": "user", "content": f"Diagnosis:\n{what_to_fix}\n\nProduce {n_options} short, numbered action options (1 per line)."}
        ]
        options_text = _ask(messages)

        # Parse numbered options
        options = []
        for line in options_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if parts and parts[0].rstrip(".").rstrip(")").isdigit():
                rest = line[line.find(parts[1]):].strip() if len(parts) > 1 else line
            else:
                rest = line
            options.append(rest)
            if len(options) >= n_options:
                break

        # 3️⃣ Ask for binary certainty on each (certain/uncertain)
        # For small models, we encourage marking as "certain" when the fix is straightforward
        certainties = []
        for opt in options:
            messages = [
                {"role": "system", "content": (
                    "You are a Python code fixing assistant. Respond EXACTLY in JSON format: "
                    "{\"certainty\": \"certain\" or \"uncertain\", \"reason\": \"brief explanation\"}. "
                    "Mark as 'certain' if the action clearly and directly fixes the identified bug. "
                    "Mark as 'uncertain' only if the action is vague, risky, or might cause side effects. "
                    "Be confident when the fix is straightforward (e.g., changing an operator, fixing a condition)."
                )},
                {"role": "user", "content": (
                    f"Buggy code:\n{buggy_code}\n\n"
                    f"Diagnosis: {what_to_fix}\n\n"
                    f"Action: {opt}\n\n"
                    f"Based on the buggy code and diagnosis, is this action certain to fix the bug? "
                    f"Answer with JSON: {{\"certainty\": \"certain\" or \"uncertain\", \"reason\": \"...\"}}"
                )}
            ]
            raw_resp = _ask(messages)
            try:
                # Try to parse JSON, handle markdown code blocks
                resp_clean = raw_resp.strip()
                if resp_clean.startswith('```'):
                    # Extract JSON from code block
                    lines = resp_clean.split('\n')
                    json_lines = [l for l in lines if not l.strip().startswith('```')]
                    resp_clean = '\n'.join(json_lines).strip()
                parsed = json.loads(resp_clean)
                # Normalize: convert to binary certainty
                cert = parsed.get("certainty", "").lower().strip()
                if cert in ("high", "certain", "yes", "true", "confident"):
                    parsed["certainty"] = "certain"
                elif cert in ("low", "medium", "uncertain", "no", "false", "unsure"):
                    parsed["certainty"] = "uncertain"
                else:
                    # Default to uncertain if unclear
                    parsed["certainty"] = "uncertain"
            except Exception:
                # If JSON parsing fails, try to extract certainty from text
                resp_lower = raw_resp.strip().lower()
                # Look for positive indicators
                if any(word in resp_lower for word in ["certain", "confident", "yes", "correct", "fix", "will work", "straightforward"]):
                    parsed = {"certainty": "certain", "reason": f"Extracted from: {raw_resp.strip()[:100]}"}
                else:
                    parsed = {"certainty": "uncertain", "reason": f"Extracted from: {raw_resp.strip()[:100]}"}
            certainties.append(parsed)

        return {
            "what_to_fix": what_to_fix,
            "options_text": options_text,
            "options": options,
            "certainties": certainties,
        }