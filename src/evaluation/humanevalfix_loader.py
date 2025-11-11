import json
import os

class HumanEvalFixLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        self.examples = []

    def load_dataset(self):
        """Load dataset from JSON or JSONL file.
        
        Supports:
        - JSON file with 'examples' key: {"examples": [...]}
        - JSON file with array: [...]
        - JSONL file: one JSON object per line
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        # Check if file is JSONL (ends with .jsonl)
        if self.dataset_path.endswith('.jsonl'):
            self._load_jsonl()
        else:
            self._load_json()

    def _load_jsonl(self):
        """Load JSONL file (one JSON object per line)."""
        self.examples = []
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    self.examples.append(example)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {self.dataset_path}: {e}")
        self.data = {'examples': self.examples}

    def _load_json(self):
        """Load JSON file (supports both dict with 'examples' key and direct array)."""
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        
        # Handle different JSON formats
        if isinstance(self.data, dict):
            # If it has 'examples' key, use that
            if 'examples' in self.data:
                self.examples = self.data['examples']
            else:
                # Otherwise, treat the whole dict as a single example
                self.examples = [self.data]
                self.data = {'examples': self.examples}
        elif isinstance(self.data, list):
            # Direct array of examples
            self.examples = self.data
            self.data = {'examples': self.examples}
        else:
            # Single value, wrap it
            self.examples = [self.data]
            self.data = {'examples': self.examples}

    def get_examples(self):
        if not self.examples and not self.data:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if self.examples:
            return self.examples
        return self.data.get('examples', [])

    def get_example(self, index):
        examples = self.get_examples()
        if index < 0 or index >= len(examples):
            raise IndexError("Index out of range.")
        return examples[index]
