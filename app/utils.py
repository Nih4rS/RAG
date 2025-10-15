import yaml
from pathlib import Path
from typing import List, Tuple, Dict

def load_eval_yaml(path: str = "app/eval_items.yaml") -> Dict[str, Dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Eval YAML not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Eval YAML root must be a mapping")
    return data
