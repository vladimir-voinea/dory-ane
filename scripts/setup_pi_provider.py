#!/usr/bin/env python3
"""Register Dory as a provider in pi (or any models.json based harness).

    python scripts/setup_pi_provider.py            # add the provider + enable it
    python scripts/setup_pi_provider.py --port 8008 --model-dir ./dory-1.0-4b-ane-coreml

Adds to ~/.pi/agent/models.json and lists the model in ~/.pi/agent/settings.json so it
shows up in the /model picker and Ctrl+P cycling.  Both files are backed up first.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

AGENT_DIR = Path.home() / ".pi" / "agent"
PROVIDER = "dory-ane"
MODEL_ID = "dory-1.0-4b-ane"


def backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_suffix(path.suffix + f".bak-dory-{stamp}")
    shutil.copy2(path, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    if not AGENT_DIR.exists():
        print(f"pi is not installed ({AGENT_DIR} is missing)")
        return 1

    models_path = AGENT_DIR / "models.json"
    models = json.loads(models_path.read_text()) if models_path.exists() else {"providers": {}}
    models.setdefault("providers", {})[PROVIDER] = {
        "api": "openai-completions",
        "apiKey": "no-key",
        "baseUrl": f"http://127.0.0.1:{args.port}/v1",
        "compat": {
            "maxTokensField": "max_tokens",
            "reasoningContentField": "reasoning_content",
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        },
        "models": [{
            "id": MODEL_ID,
            "name": "Dory 1.0 4B (ANE, 24k)",
            "contextWindow": args.context_window,
            "maxTokens": args.max_tokens,
            "input": ["text"],
            "reasoning": False,
            "cost": {"cacheRead": 0, "cacheWrite": 0, "input": 0, "output": 0},
        }],
    }
    print("backup:", backup(models_path) if models_path.exists() else "(new file)")
    models_path.write_text(json.dumps(models, indent=1, sort_keys=True))

    settings_path = AGENT_DIR / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    enabled = settings.get("enabledModels") or []
    for pattern in (f"{PROVIDER}/{MODEL_ID}", MODEL_ID):
        if pattern not in enabled:
            enabled.append(pattern)
    settings["enabledModels"] = enabled
    print("backup:", backup(settings_path) if settings_path.exists() else "(new file)")
    settings_path.write_text(json.dumps(settings, indent=1))

    print(f"\nregistered {PROVIDER}/{MODEL_ID} -> http://127.0.0.1:{args.port}/v1")
    print("check with:  pi --list-models dory")
    print("use with:    pi --provider dory-ane --model dory-1.0-4b-ane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())