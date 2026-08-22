#!/usr/bin/env python3
"""Capybara GUI launcher - opens the built-in web UI in your browser.

The desktop experience is now the web app served by `capybara serve` at
http://127.0.0.1:11434/. Running this file makes sure the server is up,
then points your default browser at it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(os.environ.get("CAPYBARA_HOME", str(Path.home() / ".capybara")))
for cand in (Path(__file__).resolve().parent, HOME):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

try:
    import capybara as cb
except ImportError:
    print("error: capybara.py not found next to gui.py or in ~/.capybara",
          file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    """Start Capybara if needed and open the chat UI."""
    cb.open_ui(cb.load_settings(HOME))


if __name__ == "__main__":
    main()
