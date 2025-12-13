#!/usr/bin/env python3
"""
Thin wrapper to keep your existing CLI usage:

  perceptrome (or python -m perceptrome.cli_main) ...

delegates to perceptrome.cli_main.main().
"""

from perceptrome.cli_main import main

if __name__ == "__main__":
    raise SystemExit(main())
