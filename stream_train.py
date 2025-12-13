#!/usr/bin/env python3
"""
Thin wrapper to keep your existing CLI usage:

  python3 stream_train.py ...

delegates to genostream.cli.main().
"""

from genostream.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
