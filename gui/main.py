from __future__ import annotations

import argparse

from .app import PerceptromeGUI
from .settings import load_settings, save_settings
from .tk_compat import tk


def main() -> None:
    ap = argparse.ArgumentParser(description="Perceptrome Tk GUI")
    ap.add_argument("--config", default=None, help="Default config path")
    ap.add_argument("--geometry", default=None, help='Window geometry, e.g. "1020x780"')
    ap.add_argument("--theme", choices=["dark", "light"], default=None, help="Color theme")
    args = ap.parse_args()

    settings = load_settings()
    theme = args.theme or settings.get("theme", "dark")
    dark = (theme == "dark")

    root = tk.Tk()
    app = PerceptromeGUI(root, dark=dark, geometry=args.geometry or settings.get("geometry", "1020x780"))

    if args.config:
        app.config_path.set(args.config)
        app._persist_settings()

    settings["theme"] = theme
    save_settings(settings)

    root.mainloop()
