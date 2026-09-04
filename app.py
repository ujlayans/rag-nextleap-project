"""Legacy root entry point — delegates to the architecture's src/ui/app.py.

Prefer running the UI directly:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.app import main

main()
