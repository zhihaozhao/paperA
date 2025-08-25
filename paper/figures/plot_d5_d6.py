#!/usr/bin/env python3
from pathlib import Path
import sys

# Reuse the implementation from scripts/plot_d5_d6.py but run from here by default
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

try:
    import plot_d5_d6 as impl
except Exception as e:
    print("Failed to import scripts/plot_d5_d6.py:", e)
    sys.exit(1)

if __name__ == "__main__":
    impl.main()

