#!/usr/bin/env python3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper" / "PlotPy"))

import plot_d5_d6 as impl

if __name__ == "__main__":
    # Run generation to produce d5_d6_composite.pdf
    impl.main()
    # Alias to fig8 name for paper include
    figs = REPO / "paper" / "figures"
    src = figs / "d5_d6_composite.pdf"
    dst = figs / "fig8_d56_composite.pdf"
    try:
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print("✅ Saved alias:", dst)
    except Exception as err:
        print("[warn] failed to create fig8_d56_composite.pdf:", err)

