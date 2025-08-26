#!/usr/bin/env python3
from pathlib import Path
import sys

# Reuse the implementation from paper/PlotPy/plot_d5_d6.py but run from here by default
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper" / "PlotPy"))

try:
    import plot_d5_d6 as impl
except Exception as e:
    print("Failed to import paper/PlotPy/plot_d5_d6.py:", e)
    sys.exit(1)

if __name__ == "__main__":
    # Run main to generate standard outputs
    impl.main()
    # Create the alias copy for normalized naming
    figs = REPO / "paper" / "figures"
    alias_src = figs / "d5_d6_results_slope.pdf"
    alias_dst = figs / "figureD56_slope.pdf"
    try:
        if alias_src.exists():
            alias_dst.write_bytes(alias_src.read_bytes())
            print("✅ Saved alias:", alias_dst)
    except Exception as err:
        print("[warn] failed to create alias figureD56_slope.pdf:", err)

