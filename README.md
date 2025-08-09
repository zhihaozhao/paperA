# Route-A Pipeline (Quick Start)
- conda env create -f env.yml && conda activate csi-fall-route-a
- python -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # 根据GPU情况调整
- bash scripts/make_all.sh
Outputs:
- plots/*.pdf
- tables/*.tex
- paper/main.pdf (compiled manuscript)