# Default paths (can be overridden on command line)
CKPT ?= results/ckpt/best_enhanced_mid_s0.pt
OUTJSON ?= results/synth/out_mid.json
NPZ ?= results/preds/enhanced_mid_s0.npz
PLOT ?= results/plots/reliability_enhanced_mid.png
N_BINS ?= 15
SPLIT ?= te
CSV ?=
CSV_PER_CLASS ?=
DIFF ?= mid
SEED ?= 0
LAMBDAS ?= 0,0.02,0.05,0.08,0.12,0.18
JOBS ?= 1

.PHONY: infer reliability train all clean dirs lambda_sweep

dirs:
	mkdir -p results/preds results/plots results/synth results/ckpt

infer: dirs
	bash scripts/run_infer.sh "$(CKPT)" "$(OUTJSON)" "$(NPZ)" "$(SPLIT)"

reliability: dirs
	bash scripts/plot_reliability.sh "$(NPZ)" "$(N_BINS)" "$(PLOT)" "$(CSV)" "$(CSV_PER_CLASS)"

train: dirs
	bash scripts/run_train.sh

lambda_sweep:
	bash scripts/run_lambda_sweep.sh "$(DIFF)" "$(SEED)" "$(LAMBDAS)" "$(JOBS)"

all: infer reliability

clean:
	rm -f "$(NPZ)" "$(PLOT)"
	@echo "Cleaned: $(NPZ) $(PLOT)"