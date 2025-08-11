import argparse
import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def bin_stats(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, cnts = [], [], []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.any():
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
            cnts.append(int(m.sum()))
        else:
            accs.append(np.nan)
            confs.append((bins[i] + bins[i + 1]) / 2)
            cnts.append(0)
    return np.array(confs), np.array(accs), np.array(cnts)


def ece(conf, correct, n_bins=15):
    confs, accs, cnts = bin_stats(conf, correct, n_bins)
    w = cnts / max(cnts.sum(), 1)
    return float(np.nansum(w * np.abs(accs - confs)))


def nll_from_probs(probs, y):
    eps = 1e-12
    p = np.clip(probs[np.arange(len(y)), y], eps, 1 - eps)
    return float(-np.mean(np.log(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_npz", required=True)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--per_class", type=str, default="false")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    per_class = args.per_class.lower() in ("1", "true", "yes")
    dat = np.load(args.from_npz)
    logits_raw = dat["logits_raw"]
    logits_cal = dat["logits_cal"]
    labels = dat["labels"]
    preds = dat["preds"]

    probs_raw = softmax(logits_raw)
    probs_cal = softmax(logits_cal)
    conf_raw = probs_raw.max(axis=1)
    conf_cal = probs_cal.max(axis=1)
    correct = (preds == labels)

    ece_raw = ece(conf_raw, correct, n_bins=args.n_bins)
    ece_cal = ece(conf_cal, correct, n_bins=args.n_bins)
    nll_raw = nll_from_probs(probs_raw, labels)
    nll_cal = nll_from_probs(probs_cal, labels)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    for name, conf in [("raw", conf_raw), ("cal", conf_cal)]:
        bconf, bacc, _ = bin_stats(conf, correct, args.n_bins)
        lab = f"{name.upper()} ECE={ece_raw:.3f}" if name == "raw" else f"{name.upper()} ECE={ece_cal:.3f}"
        # 修正：避免 name.UPPER() 的拼写错误
    plt.clf()
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    bconf_r, bacc_r, _ = bin_stats(conf_raw, correct, args.n_bins)
    bconf_c, bacc_c, _ = bin_stats(conf_cal, correct, args.n_bins)
    plt.plot(bconf_r, bacc_r, "o-", label=f"RAW ECE={ece_raw:.3f}")
    plt.plot(bconf_c, bacc_c, "s-", label=f"CAL ECE={ece_cal:.3f}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"NLL raw={nll_raw:.3f} | cal={nll_cal:.3f}")
    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    print(f"Saved {args.save}")

    if per_class:
        K = probs_raw.shape[1]
        R = int(np.ceil(K / 2))
        fig, axes = plt.subplots(R, 2, figsize=(10, 4 * R))
        axes = axes.ravel()
        for k in range(K):
            mk = labels == k
            if not np.any(mk):
                continue
            conf_r = probs_raw[mk, k]
            conf_c = probs_cal[mk, k]
            corr_r = (np.argmax(probs_raw[mk], 1) == labels[mk])
            corr_c = (np.argmax(probs_cal[mk], 1) == labels[mk])
            ax = axes[k]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            if conf_r.size > 0:
                bc, ba, _ = bin_stats(conf_r, corr_r, args.n_bins)
                ax.plot(bc, ba, "o-", label="raw")
            if conf_c.size > 0:
                bc, ba, _ = bin_stats(conf_c, corr_c, args.n_bins)
                ax.plot(bc, ba, "s-", label="cal")
            ax.set_title(f"class {k}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.legend()
        plt.tight_layout()
        out2 = args.save.replace(".png", "_perclass.png")
        plt.savefig(out2, dpi=200)
        print(f"Saved {out2}")


if __name__ == "__main__":
    main()