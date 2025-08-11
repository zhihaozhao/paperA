import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path


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


def brier_score(probs, y):
    # Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2
    K = probs.shape[1]
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_npz", required=True)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--per_class", type=str, default="false")
    ap.add_argument("--save", required=True)
    ap.add_argument("--csv", type=str, default="", help="Optional CSV path to export overall binned reliability")
    ap.add_argument("--csv_per_class", type=str, default="", help="Optional CSV path to export per-class binned reliability")
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
    brier_raw = brier_score(probs_raw, labels)
    brier_cal = brier_score(probs_cal, labels)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    for name, conf in [("raw", conf_raw), ("cal", conf_cal)]:
        bconf, bacc, _ = bin_stats(conf, correct, args.n_bins)
        lab = f"{name.upper()} ECE={ece_raw:.3f}" if name == "raw" else f"{name.upper()} ECE={ece_cal:.3f}"
        # 前一版此处只是为了示意，实际绘图在下方汇总
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

    # Overall CSV export
    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        bconf_r_all, bacc_r_all, cnt_r = bin_stats(conf_raw, correct, args.n_bins)
        bconf_c_all, bacc_c_all, cnt_c = bin_stats(conf_cal, correct, args.n_bins)
        def to_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)
        bcr = to_list(bconf_r_all)
        bar = to_list(bacc_r_all)
        crr = to_list(cnt_r)
        bcc = to_list(bconf_c_all)
        bac = to_list(bacc_c_all)
        crc = to_list(cnt_c)
        max_len = max(len(bcr), len(bcc))
        def pad(arr, n):
            return arr + ["" for _ in range(n - len(arr))]
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "raw", "cal"])
            w.writerow(["ece", f"{ece_raw:.6f}", f"{ece_cal:.6f}"])
            w.writerow(["nll", f"{nll_raw:.6f}", f"{nll_cal:.6f}"])
            w.writerow(["brier", f"{brier_raw:.6f}", f"{brier_cal:.6f}"])
            w.writerow([])
            w.writerow(["bin_index", "raw_bin_conf", "raw_bin_acc", "raw_bin_count", "cal_bin_conf", "cal_bin_acc", "cal_bin_count"])
            BCR = pad(bcr, max_len)
            BAR = pad(bar, max_len)
            CRR = pad(crr, max_len)
            BCC = pad(bcc, max_len)
            BAC = pad(bac, max_len)
            CRC = pad(crc, max_len)
            for i in range(max_len):
                w.writerow([i + 1, BCR[i], BAR[i], CRR[i], BCC[i], BAC[i], CRC[i]])
        print(f"Exported CSV: {out_csv}")

    # Per-class CSV export
    if args.csv_per_class:
        out_csv_pc = Path(args.csv_per_class)
        out_csv_pc.parent.mkdir(parents=True, exist_ok=True)
        K = probs_raw.shape[1]
        with out_csv_pc.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "class_name", "brier_raw", "brier_cal", "bin_index",
                        "raw_bin_conf", "raw_bin_acc", "raw_bin_count",
                        "cal_bin_conf", "cal_bin_acc", "cal_bin_count"])
            for k in range(K):
                mk = (labels == k)
                if not np.any(mk):
                    continue
                # Per-class probabilities and correctness
                conf_r_k = probs_raw[mk, k]
                conf_c_k = probs_cal[mk, k]
                corr_r_k = (np.argmax(probs_raw[mk], axis=1) == labels[mk])
                corr_c_k = (np.argmax(probs_cal[mk], axis=1) == labels[mk])

                # Per-class brier: restrict to samples of class k
                brier_raw_k = brier_score(probs_raw[mk], labels[mk])
                brier_cal_k = brier_score(probs_cal[mk], labels[mk])

                bconf_r_k, bacc_r_k, cnt_r_k = bin_stats(conf_r_k, corr_r_k, args.n_bins)
                bconf_c_k, bacc_c_k, cnt_c_k = bin_stats(conf_c_k, corr_c_k, args.n_bins)
                max_len = max(len(bconf_r_k), len(bconf_c_k))

                def to_list(x):
                    return x.tolist() if hasattr(x, "tolist") else list(x)
                def pad(arr, n):
                    return arr + ["" for _ in range(n - len(arr))]
                BCR = pad(to_list(bconf_r_k), max_len)
                BAR = pad(to_list(bacc_r_k), max_len)
                CRR = pad(to_list(cnt_r_k),  max_len)
                BCC = pad(to_list(bconf_c_k), max_len)
                BAC = pad(to_list(bacc_c_k), max_len)
                CRC = pad(to_list(cnt_c_k),  max_len)

                for i in range(max_len):
                    w.writerow([k, "", f"{brier_raw_k:.6f}", f"{brier_cal_k:.6f}", i + 1,
                                BCR[i], BAR[i], CRR[i], BCC[i], BAC[i], CRC[i]])
        print(f"Exported per-class CSV: {out_csv_pc}")

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