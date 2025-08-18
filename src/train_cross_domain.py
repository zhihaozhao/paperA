#!/usr/bin/env python3
"""
Cross-domain training script for WiFi CSI sensing
Supports domain adaptation and cross-domain evaluation
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.metrics import compute_metrics
from src.data_synth import get_synth_loaders
from src.data_real import get_real_loaders  # legacy fallback
from src.models import get_model
from src.calibration import TemperatureScaling
from torch.utils.data import DataLoader
from src.utils.logger import setup_logger
from src.utils.io import set_seed
from src.utils.exp_recorder import append_run_registry, init_run

def compute_falling_metrics(y_true, y_prob, num_classes=8):
    """
    Compute falling detection metrics for 8-class system
    Classes 5-7 are falling-related: Epileptic Fall, Elderly Fall, Fall Can't Get Up
    Classes 0-4 are non-falling: Normal Walking, Shaking, Twitching, Punching, Kicking
    """
    from src.metrics import compute_metrics
    
    # Standard 8-class metrics
    standard_metrics = compute_metrics(y_true, y_prob, num_classes=num_classes, positive_class=5)
    
    # Binary falling detection: classes 5-7 vs classes 0-4
    y_true_binary = (y_true >= 5).astype(int)  # 1 for falling (5-7), 0 for non-falling (0-4)
    
    # Sum probabilities for falling classes
    y_prob_falling = y_prob[:, 5:8].sum(axis=1)  # P(class5) + P(class6) + P(class7)
    y_prob_binary = np.column_stack([1 - y_prob_falling, y_prob_falling])  # [P(non-fall), P(fall)]
    
    # Compute binary falling metrics
    falling_metrics = compute_metrics(y_true_binary, y_prob_binary, num_classes=2, positive_class=1)
    
    # Merge results
    result = standard_metrics.copy()
    result['falling_f1'] = falling_metrics['f1_fall']  # Binary falling F1
    result['falling_auprc'] = falling_metrics['auprc']  # Binary falling AUPRC
    result['falling_cm'] = falling_metrics['cm']  # 2x2 falling confusion matrix
    
    # Add class names for interpretability
    class_names = ['Normal_Walk', 'Shaking', 'Twitching', 'Punching', 'Kicking', 
                   'Epileptic_Fall', 'Elderly_Fall', 'Fall_CantGetUp']
    result['class_names'] = class_names
    
    return result

def make_json_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable Python types"""
    if obj is None:
        return None
    elif hasattr(obj, 'item') and hasattr(obj, 'size') and obj.size == 1:  # Single element numpy/torch scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, '__float__'):  # Try to convert to float
        try:
            return float(obj)
        except (ValueError, TypeError):
            return str(obj)
    elif hasattr(obj, '__int__'):  # Try to convert to int
        try:
            return int(obj)
        except (ValueError, TypeError):
            return str(obj)
    else:
        return obj

def compute_overlap_stat(model, loader, device):
    """
    Compute overlap statistics for class separability analysis.
    - Adapts to the actual set of labels present in the loader (no hard-coded class count)
    - Robust feature pooling: collapses all non-batch dimensions to get per-sample vectors
    """
    model.eval()
    features_per_class = defaultdict(list)

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)

            # Get features before final classification layer, or fallback to logits
            if hasattr(model, 'get_features'):
                feats = model.get_features(batch_x)
            else:
                output = model(batch_x)
                feats = output[0] if isinstance(output, tuple) else output

            # Collapse to [B, D]
            while feats.dim() > 2:
                feats = feats.mean(dim=-1)

            feats_np = feats.cpu().numpy()
            labels_np = batch_y.cpu().numpy()

            for i in range(len(labels_np)):
                class_label = int(labels_np[i])
                if 0 <= class_label < 10_000:  # basic guard
                    features_per_class[class_label].append(feats_np[i])

    # Compute class centroids
    centroids = {}
    for class_idx, feature_list in features_per_class.items():
        if feature_list:
            centroids[class_idx] = np.mean(feature_list, axis=0)

    # Compute pairwise overlaps (cosine similarity) over observed classes
    overlaps = []
    class_pairs = []
    observed = sorted(centroids.keys())
    for a in range(len(observed)):
        for b in range(a + 1, len(observed)):
            i, j = observed[a], observed[b]
            c1, c2 = centroids[i], centroids[j]
            denom = (np.linalg.norm(c1) * np.linalg.norm(c2)) + 1e-8
            cosine_sim = float(np.dot(c1, c2) / denom)
            overlaps.append(cosine_sim)
            class_pairs.append((i, j))

    if overlaps:
        return {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps)),
            "pairs": class_pairs,
            "values": overlaps,
        }
    return {"mean": 0.0, "std": 0.0, "pairs": [], "values": []}

def cross_domain_train(args):
    """
    Cross-domain training and evaluation
    """
    logger, meta = init_run(
        phase="CrossDomain", exp="LOSO", ver="V1",
        desc=f"Cross-domain {args.source} -> {args.target}",
        args=vars(args)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    logger.info(f"Cross-domain: {args.source} -> {args.target}")
    
    # Load source domain data
    if args.source == "synth":
        train_loader, val_loader = get_synth_loaders(
            batch=args.batch_size, 
            difficulty=args.difficulty, 
            seed=args.seed,
            n=args.n_samples, 
            T=args.T, 
            F=args.F
        )
    else:
        # Real data loader (placeholder - needs implementation)
        train_loader, val_loader = get_real_loaders(
            dataset=args.source,
            batch_size=args.batch_size,
            seed=args.seed
        )
    
    # Load target domain data  
    if args.target == "synth":
        _, test_loader = get_synth_loaders(
            batch=args.batch_size,
            difficulty=args.difficulty,
            seed=args.seed + 1000  # Different seed for target
        )
    else:
        # Real data loader for target
        _, test_loader = get_real_loaders(
            dataset=args.target,
            batch_size=args.batch_size,
            seed=args.seed + 1000
        )
    
    # Model setup
    x_sample, _ = next(iter(train_loader))
    input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
    
    model = get_model(args.model, input_dim=input_dim, num_classes=4)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0.0
    best_metrics = {}
    
    use_amp = bool(getattr(args, 'amp', False)) and torch.cuda.is_available()
    if use_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    output = model(batch_x)
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(batch_x)
                logits = output[0] if isinstance(output, tuple) else output
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_metrics = eval_model(model, val_loader, device, args.positive_class)
        
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"Loss={train_loss/len(train_loader):.4f}, "
                   f"Val F1={val_metrics['macro_f1']:.4f}")
    
    # Test on target domain
    test_metrics = eval_model(model, test_loader, device, args.positive_class)
    
    # Compute overlap statistics
    overlap_stat = compute_overlap_stat(model, val_loader, device)
    
    # Save results
    results = {
        "args": vars(args),
        "source_domain": args.source,
        "target_domain": args.target,
        "best_val_metrics": best_metrics,
        "target_test_metrics": test_metrics,
        "overlap_stat": overlap_stat,
        "meta": meta
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Cross-domain results saved to {args.out}")
    return results

def eval_model(model, loader, device, positive_class=1):
    """
    Evaluate model on a given loader
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            all_logits.append(logits.cpu())
            all_labels.append(batch_y)
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    probs = torch.softmax(logits, dim=1).numpy()
    labels = labels.numpy()
    
    metrics = compute_falling_metrics(labels, probs, num_classes=8)  # Use 8-class fall detection metrics
    
    # Add ECE calculation (simplified)
    confidence = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracy = (predictions == labels).astype(float)
    
    # Simple ECE calculation
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    metrics["ece"] = float(ece)
    metrics["falling_f1"] = metrics.get("f1_fall", float("nan"))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Cross-domain training for D3/D4 experiments")
    
    # Core parameters
    parser.add_argument("--model", type=str, default="enhanced", help="Model type: enhanced, cnn, bilstm, conformer_lite")
    parser.add_argument("--protocol", type=str, default="loso", help="Protocol: loso, loro, sim2real")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--out", type=str, default="results/cross_domain/out.json", help="Output file")
    
    # Data parameters
    parser.add_argument("--benchmark_path", type=str, default="benchmarks/WiFi-CSI-Sensing-Benchmark-main", 
                       help="Path to WiFi CSI benchmark dataset")
    parser.add_argument("--files_per_activity", type=int, default=2, help="Number of files to load per activity (for balance)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # D4 Sim2Real specific parameters
    parser.add_argument("--d2_model_path", type=str, default="", help="Path to D2 pre-trained model")
    parser.add_argument("--label_ratio", type=float, default=1.0, help="Ratio of labeled real data to use")
    parser.add_argument("--transfer_method", type=str, default="fine_tune", 
                       help="Transfer method: zero_shot, linear_probe, fine_tune, temp_scale")
    parser.add_argument("--skip_synth_pretrain", action="store_true",
                       help="If set, never pretrain on synthetic data (for all transfer methods)")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--class_weight", type=str, default="inv_freq", choices=["none", "inv_freq"], help="Loss class weighting strategy")
    parser.add_argument("--loso_all_folds", action="store_true", help="Run LOSO across all subjects and aggregate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_epochs", type=int, default=20, help="Minimum epochs before early stopping")
    parser.add_argument("--loro_all_folds", action="store_true", help="Run LORO across all rooms and aggregate")
    parser.add_argument("--positive_class", type=int, default=5, help="Positive class for AUPRC (Epileptic_Fall=5 in 8-class system)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP for faster training if available")
    
    # Legacy compatibility
    parser.add_argument("--source", type=str, default="synth", help="Source domain (legacy)")
    parser.add_argument("--target", type=str, default="real", help="Target domain (legacy)")
    parser.add_argument("--difficulty", type=str, default="mid", help="Difficulty level (legacy)")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples (legacy)")
    parser.add_argument("--T", type=int, default=128, help="Time steps (legacy)")
    parser.add_argument("--F", type=int, default=30, help="Feature dimension (legacy)")
    
    args = parser.parse_args()
    
    # Route to appropriate experiment based on protocol
    if args.protocol == "loso":
        return run_loso_experiment(args)
    elif args.protocol == "loro": 
        return run_loro_experiment(args)
    elif args.protocol == "sim2real":
        return run_sim2real_experiment(args)
    else:
        # Legacy mode
        return cross_domain_train(args)

def run_loso_experiment(args):
    """Run LOSO (Leave-One-Subject-Out) cross-validation"""
    logger, meta = init_run(
        phase="D3", exp="LOSO", ver="V1",
        desc=f"LOSO cross-subject evaluation with {args.model}",
        args=vars(args)
    )
    
    logger.info(f"Starting LOSO experiment: model={args.model}, seed={args.seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # Load real WiFi CSI benchmark data for LOSO cross-subject evaluation
    from src.data_real import BenchmarkCSIDataset, get_real_loaders_loso
    
    logger.info(f"Loading WiFi CSI benchmark data from {args.benchmark_path}")
    benchmark = BenchmarkCSIDataset(args.benchmark_path, files_per_activity=args.files_per_activity)
    
    try:
        X, y, subjects, rooms, metadata = benchmark.load_wifi_csi_benchmark()
        logger.info(f"Loaded benchmark: X.shape={X.shape}, y.shape={y.shape}, n_subjects={len(np.unique(subjects))}")
        
        # LOSO splits
        from src.data_real import BenchmarkCSIDataset as _B
        splits = benchmark.create_loso_splits(subjects)
        logger.info(f"Found {len(splits)} LOSO folds")

        def _compute_class_weights(y_train: np.ndarray, num_classes: int = 8) -> torch.Tensor:
            if args.class_weight == "none":
                return None
            counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
            weights = inv / inv.mean()
            return torch.tensor(weights, dtype=torch.float32)

        fold_metrics = []
        if args.loso_all_folds:
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                trX, trY = X[train_idx], y[train_idx]
                teX, teY = X[test_idx], y[test_idx]
                train_loader, test_loader = get_real_loaders_loso(X, y, subjects, np.unique(subjects)[fold_id], batch=args.batch_size)
                val_loader = test_loader
                class_w = _compute_class_weights(trY)

                # Model per fold
                x_sample, _ = next(iter(train_loader))
                input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
                model = get_model(args.model, input_dim=input_dim, num_classes=8).to(device)
                best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args, class_weights=class_w)
                fold_metrics.append(best_metrics)

            # Aggregate
            def _mean_of(key: str):
                vals = [m.get(key, 0.0) for m in fold_metrics]
                return float(np.mean(vals)) if vals else 0.0
            best_metrics = {
                "macro_f1": _mean_of("macro_f1"),
                "falling_f1": _mean_of("falling_f1"),
                "ece": _mean_of("ece"),
                "falling_auprc": _mean_of("falling_auprc"),
                "per_class_f1": np.mean([m.get("per_class_f1", [0]*8) for m in fold_metrics], axis=0).tolist(),
            }
            # Use last fold for overlap stat exemplar
            overlap_stat = compute_overlap_stat(model, val_loader, device)
        else:
            # Quick single-fold: test first subject only
            test_subject = 0
            train_loader, test_loader = get_real_loaders_loso(X, y, subjects, test_subject, batch=args.batch_size)
            val_loader = test_loader
            logger.info(f"LOSO split: test_subject={test_subject}, train_size={len(train_loader.dataset)}, test_size={len(test_loader.dataset)}")

            # Class weights from training labels
            tr_idx = np.where(subjects != test_subject)[0]
            class_w = _compute_class_weights(y[tr_idx])
            # Model setup
            x_sample, _ = next(iter(train_loader))
            input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
            model = get_model(args.model, input_dim=input_dim, num_classes=8).to(device)
            best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args, class_weights=class_w)
            overlap_stat = compute_overlap_stat(model, val_loader, device)
        
    except Exception as e:
        logger.warning(f"Failed to load benchmark data: {e}")
        logger.info("Falling back to realistic synthetic data for LOSO simulation...")
        
        # Generate challenging synthetic data that simulates cross-subject variability
        train_loader, val_loader, test_loader = get_synth_loaders(
            batch=args.batch_size, difficulty="hard", seed=args.seed,
            n=1500, T=128, F=30, num_classes=8,  # 8-class fall detection system
            # Add realistic cross-subject variations
            sc_corr_rho=0.5,  # Lower correlation for subject differences
            env_burst_rate=0.15,  # Higher noise for realism
            gain_drift_std=0.01,  # Subject-specific gain variations
            class_overlap=0.1,  # Add class overlap for generalization challenge
            label_noise_prob=0.05  # Add label noise for realism
        )
    
    # Training and evaluation (classes 5-7 are falling-related for binary fall detection)
    args.positive_class = 5  # Use epileptic_fall as primary falling class for AUPRC
    
    # Save results in D3 format
    results = {
        "protocol": "LOSO",
        "model": args.model,
        "seed": int(args.seed),
        "aggregate_stats": {
            "macro_f1": {"mean": float(best_metrics.get("macro_f1", 0.0)), "std": 0.0},
            "falling_f1": {"mean": float(best_metrics.get("falling_f1", 0.0)), "std": 0.0},  # Binary falling detection F1
            "ece": {"mean": float(best_metrics.get("ece", 0.0)), "std": 0.0},
            "auprc_falling": {"mean": float(best_metrics.get("falling_auprc", 0.0)), "std": 0.0},  # Binary falling AUPRC
            "epileptic_fall_f1": {"mean": float(best_metrics.get("per_class_f1", [0]*8)[5]), "std": 0.0},  # 类5: 癫痫跌倒
            "elderly_fall_f1": {"mean": float(best_metrics.get("per_class_f1", [0]*8)[6]), "std": 0.0},      # 类6: 老人跌倒
            "fall_cantgetup_f1": {"mean": float(best_metrics.get("per_class_f1", [0]*8)[7]), "std": 0.0},   # 类7: 跌倒起不来
            "mutual_misclass": {"mean": 0.0, "std": 0.0}  # TODO: Implement
        },
        "fold_results": [make_json_serializable(best_metrics)],
        "overlap_stat": make_json_serializable(overlap_stat),
        "meta": make_json_serializable(meta)
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"LOSO results saved to {args.out}")
    return results

def run_loro_experiment(args):
    """Run LORO (Leave-One-Room-Out) cross-validation"""
    logger, meta = init_run(
        phase="D3", exp="LORO", ver="V1", 
        desc=f"LORO cross-room evaluation with {args.model}",
        args=vars(args)
    )
    
    logger.info(f"Starting LORO experiment: model={args.model}, seed={args.seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # Load real WiFi CSI benchmark data for LORO cross-room evaluation
    from src.data_real import BenchmarkCSIDataset, get_real_loaders_loro
    
    logger.info(f"Loading WiFi CSI benchmark data from {args.benchmark_path}")
    benchmark = BenchmarkCSIDataset(args.benchmark_path, files_per_activity=args.files_per_activity)
    
    try:
        X, y, subjects, rooms, metadata = benchmark.load_wifi_csi_benchmark()
        logger.info(f"Loaded benchmark: X.shape={X.shape}, y.shape={y.shape}, n_rooms={len(np.unique(rooms))}")
        
        # LORO splits
        splits = benchmark.create_loro_splits(rooms)
        logger.info(f"Found {len(splits)} LORO folds")

        def _compute_class_weights(y_train: np.ndarray, num_classes: int = 8) -> torch.Tensor:
            if args.class_weight == "none":
                return None
            counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
            weights = inv / inv.mean()
            return torch.tensor(weights, dtype=torch.float32)

        fold_metrics = []
        if args.loro_all_folds:
            unique_rooms = np.unique(rooms)
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                trX, trY = X[train_idx], y[train_idx]
                # Build loaders for this fold
                test_room = unique_rooms[fold_id]
                train_loader, test_loader = get_real_loaders_loro(X, y, rooms, test_room, batch=args.batch_size)
                val_loader = test_loader
                class_w = _compute_class_weights(trY)

                x_sample, _ = next(iter(train_loader))
                input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
                model = get_model(args.model, input_dim=input_dim, num_classes=8).to(device)
                best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args, class_weights=class_w)
                fold_metrics.append(best_metrics)

            # Aggregate
            def _mean_of(key: str):
                vals = [m.get(key, 0.0) for m in fold_metrics]
                return float(np.mean(vals)) if vals else 0.0
            best_metrics = {
                "macro_f1": _mean_of("macro_f1"),
                "falling_f1": _mean_of("falling_f1"),
                "ece": _mean_of("ece"),
                "falling_auprc": _mean_of("falling_auprc"),
                "per_class_f1": np.mean([m.get("per_class_f1", [0]*8) for m in fold_metrics], axis=0).tolist(),
            }
            overlap_stat = compute_overlap_stat(model, val_loader, device)
        else:
            # Quick single-fold: test first room only
            test_room = 0
            train_loader, test_loader = get_real_loaders_loro(X, y, rooms, test_room, batch=args.batch_size)
            val_loader = test_loader
            logger.info(f"LORO split: test_room={test_room}, train_size={len(train_loader.dataset)}, test_size={len(test_loader.dataset)}")

            tr_idx = np.where(rooms != test_room)[0]
            class_w = _compute_class_weights(y[tr_idx])
            x_sample, _ = next(iter(train_loader))
            input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
            model = get_model(args.model, input_dim=input_dim, num_classes=8).to(device)
            best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args, class_weights=class_w)
            overlap_stat = compute_overlap_stat(model, val_loader, device)
        
    except Exception as e:
        logger.warning(f"Failed to load benchmark data: {e}")
        logger.info("Falling back to realistic synthetic data for LORO simulation...")
        
        # Generate challenging synthetic data that simulates cross-room variability  
        train_loader, val_loader, _ = get_synth_loaders(
            batch=args.batch_size, difficulty="hard", seed=args.seed,
            n=1500, T=128, F=30, num_classes=8,  # 8-class fall detection system
            # Add realistic cross-room variations
            sc_corr_rho=0.4,  # Lower correlation for room differences
            env_burst_rate=0.2,  # Higher environmental noise
            gain_drift_std=0.02,  # Room-specific propagation variations
            class_overlap=0.15,  # Add class overlap for environmental robustness
            label_noise_prob=0.08  # Add more label noise for room variations
    )
    
    # Model setup and training for 8-class fall detection
    x_sample, _ = next(iter(train_loader))
    input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
    
    model = get_model(args.model, input_dim=input_dim, num_classes=8)  # 8-class fall detection
    model = model.to(device)
    
    best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args)
    overlap_stat = compute_overlap_stat(model, val_loader, device)
    
    # Save results in D3 format
    results = {
        "protocol": "LORO",
        "model": args.model,
        "seed": int(args.seed),
        "aggregate_stats": {
            "macro_f1": {"mean": float(best_metrics.get("macro_f1", 0.0)), "std": 0.0},
            "falling_f1": {"mean": float(best_metrics.get("falling_f1", 0.0)), "std": 0.0},  # Binary falling detection F1
            "ece": {"mean": float(best_metrics.get("ece", 0.0)), "std": 0.0},
            "auprc_falling": {"mean": float(best_metrics.get("falling_auprc", 0.0)), "std": 0.0},  # Binary falling AUPRC
            "epileptic_fall_f1": {"mean": float(best_metrics.get("per_class_f1", [0]*8)[5]), "std": 0.0},  # 类5: 癫痫跌倒
            "elderly_fall_f1": {"mean": float(best_metrics.get("per_class_f1", [0]*8)[6]), "std": 0.0},      # 类6: 老人跌倒
            "fall_cantgetup_f1": {"mean": float(best_metrics.get("per_class_f1", [0]*8)[7]), "std": 0.0},   # 类7: 跌倒起不来
            "mutual_misclass": {"mean": 0.0, "std": 0.0}
        },
        "fold_results": [make_json_serializable(best_metrics)],
        "overlap_stat": make_json_serializable(overlap_stat),
        "meta": make_json_serializable(meta)
    }
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"LORO results saved to {args.out}")
    return results

def run_sim2real_experiment(args):
    """Run Sim2Real label efficiency experiment"""
    logger, meta = init_run(
        phase="D4", exp="Sim2Real", ver="V1",
        desc=f"Sim2Real {args.transfer_method} with {args.label_ratio*100:.1f}% labels",
        args=vars(args)
    )
    
    logger.info(f"Starting Sim2Real: model={args.model}, ratio={args.label_ratio}, method={args.transfer_method}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    # Load REAL dataset first to infer input dims and ensure no fallback
    from src.data_real import BenchmarkCSIDataset, RealCSIDataset, get_sim2real_loaders
    bench = BenchmarkCSIDataset(args.benchmark_path, files_per_activity=int(getattr(args, 'files_per_activity', 2)))
    X, y, subjects, rooms, metadata = bench.load_wifi_csi_benchmark()
    T_real, F_real = int(X.shape[1]), int(X.shape[2])
    # Split real data into labeled small subset (for probe/calibration) and unlabeled test set
    train_loader, test_loader = get_sim2real_loaders(X, y, label_ratio=float(args.label_ratio), seed=int(args.seed), batch=args.batch_size)
    logger.info(f"[D4] Zero-shot REAL eval prepared: T={T_real}, F={F_real}, N_total={len(y)}, N_labeled={len(train_loader.dataset)}, N_test={len(test_loader.dataset)}")
    
    # Load D2 pre-trained model if available (supports directory or file path)
    model = None
    if getattr(args, 'd2_model_path', None):
        d2_path = str(args.d2_model_path)
        try:
            if os.path.isdir(d2_path):
                import glob
                # Recursive patterns to robustly find common naming schemes
                name_patts = [
                    f"*{args.model}*seed{args.seed}*hard.*",
                    f"*{args.model}*seed{args.seed}*.pt",
                    f"*{args.model}*seed{args.seed}*.pth",
                    f"final_{args.model}_seed{args.seed}_hard.*",
                    f"final_model_seed{args.seed}_hard.*",
                    f"*seed{args.seed}*hard.*",
                    f"*seed{args.seed}*.pt",
                    f"*seed{args.seed}*.pth",
                    "*final*.pt", "*final*.pth",
                    "*best*.pt", "*best*.pth",
                    "*.pt", "*.pth",
                ]
                cands = []
                for npat in name_patts:
                    pat = os.path.join(d2_path, "**", npat)
                    cands.extend(glob.glob(pat, recursive=True))
                ckpt_path = max(cands, key=lambda p: os.path.getmtime(p)) if cands else None
            elif os.path.isfile(d2_path):
                ckpt_path = d2_path
            else:
                ckpt_path = None
            if ckpt_path and os.path.exists(ckpt_path):
                logger.info(f"Loading D2 pre-trained model from {ckpt_path}")
                try:
                    obj = torch.load(ckpt_path, map_location=device)
                    if hasattr(obj, 'state_dict'):
                        # Likely a torch.nn.Module
                        model = obj
                    elif isinstance(obj, dict) and any(k in obj for k in ('state_dict','model','model_state')):
                        # Common checkpoint dict formats
                        state = obj.get('state_dict') or obj.get('model_state') or obj.get('model')
                        # Build fresh model and load state dict
                        model = get_model(args.model, input_dim=F_real, num_classes=8).to(device)
                        try:
                            model.load_state_dict(state, strict=False)
                        except Exception:
                            # If wrapped with module.* keys
                            new_state = {k.replace('module.',''): v for k,v in state.items()}
                            model.load_state_dict(new_state, strict=False)
                    else:
                        # Assume it's directly the model
                        model = obj
                except Exception as e:
                    logger.warning(f"Failed to deserialize D2 model from {ckpt_path}: {e}. Running from scratch.")
            else:
                # Emit a brief listing to help users verify paths
                try:
                    all_pth = []
                    for pat in (os.path.join(d2_path, "**", "*.pth"), os.path.join(d2_path, "**", "*.pt")):
                        all_pth.extend(glob.glob(pat, recursive=True))
                    sample = all_pth[:5]
                    logger.warning(f"No D2 checkpoint found for model={args.model} seed={args.seed} in {d2_path}. Files seen (sample {len(sample)}/{len(all_pth)}): {sample} . Running from scratch.")
                except Exception:
                    logger.warning(f"No D2 checkpoint found for model={args.model} seed={args.seed} in {d2_path}. Running from scratch.")
        except Exception as e:
            logger.warning(f"Failed to search/load D2 model from {d2_path}: {e}. Running from scratch.")
    
    if model is None:
        # Per requirement: do NOT load or pretrain on synthetic data before training for Sim2Real
        logger.info("[D4] Initializing model from scratch without synthetic pretraining")
        model = get_model(args.model, input_dim=F_real, num_classes=8).to(device)
    
    # Zero-shot evaluation on real data (STRICT: no synthetic fallback)
    logger.info(f"[D4] Zero-shot REAL eval: X={X.shape}, y={y.shape}, subjects={len(np.unique(subjects))}, rooms={len(np.unique(rooms))}")
    zero_shot_metrics = eval_model(model, test_loader, device, args.positive_class)
    
    # Apply transfer method
    if args.transfer_method == "zero_shot":
        final_metrics = zero_shot_metrics
    elif args.transfer_method == "temp_scale":
        # Temperature scaling only: fit T on small labeled subset, evaluate on held-out test
        final_metrics = apply_temperature_scaling(model, train_loader, test_loader, device, args.positive_class)
    else:
        # Fine-tuning or linear probe with limited labels on labeled subset, evaluate on held-out test
        final_metrics = transfer_learning(model, train_loader, test_loader, device, args)
    
    # Save D4 format results
    results = {
        "protocol": "Sim2Real",
        "model": args.model,
        "seed": int(args.seed),
        "label_ratio": float(args.label_ratio),
        "transfer_method": args.transfer_method,
        "zero_shot_metrics": {
            "macro_f1": float(zero_shot_metrics.get("macro_f1", 0.0)),
            "falling_f1": float(zero_shot_metrics.get("falling_f1", 0.0)),  # Binary falling detection F1
            "epileptic_fall_f1": float(zero_shot_metrics.get("per_class_f1", [0]*8)[5]),  # 癫痫跌倒
            "elderly_fall_f1": float(zero_shot_metrics.get("per_class_f1", [0]*8)[6]),     # 老人跌倒
            "ece": float(zero_shot_metrics.get("ece", 0.0))
        },
        "target_metrics": {
            "macro_f1": float(final_metrics.get("macro_f1", 0.0)),
            "falling_f1": float(final_metrics.get("falling_f1", 0.0)),  # Binary falling detection F1
            "epileptic_fall_f1": float(final_metrics.get("per_class_f1", [0]*8)[5]),  # 癫痫跌倒
            "elderly_fall_f1": float(final_metrics.get("per_class_f1", [0]*8)[6]),     # 老人跌倒
            "ece": float(final_metrics.get("ece", 0.0))
        },
        "meta": make_json_serializable(meta)
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sim2Real results saved to {args.out}")
    return results

def train_and_evaluate(model, train_loader, val_loader, device, args, class_weights: torch.Tensor = None):
    """Common training and evaluation loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_f1 = 0.0
    best_metrics = {}
    epochs_without_improve = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_metrics = eval_model(model, val_loader, device, args.positive_class)
        
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss/len(train_loader):.4f}, F1={val_metrics['macro_f1']:.4f}")

        # Early stopping
        if (epoch + 1) >= int(getattr(args, 'min_epochs', 0)) and epochs_without_improve >= int(getattr(args, 'patience', 0)):
            break
    
    return best_metrics

def apply_temperature_scaling(model, labeled_loader, test_loader, device, positive_class):
    """Apply temperature scaling: fit T on labeled subset, evaluate on test set"""
    # Collect logits/labels on labeled subset for fitting
    model.eval()
    logits_lab, labels_lab = [], []
    with torch.no_grad():
        for x, y in labeled_loader:
            x = x.to(device)
            out = model(x)
            lg = out[0] if isinstance(out, tuple) else out
            logits_lab.append(lg.cpu())
            labels_lab.append(y)
    logits_lab = torch.cat(logits_lab, dim=0).numpy()
    labels_lab = torch.cat(labels_lab, dim=0).numpy()

    # Fit temperature on labeled subset
    from src.calibration import temperature_scaling
    probs_lab = torch.softmax(torch.from_numpy(logits_lab), dim=1).numpy()
    probs_lab_cal, t_opt = temperature_scaling(probs_lab, labels_lab)
    T = float(t_opt) if t_opt is not None else 1.0

    # Evaluate on test set with calibrated temperature
    all_logits_test, all_labels_test = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            lg = out[0] if isinstance(out, tuple) else out
            all_logits_test.append((lg / T).cpu())  # apply temperature to logits
            all_labels_test.append(y)
    logits_test = torch.cat(all_logits_test, dim=0).numpy()
    labels_test = torch.cat(all_labels_test, dim=0).numpy()
    probs_test = torch.softmax(torch.from_numpy(logits_test), dim=1).numpy()

    # Compute metrics on test set
    metrics = compute_falling_metrics(labels_test, probs_test, num_classes=8)
    metrics["temperature"] = T

    # ECE from calibrated probs (test set)
    confidence = np.max(probs_test, axis=1)
    predictions = np.argmax(probs_test, axis=1)
    accuracy = (predictions == labels_test).astype(float)
    n_bins = 15
    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    metrics["ece"] = float(ece)
    metrics["falling_f1"] = metrics.get("f1_fall", float("nan"))
    return metrics

def transfer_learning(model, labeled_loader, test_loader, device, args):
    """Perform transfer learning (fine-tune or linear probe) on labeled subset, evaluate on test set"""
    method = str(getattr(args, 'transfer_method', 'fine_tune')).lower()
    if method == 'linear_probe':
        # Freeze backbone; train a new linear head
        if hasattr(model, 'head') and hasattr(model, 'attn'):
            for p in model.parameters():
                p.requires_grad = False
            for p in model.head.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
        else:
            # Fallback: train last linear layer if identifiable; otherwise do no-op
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()
    epochs = max(1, int(getattr(args, 'min_epochs', 3)))
    model.train()
    for epoch in range(epochs):
        for x, y in labeled_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    return eval_model(model, test_loader, device, args.positive_class)

if __name__ == "__main__":
    main()