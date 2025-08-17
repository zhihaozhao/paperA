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

from src.metrics import compute_metrics
from src.data_synth import get_synth_loaders
from src.data_real import get_real_loaders  # Assuming this exists or will be created
from src.models import get_model
from src.calibration import TemperatureScaling
from src.utils.logger import setup_logger
from src.utils.io import set_seed
from src.utils.exp_recorder import append_run_registry, init_run

def compute_overlap_stat(model, loader, device):
    """
    Compute overlap statistics for class separability analysis
    """
    model.eval()
    features_per_class = {i: [] for i in range(4)}
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            
            # Get features before final classification layer
            if hasattr(model, 'get_features'):
                features = model.get_features(batch_x)
            else:
                # Fallback: use penultimate layer if available
                output = model(batch_x)
                features = output[0] if isinstance(output, tuple) else output
                if len(features.shape) > 2:
                    features = features.mean(dim=-1)  # Global average pooling
            
            features = features.cpu().numpy()
            batch_y = batch_y.numpy()
            
            for i in range(len(batch_y)):
                class_label = int(batch_y[i])
                features_per_class[class_label].append(features[i])
    
    # Compute class centroids
    centroids = {}
    for class_idx, feature_list in features_per_class.items():
        if feature_list:
            centroids[class_idx] = np.mean(feature_list, axis=0)
    
    # Compute pairwise overlaps (cosine similarity)
    overlaps = []
    class_pairs = []
    for i in range(4):
        for j in range(i+1, 4):
            if i in centroids and j in centroids:
                c1, c2 = centroids[i], centroids[j]
                cosine_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                overlaps.append(cosine_sim)
                class_pairs.append((i, j))
    
    if overlaps:
        return {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps)),
            "pairs": class_pairs,
            "values": overlaps
        }
    else:
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
    
    for epoch in range(args.epochs):
        # Train
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
    
    metrics = compute_metrics(labels, probs, num_classes=4, positive_class=positive_class)
    
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
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # D4 Sim2Real specific parameters
    parser.add_argument("--d2_model_path", type=str, default="", help="Path to D2 pre-trained model")
    parser.add_argument("--label_ratio", type=float, default=1.0, help="Ratio of labeled real data to use")
    parser.add_argument("--transfer_method", type=str, default="fine_tune", 
                       help="Transfer method: zero_shot, linear_probe, fine_tune, temp_scale")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--positive_class", type=int, default=3, help="Positive class for AUPRC (Falling=3)")
    
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
    
    # For now, use synthetic data as placeholder until benchmark integration is complete
    # TODO: Replace with actual benchmark data loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    train_loader, val_loader, test_loader = get_synth_loaders(
        batch=args.batch_size, difficulty="mid", seed=args.seed,
        n=1500, T=128, F=30
    )
    
    # Model setup
    x_sample, _ = next(iter(train_loader))
    input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
    
    model = get_model(args.model, input_dim=input_dim, num_classes=4)
    model = model.to(device)
    
    # Training and evaluation
    best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args)
    
    # Compute overlap statistics
    overlap_stat = compute_overlap_stat(model, val_loader, device)
    
    # Save results in D3 format
    results = {
        "protocol": "LOSO",
        "model": args.model,
        "seed": args.seed,
        "aggregate_stats": {
            "macro_f1": {"mean": best_metrics.get("macro_f1", 0.0), "std": 0.0},
            "falling_f1": {"mean": best_metrics.get("f1_fall", 0.0), "std": 0.0},
            "ece": {"mean": best_metrics.get("ece", 0.0), "std": 0.0},
            "auprc_falling": {"mean": best_metrics.get("auprc", 0.0), "std": 0.0},
            "mutual_misclass": {"mean": 0.0, "std": 0.0}  # TODO: Implement
        },
        "fold_results": [best_metrics],  # Single fold for now
        "overlap_stat": overlap_stat,
        "meta": meta
    }
    
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
    
    # Similar to LOSO but with room-based splits
    # For now, use synthetic data with perturbations to simulate room differences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    train_loader, val_loader, _ = get_synth_loaders(
        batch=args.batch_size, difficulty="hard", seed=args.seed,
        n=1500, T=128, F=30,
        sc_corr_rho=0.7, env_burst_rate=0.05, gain_drift_std=0.003
    )
    
    # Model setup and training
    x_sample, _ = next(iter(train_loader))
    input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
    
    model = get_model(args.model, input_dim=input_dim, num_classes=4)
    model = model.to(device)
    
    best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args)
    overlap_stat = compute_overlap_stat(model, val_loader, device)
    
    # Save results in D3 format
    results = {
        "protocol": "LORO",
        "model": args.model,
        "seed": args.seed,
        "aggregate_stats": {
            "macro_f1": {"mean": best_metrics.get("macro_f1", 0.0), "std": 0.0},
            "falling_f1": {"mean": best_metrics.get("f1_fall", 0.0), "std": 0.0},
            "ece": {"mean": best_metrics.get("ece", 0.0), "std": 0.0},
            "auprc_falling": {"mean": best_metrics.get("auprc", 0.0), "std": 0.0},
            "mutual_misclass": {"mean": 0.0, "std": 0.0}
        },
        "fold_results": [best_metrics],
        "overlap_stat": overlap_stat,
        "meta": meta
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
    
    # Load D2 pre-trained model if available
    model = None
    if args.d2_model_path and os.path.exists(args.d2_model_path):
        try:
            logger.info(f"Loading D2 pre-trained model from {args.d2_model_path}")
            model = torch.load(args.d2_model_path, map_location=device)
        except Exception as e:
            logger.warning(f"Failed to load D2 model: {e}, training from scratch")
    
    if model is None:
        # Train from scratch on synthetic data
        logger.info("Training model from scratch on synthetic data")
        train_loader, val_loader, _ = get_synth_loaders(
            batch=args.batch_size, difficulty="mid", seed=args.seed,
            n=2000, T=128, F=30
        )
        
        x_sample, _ = next(iter(train_loader))
        input_dim = x_sample.shape[-1] if x_sample.dim() == 3 else x_sample.shape[1]
        
        model = get_model(args.model, input_dim=input_dim, num_classes=4)
        model = model.to(device)
        
        # Pre-train on synthetic data
        best_metrics = train_and_evaluate(model, train_loader, val_loader, device, args)
        logger.info(f"Synthetic pre-training completed: F1={best_metrics.get('macro_f1', 0.0):.3f}")
    
    # Zero-shot evaluation on real data
    _, test_loader, _ = get_real_loaders(batch_size=args.batch_size, seed=args.seed + 1000)
    zero_shot_metrics = eval_model(model, test_loader, device, args.positive_class)
    
    # Apply transfer method
    if args.transfer_method == "zero_shot":
        final_metrics = zero_shot_metrics
    elif args.transfer_method == "temp_scale":
        # Temperature scaling only
        final_metrics = apply_temperature_scaling(model, test_loader, device, args.positive_class)
    else:
        # Fine-tuning or linear probe with limited labels
        final_metrics = transfer_learning(model, test_loader, device, args)
    
    # Save D4 format results
    results = {
        "protocol": "Sim2Real",
        "model": args.model,
        "seed": args.seed,
        "label_ratio": args.label_ratio,
        "transfer_method": args.transfer_method,
        "zero_shot_metrics": {
            "macro_f1": zero_shot_metrics.get("macro_f1", 0.0),
            "falling_f1": zero_shot_metrics.get("f1_fall", 0.0),
            "ece": zero_shot_metrics.get("ece", 0.0)
        },
        "target_metrics": {
            "macro_f1": final_metrics.get("macro_f1", 0.0),
            "falling_f1": final_metrics.get("f1_fall", 0.0),
            "ece": final_metrics.get("ece", 0.0)
        },
        "meta": meta
    }
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sim2Real results saved to {args.out}")
    return results

def train_and_evaluate(model, train_loader, val_loader, device, args):
    """Common training and evaluation loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    best_metrics = {}
    
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss/len(train_loader):.4f}, F1={val_metrics['macro_f1']:.4f}")
    
    return best_metrics

def apply_temperature_scaling(model, loader, device, positive_class):
    """Apply temperature scaling calibration"""
    # Get model outputs
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
    
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Apply temperature scaling
    from src.calibration import temperature_scaling
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    probs_cal, t_opt = temperature_scaling(probs, labels)
    
    if probs_cal is not None and t_opt is not None:
        metrics = compute_metrics(labels, probs_cal, num_classes=4, positive_class=positive_class)
        metrics["temperature"] = t_opt
    else:
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metrics = compute_metrics(labels, probs, num_classes=4, positive_class=positive_class)
        metrics["temperature"] = 1.0
    
    # Add ECE
    confidence = np.max(probs_cal if probs_cal is not None else probs, axis=1)
    predictions = np.argmax(probs_cal if probs_cal is not None else probs, axis=1)
    accuracy = (predictions == labels).astype(float)
    
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

def transfer_learning(model, loader, device, args):
    """Perform transfer learning (fine-tune or linear probe)"""
    # Simple implementation for now - just return evaluation
    return eval_model(model, loader, device, args.positive_class)

if __name__ == "__main__":
    main()