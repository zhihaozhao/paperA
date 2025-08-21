#!/usr/bin/env python3
"""
Parameter Tuning Tool - PhD Thesis Hyperparameter Optimization System (English Version)
Supports grid search, Bayesian optimization, random search and other tuning strategies

Author: PhD Thesis Research
Date: 2025
Version: v2.0-en
"""

import os
import sys
import json
import numpy as np
import itertools
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
from pathlib import Path

# Add core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))


class ParameterTuner:
    """Unified Parameter Tuner - Supports multiple optimization strategies"""
    
    def __init__(self, 
                 base_config: Dict[str, Any],
                 optimization_target: str = "macro_f1",
                 tuning_strategy: str = "grid_search"):
        """
        Initialize parameter tuner
        Args:
            base_config: Base experiment configuration
            optimization_target: Optimization target metric
            tuning_strategy: Tuning strategy selection
        """
        self.base_config = base_config
        self.optimization_target = optimization_target
        self.tuning_strategy = tuning_strategy
        self.results_dir = Path("experiments/results/parameter_tuning")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging_system()
        
        # Tuning history tracking
        self.tuning_history = []
        self.best_params = None
        self.best_score = -np.inf if "f1" in optimization_target.lower() else np.inf
        
    def _setup_logging_system(self):
        """Setup tuning logging system"""
        log_file = self.results_dir / f"parameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ParameterTuning - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("ParameterTuner")
    
    def define_search_space(self) -> Dict[str, List]:
        """Define hyperparameter search space"""
        search_space = {
            # Learning rate related
            "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "weight_decay": [0, 1e-5, 1e-4, 1e-3],
            "lr_scheduler": ["constant", "step", "cosine", "exponential"],
            
            # Model architecture related
            "conv_channels": [
                [32, 64, 128],
                [64, 128, 256], 
                [128, 256, 512]
            ],
            "lstm_hidden": [64, 128, 256],
            "lstm_layers": [1, 2, 3],
            "attention_heads": [4, 8, 16],
            
            # Regularization related
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "confidence_reg_lambda": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "se_reduction": [8, 16, 32],
            
            # Training related
            "batch_size": [32, 64, 128],
            "gradient_clipping": [0.5, 1.0, 2.0, 5.0],
            "early_stop_patience": [10, 15, 20, 25]
        }
        
        self.logger.info(f"🔍 Search space defined with {len(search_space)} hyperparameter categories")
        for param_name, candidates in search_space.items():
            self.logger.info(f"   {param_name}: {len(candidates)} candidates")
        
        return search_space
    
    def grid_search_optimization(self, search_space: Dict[str, List], max_experiments: int = 100) -> Dict[str, Any]:
        """
        Grid search hyperparameter optimization
        Args:
            search_space: Hyperparameter search space dictionary
            max_experiments: Maximum number of experiments limit
        Returns:
            Optimization result dictionary
        """
        self.logger.info(f"🔬 Starting grid search optimization - Max experiments: {max_experiments}")
        
        # Generate all parameter combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        total_combinations = np.prod([len(value_list) for value_list in param_values])
        self.logger.info(f"Total parameter combinations: {total_combinations}")
        
        if total_combinations > max_experiments:
            self.logger.warning(f"⚠️  Combinations exceed limit, randomly sampling {max_experiments} combinations")
            param_combinations = self._random_sample_combinations(search_space, max_experiments)
        else:
            param_combinations = list(itertools.product(*param_values))
        
        # Execute grid search
        for exp_idx, param_combination in enumerate(param_combinations):
            experiment_config = dict(zip(param_names, param_combination))
            experiment_config.update(self.base_config)
            
            self.logger.info(f"Experiment {exp_idx+1}/{len(param_combinations)}: {experiment_config}")
            
            try:
                # Execute single experiment
                experiment_result = self._execute_single_experiment(experiment_config)
                
                # Record result
                self._record_tuning_result(experiment_config, experiment_result)
                
                # Update best parameters
                self._update_best_parameters(experiment_config, experiment_result)
                
            except Exception as error:
                self.logger.error(f"Experiment {exp_idx+1} failed: {error}")
        
        return self._generate_tuning_report()
    
    def bayesian_optimization(self, search_space: Dict[str, List], max_experiments: int = 50) -> Dict[str, Any]:
        """
        Bayesian optimization hyperparameter tuning
        Args:
            search_space: Hyperparameter search space
            max_experiments: Maximum number of experiments
        Returns:
            Optimization result dictionary
        """
        self.logger.info(f"🧠 Starting Bayesian optimization - Max experiments: {max_experiments}")
        
        try:
            # Try importing optuna
            import optuna
        except ImportError:
            self.logger.warning("Optuna not installed, falling back to random search")
            return self.random_search_optimization(search_space, max_experiments)
        
        def objective_function(trial):
            """Optuna objective function"""
            # Sample parameters from trial
            trial_params = {}
            for param_name, candidates in search_space.items():
                if isinstance(candidates[0], (int, float)):
                    if isinstance(candidates[0], int):
                        trial_params[param_name] = trial.suggest_int(param_name, min(candidates), max(candidates))
                    else:
                        trial_params[param_name] = trial.suggest_float(param_name, min(candidates), max(candidates))
                else:
                    trial_params[param_name] = trial.suggest_categorical(param_name, candidates)
            
            # Execute experiment
            experiment_config = {**trial_params, **self.base_config}
            experiment_result = self._execute_single_experiment(experiment_config)
            
            # Record result
            self._record_tuning_result(experiment_config, experiment_result)
            
            # Return objective value
            return experiment_result.get(self.optimization_target, 0.0)
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize' if "f1" in self.optimization_target.lower() else 'minimize')
        study.optimize(objective_function, n_trials=max_experiments)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        self.logger.info(f"🏆 Bayesian optimization completed:")
        self.logger.info(f"   Best {self.optimization_target}: {self.best_score:.4f}")
        self.logger.info(f"   Best parameters: {self.best_params}")
        
        return self._generate_tuning_report()
    
    def random_search_optimization(self, search_space: Dict[str, List], max_experiments: int = 50) -> Dict[str, Any]:
        """
        Random search hyperparameter optimization
        Args:
            search_space: Hyperparameter search space
            max_experiments: Maximum number of experiments
        Returns:
            Optimization result dictionary
        """
        self.logger.info(f"🎲 Starting random search optimization - Max experiments: {max_experiments}")
        
        for exp_idx in range(max_experiments):
            # Random sample parameters
            random_params = {}
            for param_name, candidates in search_space.items():
                random_params[param_name] = np.random.choice(candidates)
            
            experiment_config = {**random_params, **self.base_config}
            
            self.logger.info(f"Random experiment {exp_idx+1}/{max_experiments}")
            
            try:
                experiment_result = self._execute_single_experiment(experiment_config)
                self._record_tuning_result(experiment_config, experiment_result)
                self._update_best_parameters(experiment_config, experiment_result)
                
            except Exception as error:
                self.logger.error(f"Random experiment {exp_idx+1} failed: {error}")
        
        return self._generate_tuning_report()
    
    def _random_sample_combinations(self, search_space: Dict[str, List], sample_count: int) -> List[Tuple]:
        """Randomly sample parameter combinations"""
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        sampled_combinations = []
        for _ in range(sample_count):
            single_combination = tuple(np.random.choice(value_list) for value_list in param_values)
            sampled_combinations.append(single_combination)
        
        return sampled_combinations
    
    def _execute_single_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, float]:
        """Execute single parameter tuning experiment"""
        # This should call the actual training pipeline
        # For demonstration, returning simulated results
        
        # Simulate experiment execution time
        import time
        time.sleep(0.1)  # Simulate training time
        
        # Simulate performance results (generate reasonable simulated values based on parameter combinations)
        learning_rate = experiment_config.get("learning_rate", 1e-3)
        lambda_reg = experiment_config.get("confidence_reg_lambda", 1e-3)
        
        # Heuristic performance estimation based on parameters
        base_f1 = 0.83
        lr_factor = 0.02 * np.log10(learning_rate / 1e-3) if learning_rate <= 1e-2 else -0.05
        reg_factor = 0.01 * np.log10(lambda_reg / 1e-3) if lambda_reg <= 1e-2 else -0.02
        noise = np.random.normal(0, 0.005)  # Random noise
        
        simulated_f1 = np.clip(base_f1 + lr_factor + reg_factor + noise, 0.4, 0.9)
        
        return {
            "macro_f1": float(simulated_f1),
            "accuracy": float(simulated_f1 * 1.02),
            "ece": float(np.clip(0.05 - reg_factor * 0.5, 0.01, 0.15)),
            "training_time": float(np.random.uniform(300, 1800)),  # 5-30 minutes
            "convergence_epochs": int(np.random.uniform(20, 100))
        }
    
    def _record_tuning_result(self, experiment_config: Dict[str, Any], experiment_result: Dict[str, float]):
        """Record single tuning result"""
        record_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_config": experiment_config,
            "experiment_result": experiment_result,
            "experiment_number": len(self.tuning_history) + 1
        }
        
        self.tuning_history.append(record_entry)
        
        # Save tuning history in real-time
        history_file = self.results_dir / "tuning_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.tuning_history, f, ensure_ascii=False, indent=2)
    
    def _update_best_parameters(self, experiment_config: Dict[str, Any], experiment_result: Dict[str, float]):
        """Update best parameter records"""
        current_score = experiment_result.get(self.optimization_target, 0.0)
        
        should_update = False
        if "f1" in self.optimization_target.lower() or "accuracy" in self.optimization_target.lower():
            # Maximize metrics
            if current_score > self.best_score:
                should_update = True
        else:
            # Minimize metrics (like ECE)
            if current_score < self.best_score:
                should_update = True
        
        if should_update:
            self.best_score = current_score
            self.best_params = experiment_config.copy()
            
            self.logger.info(f"🎯 Found better parameters: {self.optimization_target}={current_score:.4f}")
    
    def _generate_tuning_report(self) -> Dict[str, Any]:
        """Generate detailed tuning report"""
        if not self.tuning_history:
            return {"error": "No tuning history records"}
        
        # Analyze tuning history
        all_scores = [result["experiment_result"][self.optimization_target] for result in self.tuning_history]
        
        tuning_statistics = {
            "total_experiments": len(self.tuning_history),
            "best_score": self.best_score,
            "best_parameters": self.best_params,
            "score_statistics": {
                "mean": float(np.mean(all_scores)),
                "std_dev": float(np.std(all_scores)),
                "max": float(np.max(all_scores)),
                "min": float(np.min(all_scores)),
                "median": float(np.median(all_scores))
            }
        }
        
        # Parameter importance analysis
        param_importance = self._analyze_parameter_importance()
        tuning_statistics["parameter_importance"] = param_importance
        
        # Save complete report
        report_file = self.results_dir / f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(tuning_statistics, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Tuning report saved: {report_file}")
        return tuning_statistics
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance to performance"""
        param_importance = {}
        
        # Simplified parameter importance analysis
        for result_record in self.tuning_history:
            config = result_record["experiment_config"]
            score = result_record["experiment_result"][self.optimization_target]
            
            for param_name, param_value in config.items():
                if param_name not in param_importance:
                    param_importance[param_name] = []
                param_importance[param_name].append(score)
        
        # Calculate variance for each parameter (simplified importance metric)
        importance_scores = {}
        for param_name, score_list in param_importance.items():
            if len(score_list) > 1:
                importance_scores[param_name] = float(np.var(score_list))
            else:
                importance_scores[param_name] = 0.0
        
        return importance_scores
    
    def generate_optimal_config(self) -> Dict[str, Any]:
        """Generate optimal configuration based on tuning results"""
        if self.best_params is None:
            raise ValueError("No best parameters found, please run parameter tuning first")
        
        optimal_config = {
            "meta_info": {
                "tuning_strategy": self.tuning_strategy,
                "optimization_target": self.optimization_target,
                "best_score": self.best_score,
                "tuning_completion_time": datetime.now().isoformat()
            },
            "optimal_hyperparameters": self.best_params,
            "usage_instructions": {
                "training_command": f"python run_experiments_en.py --config optimal_config.json",
                "expected_performance": f"{self.optimization_target} ≈ {self.best_score:.4f}",
                "important_notes": [
                    "Ensure using the same random seeds",
                    "Verify hardware environment consistency", 
                    "Check data preprocessing pipeline"
                ]
            }
        }
        
        # Save optimal configuration
        config_file = self.results_dir / "optimal_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(optimal_config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"🏆 Optimal configuration saved: {config_file}")
        return optimal_config


class ParameterValidator:
    """Parameter validity validator"""
    
    @staticmethod
    def validate_d2_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate D2 protocol parameter validity"""
        error_list = []
        
        # Required field check
        required_fields = ["learning_rate", "batch_size", "max_training_epochs"]
        for field in required_fields:
            if field not in config:
                error_list.append(f"Missing required field: {field}")
        
        # Range checks
        if "learning_rate" in config:
            learning_rate = config["learning_rate"]
            if not (1e-6 <= learning_rate <= 1e-1):
                error_list.append(f"Learning rate out of reasonable range: {learning_rate}")
        
        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not (1 <= batch_size <= 512):
                error_list.append(f"Batch size out of reasonable range: {batch_size}")
        
        return len(error_list) == 0, error_list
    
    @staticmethod
    def validate_cdae_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate CDAE protocol parameter validity"""
        error_list = []
        
        # LOSO/LORO specific checks
        if "loso_subject_list" in config:
            subject_list = config["loso_subject_list"]
            if len(subject_list) < 3:
                error_list.append("Too few LOSO subjects, minimum 3 required")
        
        return len(error_list) == 0, error_list
    
    @staticmethod
    def validate_stea_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate STEA protocol parameter validity"""
        error_list = []
        
        # Label ratio checks
        if "label_ratio_list" in config:
            label_ratios = config["label_ratio_list"]
            if not all(0 < ratio <= 100 for ratio in label_ratios):
                error_list.append("Label ratios must be in (0, 100] range")
        
        return len(error_list) == 0, error_list


def main():
    """Parameter tuning main function"""
    print("🔧 WiFi CSI Parameter Tuning System (English Version)")
    print("=" * 60)
    
    # Base configuration
    base_config = {
        "model_name": "Enhanced",
        "dataset": "CSI-Fall",
        "device": "cuda",
        "random_seed": 42
    }
    
    # Create tuner
    tuner = ParameterTuner(base_config, optimization_target="macro_f1", tuning_strategy="grid_search")
    
    # Define search space
    search_space = tuner.define_search_space()
    
    # Execute parameter tuning
    print("Select tuning strategy:")
    print("1. Grid Search (comprehensive but time-consuming)")
    print("2. Bayesian Optimization (intelligent and efficient)")  
    print("3. Random Search (fast exploration)")
    
    strategy_choice = input("Please enter choice (1-3): ").strip()
    
    if strategy_choice == "1":
        tuning_result = tuner.grid_search_optimization(search_space, max_experiments=50)
    elif strategy_choice == "2":
        tuning_result = tuner.bayesian_optimization(search_space, max_experiments=30)
    elif strategy_choice == "3":
        tuning_result = tuner.random_search_optimization(search_space, max_experiments=25)
    else:
        print("Invalid choice, using default grid search")
        tuning_result = tuner.grid_search_optimization(search_space, max_experiments=20)
    
    # Generate optimal configuration
    optimal_config = tuner.generate_optimal_config()
    
    print(f"\n🎉 Parameter tuning completed!")
    print(f"🏆 Best {tuner.optimization_target}: {tuner.best_score:.4f}")
    print(f"📋 Optimal configuration saved to: experiments/results/parameter_tuning/")


if __name__ == "__main__":
    main()