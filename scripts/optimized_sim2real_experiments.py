#!/usr/bin/env python3
"""
优化的Sim2Real实验设计

保持现有模型架构不变，专注于使用benchmark的真实数据来验证：
1. 物理引导合成数据的有效性
2. Sim2Real跨域泛化能力
3. 少样本学习效率
4. 现有模型的真实世界性能

作者: AI Assistant
日期: 2025-01-16
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from src.train_eval import build_model, train_one_epoch, eval_model
from src.data_synth import get_synth_loaders


class OptimizedSim2RealExperiments:
    """优化的Sim2Real实验管理器"""
    
    def __init__(self, output_dir="results/sim2real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 现有模型列表 (保持不变)
        self.our_models = ["enhanced", "cnn", "bilstm", "conformer_lite"]
        
        # 选择性使用的benchmark参考
        self.reference_models = ["BiLSTM"]  # 只选择1-2个作为参考
        
        # 支持的真实数据集
        self.real_datasets = ["UT_HAR", "NTU_Fi_HAR"]
    
    def experiment_1_baseline_establishment(self, dataset_name="UT_HAR"):
        """
        实验1: 建立真实数据基线
        目标: 了解我们的模型在真实数据上的性能水平
        """
        print(f"🎯 实验1: {dataset_name} 基线建立")
        
        # 加载真实数据
        from docs.Optimized_Benchmark_Integration_Plan import create_sim2real_experiment
        
        try:
            exp_data = create_sim2real_experiment(dataset_name)
            real_train_loader, real_test_loader = exp_data["real"]
            metadata = exp_data["metadata"]
            
            results = {}
            
            # 在真实数据上测试我们的所有模型
            for model_name in self.our_models:
                print(f"🔄 测试模型: {model_name}")
                
                # 构建模型
                model = build_model(
                    model_name, 
                    F=metadata["F"], 
                    num_classes=metadata["num_classes"],
                    T=metadata["T"]
                )
                
                # 简单训练 (小规模，只是为了获得基线性能)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.CrossEntropyLoss()
                
                # 训练几个epoch获得基线
                for epoch in range(10):  # 少量epoch，只是为了基线
                    train_loss = train_one_epoch(model, real_train_loader, optimizer, criterion, device)
                
                # 评估性能
                eval_results = eval_model(model, real_test_loader, device)
                
                results[model_name] = {
                    "macro_f1": eval_results["macro_f1"],
                    "accuracy": eval_results.get("accuracy", 0),
                    "ece": eval_results.get("ece", 0),
                    "params": sum(p.numel() for p in model.parameters())
                }
                
                print(f"  {model_name}: F1={eval_results['macro_f1']:.3f}")
            
            # 保存结果
            result_file = self.output_dir / f"baseline_{dataset_name}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ 基线结果已保存: {result_file}")
            return results
            
        except Exception as e:
            print(f"❌ 实验1失败: {e}")
            print(f"跳过实验1，可能是数据集 {dataset_name} 未下载")
            return {}
    
    def experiment_2_sim2real_core(self, dataset_name="UT_HAR"):
        """
        实验2: 核心Sim2Real验证
        目标: 验证合成数据训练的模型在真实数据上的表现
        """
        print(f"🎯 实验2: {dataset_name} Sim2Real核心验证")
        
        try:
            from docs.Optimized_Benchmark_Integration_Plan import create_sim2real_experiment
            exp_data = create_sim2real_experiment(dataset_name)
            
            synth_train_loader, synth_test_loader = exp_data["synthetic"]
            real_train_loader, real_test_loader = exp_data["real"]
            metadata = exp_data["metadata"]
            
            results = {}
            
            for model_name in self.our_models:
                print(f"🔄 Sim2Real测试: {model_name}")
                
                # 1. 在合成数据上训练
                model = build_model(
                    model_name,
                    F=metadata["F"],
                    num_classes=metadata["num_classes"], 
                    T=metadata["T"]
                )
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.CrossEntropyLoss()
                
                # 在合成数据上训练
                print("  📊 在合成数据上训练...")
                for epoch in range(20):  # 合成数据上充分训练
                    train_loss = train_one_epoch(model, synth_train_loader, optimizer, criterion, device)
                
                # 2. 在合成数据上评估 (内部验证)
                synth_results = eval_model(model, synth_test_loader, device)
                
                # 3. 在真实数据上测试 (跨域验证)
                print("  🌍 在真实数据上测试...")
                real_results = eval_model(model, real_test_loader, device)
                
                # 计算Sim2Real比率
                sim2real_ratio = real_results["macro_f1"] / max(synth_results["macro_f1"], 0.001)
                
                results[model_name] = {
                    "synthetic_f1": synth_results["macro_f1"],
                    "real_f1": real_results["macro_f1"],
                    "sim2real_ratio": sim2real_ratio,
                    "synthetic_ece": synth_results.get("ece", 0),
                    "real_ece": real_results.get("ece", 0)
                }
                
                print(f"  {model_name}: 合成F1={synth_results['macro_f1']:.3f}, "
                      f"真实F1={real_results['macro_f1']:.3f}, 比率={sim2real_ratio:.3f}")
            
            # 保存结果
            result_file = self.output_dir / f"sim2real_core_{dataset_name}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Sim2Real结果已保存: {result_file}")
            return results
            
        except Exception as e:
            print(f"❌ 实验2失败: {e}")
            return {}
    
    def experiment_3_few_shot_learning(self, dataset_name="UT_HAR", ratios=[0.05, 0.1, 0.2, 0.5]):
        """
        实验3: 少样本学习效率
        目标: 评估用少量真实数据微调合成模型的效果
        """
        print(f"🎯 实验3: {dataset_name} 少样本学习")
        
        try:
            from docs.Optimized_Benchmark_Integration_Plan import create_sim2real_experiment
            exp_data = create_sim2real_experiment(dataset_name)
            
            synth_train_loader, synth_test_loader = exp_data["synthetic"]
            real_train_loader, real_test_loader = exp_data["real"]
            metadata = exp_data["metadata"]
            adapter = exp_data["adapter"]
            real_data = exp_data["real_data"]
            
            results = {}
            
            # 只测试我们最好的模型 (enhanced)
            model_name = "enhanced"
            print(f"🔄 少样本学习测试: {model_name}")
            
            results[model_name] = {}
            
            for ratio in ratios:
                print(f"  📊 测试比例: {ratio*100:.1f}%")
                
                # 创建少样本数据
                few_shot_loader = adapter.create_few_shot_loader(real_data, ratio)
                
                # 1. 先在合成数据上预训练
                model = build_model(
                    model_name,
                    F=metadata["F"],
                    num_classes=metadata["num_classes"],
                    T=metadata["T"]  
                )
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.CrossEntropyLoss()
                
                # 合成数据预训练
                for epoch in range(15):
                    train_loss = train_one_epoch(model, synth_train_loader, optimizer, criterion, device)
                
                # 2. 少样本微调
                optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.0001)  # 更小的学习率
                
                for epoch in range(10):  # 少量epoch避免过拟合
                    train_loss = train_one_epoch(model, few_shot_loader, optimizer_ft, criterion, device)
                
                # 3. 在真实测试集上评估
                test_results = eval_model(model, real_test_loader, device)
                
                results[model_name][f"{ratio*100:.1f}%"] = {
                    "macro_f1": test_results["macro_f1"],
                    "ece": test_results.get("ece", 0),
                    "sample_count": len(few_shot_loader.dataset)
                }
                
                print(f"    {ratio*100:.1f}%数据: F1={test_results['macro_f1']:.3f}, "
                      f"样本数={len(few_shot_loader.dataset)}")
            
            # 保存结果
            result_file = self.output_dir / f"few_shot_{dataset_name}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ 少样本学习结果已保存: {result_file}")
            return results
            
        except Exception as e:
            print(f"❌ 实验3失败: {e}")
            return {}
    
    def experiment_4_domain_gap_analysis(self, dataset_name="UT_HAR"):
        """
        实验4: 域差距分析
        目标: 分析合成数据与真实数据的差距特征
        """
        print(f"🎯 实验4: {dataset_name} 域差距分析")
        
        try:
            from docs.Optimized_Benchmark_Integration_Plan import create_sim2real_experiment
            exp_data = create_sim2real_experiment(dataset_name)
            
            synth_train_loader, synth_test_loader = exp_data["synthetic"]
            real_train_loader, real_test_loader = exp_data["real"]
            metadata = exp_data["metadata"]
            
            # 使用最好的模型进行分析
            model_name = "enhanced"
            print(f"🔄 域差距分析: {model_name}")
            
            results = {}
            
            # 1. 真实→合成方向
            print("  📊 真实数据训练 → 合成数据测试")
            model_r2s = build_model(
                model_name,
                F=metadata["F"], 
                num_classes=metadata["num_classes"],
                T=metadata["T"]
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_r2s = model_r2s.to(device)
            optimizer = torch.optim.Adam(model_r2s.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # 在真实数据上训练
            for epoch in range(15):
                train_loss = train_one_epoch(model_r2s, real_train_loader, optimizer, criterion, device)
            
            # 在真实数据测试集上评估 (内域)
            real_real_results = eval_model(model_r2s, real_test_loader, device)
            
            # 在合成数据上测试 (跨域)
            real_synth_results = eval_model(model_r2s, synth_test_loader, device)
            
            # 2. 合成→真实方向 (前面实验已做过，这里简化)
            print("  📊 合成数据训练 → 真实数据测试")
            model_s2r = build_model(
                model_name,
                F=metadata["F"],
                num_classes=metadata["num_classes"],
                T=metadata["T"]
            )
            
            model_s2r = model_s2r.to(device)
            optimizer = torch.optim.Adam(model_s2r.parameters(), lr=0.001)
            
            # 在合成数据上训练
            for epoch in range(15):
                train_loss = train_one_epoch(model_s2r, synth_train_loader, optimizer, criterion, device)
            
            # 在合成数据测试集上评估 (内域)
            synth_synth_results = eval_model(model_s2r, synth_test_loader, device)
            
            # 在真实数据上测试 (跨域)  
            synth_real_results = eval_model(model_s2r, real_test_loader, device)
            
            # 计算域差距指标
            results = {
                "real_to_real": real_real_results["macro_f1"],     # 真实内域基准
                "real_to_synthetic": real_synth_results["macro_f1"], # 真实→合成
                "synthetic_to_synthetic": synth_synth_results["macro_f1"], # 合成内域基准
                "synthetic_to_real": synth_real_results["macro_f1"],      # 合成→真实
                
                # 域差距指标
                "real2synth_gap": 1 - (real_synth_results["macro_f1"] / max(real_real_results["macro_f1"], 0.001)),
                "synth2real_gap": 1 - (synth_real_results["macro_f1"] / max(synth_synth_results["macro_f1"], 0.001)),
                
                # 对称性分析
                "domain_asymmetry": abs(
                    (real_synth_results["macro_f1"] / max(real_real_results["macro_f1"], 0.001)) -
                    (synth_real_results["macro_f1"] / max(synth_synth_results["macro_f1"], 0.001))
                )
            }
            
            # 保存结果
            result_file = self.output_dir / f"domain_gap_{dataset_name}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ 域差距分析结果已保存: {result_file}")
            print(f"  合成→真实差距: {results['synth2real_gap']:.3f}")
            print(f"  真实→合成差距: {results['real2synth_gap']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ 实验4失败: {e}")
            return {}
    
    def run_all_experiments(self, datasets=["UT_HAR"]):
        """运行所有优化实验"""
        print("🚀 开始运行所有优化Sim2Real实验")
        
        all_results = {}
        
        for dataset in datasets:
            print(f"\n{'='*50}")
            print(f"🎯 数据集: {dataset}")
            print(f"{'='*50}")
            
            all_results[dataset] = {}
            
            # 运行各个实验 (按需执行)
            try:
                # all_results[dataset]["baseline"] = self.experiment_1_baseline_establishment(dataset)
                all_results[dataset]["sim2real_core"] = self.experiment_2_sim2real_core(dataset) 
                all_results[dataset]["few_shot"] = self.experiment_3_few_shot_learning(dataset)
                # all_results[dataset]["domain_gap"] = self.experiment_4_domain_gap_analysis(dataset)
                
            except Exception as e:
                print(f"❌ 数据集 {dataset} 实验失败: {e}")
                print("可能原因: 数据集未下载或环境问题")
                continue
        
        # 保存汇总结果
        summary_file = self.output_dir / "all_experiments_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n🎉 所有实验完成! 结果汇总: {summary_file}")
        
        # 生成简要报告
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results):
        """生成实验总结报告"""
        report = []
        report.append("# 📊 Sim2Real实验总结报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for dataset, exp_results in results.items():
            if not exp_results:
                continue
                
            report.append(f"## 🎯 数据集: {dataset}\n")
            
            # Sim2Real核心结果
            if "sim2real_core" in exp_results and exp_results["sim2real_core"]:
                report.append("### Sim2Real核心结果")
                for model, metrics in exp_results["sim2real_core"].items():
                    ratio = metrics.get("sim2real_ratio", 0)
                    real_f1 = metrics.get("real_f1", 0)
                    report.append(f"- **{model}**: 真实F1={real_f1:.3f}, Sim2Real比率={ratio:.3f}")
                report.append("")
            
            # 少样本学习结果
            if "few_shot" in exp_results and exp_results["few_shot"]:
                report.append("### 少样本学习效率")
                for model, ratios in exp_results["few_shot"].items():
                    report.append(f"- **{model}**:")
                    for ratio, metrics in ratios.items():
                        f1 = metrics.get("macro_f1", 0)
                        samples = metrics.get("sample_count", 0)
                        report.append(f"  - {ratio}: F1={f1:.3f} ({samples}样本)")
                report.append("")
        
        # 保存报告
        report_file = self.output_dir / "experiment_summary_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📝 实验报告已生成: {report_file}")


# 使用示例
if __name__ == "__main__":
    print("🚀 优化Sim2Real实验")
    
    # 创建实验管理器
    experiments = OptimizedSim2RealExperiments()
    
    # 运行核心实验 (根据数据集可用性调整)
    try:
        # 只测试可用的数据集
        available_datasets = ["UT_HAR"]  # 根据实际情况调整
        
        results = experiments.run_all_experiments(available_datasets)
        
        if results:
            print("\n🎉 实验完成!")
            print("关键发现:")
            for dataset, exp_results in results.items():
                if "sim2real_core" in exp_results:
                    for model, metrics in exp_results["sim2real_core"].items():
                        ratio = metrics.get("sim2real_ratio", 0)
                        print(f"- {model}在{dataset}: Sim2Real比率={ratio:.3f}")
        
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        print("这是正常的，如果还没有下载真实数据集的话")
        print("\n📝 实验设计已完成，代码框架已就绪!")
        print("下一步: 下载数据集并运行实验")