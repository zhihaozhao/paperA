#!/usr/bin/env python3
"""
WiFi-CSI-Sensing-Benchmark 集成脚本

这个脚本帮助您立即开始使用WiFi-CSI-Sensing-Benchmark进行Sim2Real实验。

功能：
1. 环境检查和依赖安装
2. 基准实验运行
3. 模型集成准备
4. 实验结果收集

作者: AI Assistant
日期: 2025-01-16
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import torch
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "wifi_csi_benchmark"


def check_environment():
    """检查环境和依赖"""
    print("🔍 检查环境...")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
    else:
        print("⚠️ CUDA不可用，将使用CPU")
    
    # 检查benchmark目录
    if not BENCHMARK_DIR.exists():
        print(f"❌ Benchmark目录不存在: {BENCHMARK_DIR}")
        print("请先运行: git clone https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark.git benchmarks/wifi_csi_benchmark")
        return False
    
    print(f"✅ Benchmark目录: {BENCHMARK_DIR}")
    return True


def install_benchmark_requirements():
    """安装benchmark依赖"""
    print("📦 安装benchmark依赖...")
    
    req_file = BENCHMARK_DIR / "requirements.txt"
    if req_file.exists():
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                         check=True, cwd=BENCHMARK_DIR)
            print("✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
    
    # 额外安装einops (如果没有)
    try:
        import einops
        print("✅ einops已安装")
    except ImportError:
        print("📦 安装einops...")
        subprocess.run([sys.executable, "-m", "pip", "install", "einops"], check=True)
    
    return True


def check_data_availability():
    """检查数据集可用性"""
    print("📊 检查数据集...")
    
    data_dir = BENCHMARK_DIR / "Data"
    expected_datasets = ["UT_HAR", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widardata"]
    
    available_datasets = []
    for dataset in expected_datasets:
        dataset_path = data_dir / dataset
        if dataset_path.exists():
            print(f"✅ 数据集可用: {dataset}")
            available_datasets.append(dataset)
        else:
            print(f"❌ 数据集缺失: {dataset}")
    
    return available_datasets


def run_benchmark_demo(available_datasets):
    """运行benchmark演示"""
    print("🚀 运行benchmark演示...")
    
    if not available_datasets:
        print("⚠️ 没有可用数据集，跳过演示")
        return
    
    # 选择第一个可用数据集和简单模型进行快速测试
    test_dataset = available_datasets[0].replace("_", "-") if "_" in available_datasets[0] else available_datasets[0]
    if test_dataset == "UT-HAR":
        test_dataset = "UT_HAR_data"
    
    test_model = "MLP"  # 最简单的模型
    
    print(f"📊 测试: {test_model} 在 {test_dataset}")
    
    try:
        # 切换到benchmark目录并运行
        cmd = [sys.executable, "run.py", "--model", test_model, "--dataset", test_dataset]
        result = subprocess.run(cmd, cwd=BENCHMARK_DIR, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 基准测试运行成功!")
            print("输出:")
            print(result.stdout[-500:])  # 显示最后500字符
        else:
            print("❌ 基准测试运行失败")
            print("错误:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️ 测试超时（5分钟），可能数据集较大")
    except Exception as e:
        print(f"❌ 运行出错: {e}")


def create_integration_template():
    """创建集成模板"""
    print("📝 创建集成模板...")
    
    integration_script = PROJECT_ROOT / "scripts" / "sim2real_experiments.py"
    
    template_content = '''#!/usr/bin/env python3
"""
Sim2Real实验脚本模板

集成WiFi-CSI-Sensing-Benchmark进行跨域验证实验
"""

import sys
import os
from pathlib import Path

# 添加benchmark路径到Python路径
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks" / "wifi_csi_benchmark"
sys.path.append(str(BENCHMARK_DIR))

# 导入benchmark模块
from util import load_data_n_model

# 导入我们的模块
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.train_eval import build_model
from src.data_synth import get_synth_loaders

def run_sim2real_comparison():
    """运行Sim2Real对比实验"""
    print("🔄 运行Sim2Real对比实验...")
    
    # 1. 加载真实数据集 (如果可用)
    try:
        train_loader, test_loader, benchmark_model, train_epoch = load_data_n_model(
            "UT_HAR_data", "BiLSTM", str(BENCHMARK_DIR / "Data")
        )
        print("✅ 真实数据集加载成功")
    except Exception as e:
        print(f"❌ 真实数据集加载失败: {e}")
        print("请确保数据集已正确下载和配置")
        return
    
    # 2. 加载合成数据集
    try:
        synth_train, synth_test = get_synth_loaders(
            n=10000, T=128, F=52, difficulty="hard", seed=0
        )
        print("✅ 合成数据集生成成功")
    except Exception as e:
        print(f"❌ 合成数据集生成失败: {e}")
        return
    
    # 3. 对比实验
    # TODO: 实现具体的Sim2Real实验逻辑
    print("📊 开始对比实验...")
    print("  - 合成数据训练 → 真实数据测试")
    print("  - 真实数据训练 → 合成数据测试")
    print("  - 混合训练")
    print("  - 少样本微调")
    
    print("✅ Sim2Real实验模板创建完成!")

if __name__ == "__main__":
    run_sim2real_comparison()
'''
    
    with open(integration_script, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"✅ 集成模板已创建: {integration_script}")
    
    # 创建模型适配脚本
    model_adapter = PROJECT_ROOT / "scripts" / "benchmark_model_adapter.py"
    
    adapter_content = '''#!/usr/bin/env python3
"""
Benchmark模型适配器

将benchmark中的模型适配到我们的项目中
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加benchmark路径
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks" / "wifi_csi_benchmark"
sys.path.append(str(BENCHMARK_DIR))

# 导入benchmark模型
from UT_HAR_model import *
from NTU_Fi_model import *
from widar_model import *

class BenchmarkModelAdapter:
    """Benchmark模型适配器"""
    
    @staticmethod
    def adapt_ut_har_model(model_name: str):
        """适配UT-HAR模型"""
        models = {
            "MLP": UT_HAR_MLP,
            "LeNet": UT_HAR_LeNet,
            "ResNet18": UT_HAR_ResNet18,
            "ResNet50": UT_HAR_ResNet50,
            "ResNet101": UT_HAR_ResNet101,
            "RNN": UT_HAR_RNN,
            "GRU": UT_HAR_GRU,
            "LSTM": UT_HAR_LSTM,
            "BiLSTM": UT_HAR_BiLSTM,
            "CNN+GRU": UT_HAR_CNN_GRU,
            "ViT": UT_HAR_ViT
        }
        
        if model_name in models:
            return models[model_name]()
        else:
            raise ValueError(f"模型 {model_name} 不支持")
    
    @staticmethod
    def get_model_for_dataset(dataset: str, model: str):
        """为特定数据集获取模型"""
        if dataset == "UT_HAR":
            return BenchmarkModelAdapter.adapt_ut_har_model(model)
        elif dataset == "NTU-Fi_HAR":
            # TODO: 添加NTU-Fi模型适配
            pass
        elif dataset == "Widar":
            # TODO: 添加Widar模型适配
            pass
        else:
            raise ValueError(f"数据集 {dataset} 不支持")

# 使用示例
if __name__ == "__main__":
    # 创建ResNet18模型
    model = BenchmarkModelAdapter.adapt_ut_har_model("ResNet18")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
'''
    
    with open(model_adapter, 'w', encoding='utf-8') as f:
        f.write(adapter_content)
    
    print(f"✅ 模型适配器已创建: {model_adapter}")


def create_experiment_plan():
    """创建实验计划"""
    print("📋 创建实验计划...")
    
    plan = {
        "integration_phases": {
            "phase_1": {
                "name": "基准建立",
                "duration": "1-2天",
                "tasks": [
                    "运行benchmark基准测试",
                    "记录各模型在真实数据集上的性能",
                    "建立性能基线"
                ]
            },
            "phase_2": {
                "name": "模型集成",
                "duration": "3-5天",
                "tasks": [
                    "将我们的enhanced模型集成到benchmark",
                    "在真实数据集上测试enhanced模型",
                    "对比enhanced vs benchmark模型性能"
                ]
            },
            "phase_3": {
                "name": "Sim2Real验证",
                "duration": "1-2周",
                "tasks": [
                    "合成数据训练 → 真实数据测试",
                    "真实数据训练 → 合成数据测试",
                    "分析域差距(domain gap)",
                    "评估合成数据的有效性"
                ]
            },
            "phase_4": {
                "name": "少样本学习",
                "duration": "1周",
                "tasks": [
                    "10%真实数据微调实验",
                    "学习曲线分析",
                    "效率对比评估"
                ]
            }
        },
        "expected_outcomes": {
            "paper_contributions": [
                "物理引导合成数据的有效性验证",
                "Enhanced模型架构优势证明",
                "高效Sim2Real转移学习",
                "跨域泛化能力评估"
            ],
            "target_metrics": {
                "synthetic_to_real_performance": "≥80% of baseline",
                "enhanced_vs_bilstm_improvement": "+5-10% accuracy",
                "few_shot_efficiency": "10-20% data → 90-95% performance",
                "cross_domain_generalization": "≥70% performance on unseen domains"
            }
        }
    }
    
    plan_file = PROJECT_ROOT / "docs" / "WiFi_CSI_Benchmark_Integration_Plan.json"
    with open(plan_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 实验计划已创建: {plan_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WiFi-CSI-Sensing-Benchmark集成脚本")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    parser.add_argument("--install-deps", action="store_true", help="安装依赖")
    parser.add_argument("--run-demo", action="store_true", help="运行演示")
    parser.add_argument("--create-templates", action="store_true", help="创建集成模板")
    parser.add_argument("--full-setup", action="store_true", help="完整设置")
    
    args = parser.parse_args()
    
    print("🚀 WiFi-CSI-Sensing-Benchmark 集成工具")
    print("=" * 50)
    
    # 环境检查
    if not check_environment():
        return 1
    
    if args.check_only:
        return 0
    
    # 安装依赖
    if args.install_deps or args.full_setup:
        if not install_benchmark_requirements():
            return 1
    
    # 检查数据集
    available_datasets = check_data_availability()
    
    # 运行演示
    if args.run_demo or args.full_setup:
        run_benchmark_demo(available_datasets)
    
    # 创建模板
    if args.create_templates or args.full_setup:
        create_integration_template()
        create_experiment_plan()
    
    print("\n" + "=" * 50)
    print("🎯 集成准备完成!")
    print("\n📋 下一步操作:")
    print("1. 下载数据集到 benchmarks/wifi_csi_benchmark/Data/")
    print("2. 运行: python scripts/sim2real_experiments.py")
    print("3. 查看实验计划: docs/WiFi_CSI_Benchmark_Integration_Plan.json")
    print("4. 开始Sim2Real实验!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())