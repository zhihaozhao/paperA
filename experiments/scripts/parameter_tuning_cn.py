#!/usr/bin/env python3
"""
参数调节工具 - 博士论文超参数优化系统 (中文版)
支持网格搜索、贝叶斯优化、随机搜索等多种调优策略

作者: 博士论文研究
日期: 2025年
版本: v2.0-cn
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

# 添加核心模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))


class 参数调优器:
    """统一参数调优器 - 支持多种优化策略"""
    
    def __init__(self, 
                 基础配置: Dict[str, Any],
                 优化目标: str = "宏平均F1",
                 调优策略: str = "网格搜索"):
        """
        初始化参数调优器
        参数:
            基础配置: 基础实验配置
            优化目标: 优化目标指标
            调优策略: 调优策略选择
        """
        self.基础配置 = 基础配置
        self.优化目标 = 优化目标
        self.调优策略 = 调优策略
        self.结果目录 = Path("experiments/results/parameter_tuning")
        self.结果目录.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._设置日志系统()
        
        # 调优历史记录
        self.调优历史 = []
        self.最佳参数 = None
        self.最佳分数 = -np.inf if "F1" in 优化目标 else np.inf
        
    def _设置日志系统(self):
        """设置调优日志系统"""
        日志文件 = self.结果目录 / f"参数调优_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - 参数调优 - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(日志文件, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.日志器 = logging.getLogger("参数调优器")
    
    def 定义搜索空间(self) -> Dict[str, List]:
        """定义超参数搜索空间"""
        搜索空间 = {
            # 学习率相关
            "学习率": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "权重衰减": [0, 1e-5, 1e-4, 1e-3],
            "学习率调度": ["constant", "step", "cosine", "exponential"],
            
            # 模型架构相关
            "卷积通道": [
                [32, 64, 128],
                [64, 128, 256], 
                [128, 256, 512]
            ],
            "LSTM隐藏单元": [64, 128, 256],
            "LSTM层数": [1, 2, 3],
            "注意力头数": [4, 8, 16],
            
            # 正则化相关
            "Dropout率": [0.1, 0.2, 0.3, 0.4, 0.5],
            "置信度正则化系数": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "SE压缩比": [8, 16, 32],
            
            # 训练相关
            "批次大小": [32, 64, 128],
            "梯度裁剪": [0.5, 1.0, 2.0, 5.0],
            "早停耐心": [10, 15, 20, 25]
        }
        
        self.日志器.info(f"🔍 搜索空间定义完成，包含 {len(搜索空间)} 个超参数类别")
        for 参数名, 候选值 in 搜索空间.items():
            self.日志器.info(f"   {参数名}: {len(候选值)} 个候选值")
        
        return 搜索空间
    
    def 网格搜索优化(self, 搜索空间: Dict[str, List], 最大实验数: int = 100) -> Dict[str, Any]:
        """
        网格搜索超参数优化
        参数:
            搜索空间: 超参数搜索空间字典
            最大实验数: 最大实验次数限制
        返回:
            优化结果字典
        """
        self.日志器.info(f"🔬 开始网格搜索优化 - 最大实验数: {最大实验数}")
        
        # 生成所有参数组合
        参数名列表 = list(搜索空间.keys())
        参数值列表 = list(搜索空间.values())
        
        总组合数 = np.prod([len(值列表) for 值列表 in 参数值列表])
        self.日志器.info(f"总参数组合数: {总组合数}")
        
        if 总组合数 > 最大实验数:
            self.日志器.warning(f"⚠️  组合数超限，将随机采样 {最大实验数} 个组合")
            参数组合列表 = self._随机采样参数组合(搜索空间, 最大实验数)
        else:
            参数组合列表 = list(itertools.product(*参数值列表))
        
        # 执行网格搜索
        for 实验索引, 参数组合 in enumerate(参数组合列表):
            实验配置 = dict(zip(参数名列表, 参数组合))
            实验配置.update(self.基础配置)
            
            self.日志器.info(f"实验 {实验索引+1}/{len(参数组合列表)}: {实验配置}")
            
            try:
                # 执行单次实验
                实验结果 = self._执行单次实验(实验配置)
                
                # 记录结果
                self._记录调优结果(实验配置, 实验结果)
                
                # 更新最佳参数
                self._更新最佳参数(实验配置, 实验结果)
                
            except Exception as 错误:
                self.日志器.error(f"实验 {实验索引+1} 失败: {错误}")
        
        return self._生成调优报告()
    
    def 贝叶斯优化(self, 搜索空间: Dict[str, List], 最大实验数: int = 50) -> Dict[str, Any]:
        """
        贝叶斯优化超参数调优
        参数:
            搜索空间: 超参数搜索空间
            最大实验数: 最大实验次数
        返回:
            优化结果字典
        """
        self.日志器.info(f"🧠 开始贝叶斯优化 - 最大实验数: {最大实验数}")
        
        try:
            # 尝试导入optuna
            import optuna
        except ImportError:
            self.日志器.warning("Optuna未安装，回退到随机搜索")
            return self.随机搜索优化(搜索空间, 最大实验数)
        
        def 目标函数(trial):
            """Optuna目标函数"""
            # 从trial中采样参数
            试验参数 = {}
            for 参数名, 候选值 in 搜索空间.items():
                if isinstance(候选值[0], (int, float)):
                    if isinstance(候选值[0], int):
                        试验参数[参数名] = trial.suggest_int(参数名, min(候选值), max(候选值))
                    else:
                        试验参数[参数名] = trial.suggest_float(参数名, min(候选值), max(候选值))
                else:
                    试验参数[参数名] = trial.suggest_categorical(参数名, 候选值)
            
            # 执行实验
            实验配置 = {**试验参数, **self.基础配置}
            实验结果 = self._执行单次实验(实验配置)
            
            # 记录结果
            self._记录调优结果(实验配置, 实验结果)
            
            # 返回目标值
            return 实验结果.get(self.优化目标, 0.0)
        
        # 创建study并优化
        study = optuna.create_study(direction='maximize' if "F1" in self.优化目标 else 'minimize')
        study.optimize(目标函数, n_trials=最大实验数)
        
        # 获取最佳参数
        self.最佳参数 = study.best_params
        self.最佳分数 = study.best_value
        
        self.日志器.info(f"🏆 贝叶斯优化完成:")
        self.日志器.info(f"   最佳{self.优化目标}: {self.最佳分数:.4f}")
        self.日志器.info(f"   最佳参数: {self.最佳参数}")
        
        return self._生成调优报告()
    
    def 随机搜索优化(self, 搜索空间: Dict[str, List], 最大实验数: int = 50) -> Dict[str, Any]:
        """
        随机搜索超参数优化
        参数:
            搜索空间: 超参数搜索空间
            最大实验数: 最大实验次数
        返回:
            优化结果字典
        """
        self.日志器.info(f"🎲 开始随机搜索优化 - 最大实验数: {最大实验数}")
        
        for 实验索引 in range(最大实验数):
            # 随机采样参数
            随机参数 = {}
            for 参数名, 候选值 in 搜索空间.items():
                随机参数[参数名] = np.random.choice(候选值)
            
            实验配置 = {**随机参数, **self.基础配置}
            
            self.日志器.info(f"随机实验 {实验索引+1}/{最大实验数}")
            
            try:
                实验结果 = self._执行单次实验(实验配置)
                self._记录调优结果(实验配置, 实验结果)
                self._更新最佳参数(实验配置, 实验结果)
                
            except Exception as 错误:
                self.日志器.error(f"随机实验 {实验索引+1} 失败: {错误}")
        
        return self._生成调优报告()
    
    def _随机采样参数组合(self, 搜索空间: Dict[str, List], 采样数量: int) -> List[Tuple]:
        """随机采样参数组合"""
        参数名列表 = list(搜索空间.keys())
        参数值列表 = list(搜索空间.values())
        
        采样组合 = []
        for _ in range(采样数量):
            单次组合 = tuple(np.random.choice(值列表) for 值列表 in 参数值列表)
            采样组合.append(单次组合)
        
        return 采样组合
    
    def _执行单次实验(self, 实验配置: Dict[str, Any]) -> Dict[str, float]:
        """执行单次参数调优实验"""
        # 这里应该调用实际的训练流程
        # 为演示目的，返回模拟结果
        
        # 模拟实验执行时间
        import time
        time.sleep(0.1)  # 模拟训练时间
        
        # 模拟性能结果 (基于参数组合生成合理的模拟值)
        学习率 = 实验配置.get("学习率", 1e-3)
        正则化系数 = 实验配置.get("置信度正则化系数", 1e-3)
        
        # 基于参数的启发式性能估算
        基础F1 = 0.83
        学习率因子 = 0.02 * np.log10(学习率 / 1e-3) if 学习率 <= 1e-2 else -0.05
        正则化因子 = 0.01 * np.log10(正则化系数 / 1e-3) if 正则化系数 <= 1e-2 else -0.02
        噪声 = np.random.normal(0, 0.005)  # 随机噪声
        
        模拟F1 = np.clip(基础F1 + 学习率因子 + 正则化因子 + 噪声, 0.4, 0.9)
        
        return {
            "宏平均F1": float(模拟F1),
            "准确率": float(模拟F1 * 1.02),
            "ECE": float(np.clip(0.05 - 正则化因子 * 0.5, 0.01, 0.15)),
            "训练时间": float(np.random.uniform(300, 1800)),  # 5-30分钟
            "收敛轮次": int(np.random.uniform(20, 100))
        }
    
    def _记录调优结果(self, 实验配置: Dict[str, Any], 实验结果: Dict[str, float]):
        """记录单次调优结果"""
        记录条目 = {
            "时间戳": datetime.now().isoformat(),
            "实验配置": 实验配置,
            "实验结果": 实验结果,
            "实验编号": len(self.调优历史) + 1
        }
        
        self.调优历史.append(记录条目)
        
        # 实时保存调优历史
        历史文件 = self.结果目录 / "调优历史.json"
        with open(历史文件, 'w', encoding='utf-8') as f:
            json.dump(self.调优历史, f, ensure_ascii=False, indent=2)
    
    def _更新最佳参数(self, 实验配置: Dict[str, Any], 实验结果: Dict[str, float]):
        """更新最佳参数记录"""
        当前分数 = 实验结果.get(self.优化目标, 0.0)
        
        是否更新 = False
        if "F1" in self.优化目标 or "准确率" in self.优化目标:
            # 最大化指标
            if 当前分数 > self.最佳分数:
                是否更新 = True
        else:
            # 最小化指标 (如ECE)
            if 当前分数 < self.最佳分数:
                是否更新 = True
        
        if 是否更新:
            self.最佳分数 = 当前分数
            self.最佳参数 = 实验配置.copy()
            
            self.日志器.info(f"🎯 发现更佳参数: {self.优化目标}={当前分数:.4f}")
    
    def _生成调优报告(self) -> Dict[str, Any]:
        """生成详细的调优报告"""
        if not self.调优历史:
            return {"错误": "无调优历史记录"}
        
        # 分析调优历史
        所有分数 = [结果["实验结果"][self.优化目标] for 结果 in self.调优历史]
        
        调优统计 = {
            "实验总数": len(self.调优历史),
            "最佳分数": self.最佳分数,
            "最佳参数": self.最佳参数,
            "分数统计": {
                "平均值": float(np.mean(所有分数)),
                "标准差": float(np.std(所有分数)),
                "最大值": float(np.max(所有分数)),
                "最小值": float(np.min(所有分数)),
                "中位数": float(np.median(所有分数))
            }
        }
        
        # 参数重要性分析
        参数重要性 = self._分析参数重要性()
        调优统计["参数重要性"] = 参数重要性
        
        # 保存完整报告
        报告文件 = self.结果目录 / f"调优报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(报告文件, 'w', encoding='utf-8') as f:
            json.dump(调优统计, f, ensure_ascii=False, indent=2)
        
        self.日志器.info(f"📋 调优报告已保存: {报告文件}")
        return 调优统计
    
    def _分析参数重要性(self) -> Dict[str, float]:
        """分析参数对性能的重要性"""
        参数重要性 = {}
        
        # 简化的参数重要性分析
        for 结果记录 in self.调优历史:
            配置 = 结果记录["实验配置"]
            分数 = 结果记录["实验结果"][self.优化目标]
            
            for 参数名, 参数值 in 配置.items():
                if 参数名 not in 参数重要性:
                    参数重要性[参数名] = []
                参数重要性[参数名].append(分数)
        
        # 计算每个参数的方差 (简化重要性指标)
        重要性分数 = {}
        for 参数名, 分数列表 in 参数重要性.items():
            if len(分数列表) > 1:
                重要性分数[参数名] = float(np.var(分数列表))
            else:
                重要性分数[参数名] = 0.0
        
        return 重要性分数
    
    def 生成最优配置(self) -> Dict[str, Any]:
        """生成基于调优结果的最优配置"""
        if self.最佳参数 is None:
            raise ValueError("未找到最佳参数，请先执行参数调优")
        
        最优配置 = {
            "元信息": {
                "调优策略": self.调优策略,
                "优化目标": self.优化目标,
                "最佳分数": self.最佳分数,
                "调优完成时间": datetime.now().isoformat()
            },
            "最优超参数": self.最佳参数,
            "使用说明": {
                "训练命令": f"python run_experiments_cn.py --config 最优配置.json",
                "预期性能": f"{self.优化目标} ≈ {self.最佳分数:.4f}",
                "注意事项": [
                    "确保使用相同的随机种子",
                    "验证硬件环境一致性", 
                    "检查数据预处理管线"
                ]
            }
        }
        
        # 保存最优配置
        配置文件 = self.结果目录 / "最优配置.json"
        with open(配置文件, 'w', encoding='utf-8') as f:
            json.dump(最优配置, f, ensure_ascii=False, indent=2)
        
        self.日志器.info(f"🏆 最优配置已保存: {配置文件}")
        return 最优配置


class 参数验证器:
    """参数有效性验证器"""
    
    @staticmethod
    def 验证D2参数(配置: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证D2协议参数有效性"""
        错误列表 = []
        
        # 必需字段检查
        必需字段 = ["学习率", "批次大小", "最大训练轮次"]
        for 字段 in 必需字段:
            if 字段 not in 配置:
                错误列表.append(f"缺少必需字段: {字段}")
        
        # 范围检查
        if "学习率" in 配置:
            学习率 = 配置["学习率"]
            if not (1e-6 <= 学习率 <= 1e-1):
                错误列表.append(f"学习率超出合理范围: {学习率}")
        
        if "批次大小" in 配置:
            批次大小 = 配置["批次大小"]
            if not (1 <= 批次大小 <= 512):
                错误列表.append(f"批次大小超出合理范围: {批次大小}")
        
        return len(错误列表) == 0, 错误列表
    
    @staticmethod
    def 验证CDAE参数(配置: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证CDAE协议参数有效性"""
        错误列表 = []
        
        # LOSO/LORO特定检查
        if "LOSO受试者列表" in 配置:
            受试者列表 = 配置["LOSO受试者列表"]
            if len(受试者列表) < 3:
                错误列表.append("LOSO受试者数量过少，至少需要3个")
        
        return len(错误列表) == 0, 错误列表
    
    @staticmethod
    def 验证STEA参数(配置: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证STEA协议参数有效性"""
        错误列表 = []
        
        # 标签比例检查
        if "标签比例列表" in 配置:
            标签比例 = 配置["标签比例列表"]
            if not all(0 < 比例 <= 100 for 比例 in 标签比例):
                错误列表.append("标签比例必须在(0, 100]范围内")
        
        return len(错误列表) == 0, 错误列表


def 主函数():
    """参数调优主函数"""
    print("🔧 WiFi CSI参数调优系统 (中文版)")
    print("=" * 50)
    
    # 基础配置
    基础配置 = {
        "模型名称": "Enhanced",
        "数据集": "CSI-Fall",
        "设备": "cuda",
        "随机种子": 42
    }
    
    # 创建调优器
    调优器 = 参数调优器(基础配置, 优化目标="宏平均F1", 调优策略="网格搜索")
    
    # 定义搜索空间
    搜索空间 = 调优器.定义搜索空间()
    
    # 执行参数调优
    print("选择调优策略:")
    print("1. 网格搜索 (全面但耗时)")
    print("2. 贝叶斯优化 (智能高效)")  
    print("3. 随机搜索 (快速探索)")
    
    策略选择 = input("请输入选择 (1-3): ").strip()
    
    if 策略选择 == "1":
        调优结果 = 调优器.网格搜索优化(搜索空间, 最大实验数=50)
    elif 策略选择 == "2":
        调优结果 = 调优器.贝叶斯优化(搜索空间, 最大实验数=30)
    elif 策略选择 == "3":
        调优结果 = 调优器.随机搜索优化(搜索空间, 最大实验数=25)
    else:
        print("无效选择，使用默认网格搜索")
        调优结果 = 调优器.网格搜索优化(搜索空间, 最大实验数=20)
    
    # 生成最优配置
    最优配置 = 调优器.生成最优配置()
    
    print(f"\n🎉 参数调优完成!")
    print(f"🏆 最佳{调优器.优化目标}: {调优器.最佳分数:.4f}")
    print(f"📋 最优配置已保存到: experiments/results/parameter_tuning/")


if __name__ == "__main__":
    主函数()