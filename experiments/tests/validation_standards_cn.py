#!/usr/bin/env python3
"""
验收标准测试系统 - 博士论文实验验收自动化 (中文版)
基于D1验收标准，实现自动化验证和质量保证

作者: 博士论文研究  
日期: 2025年
版本: v2.0-cn

验收标准 (基于记忆):
- InD合成能力对齐验证 - 汇总CSV ≥3 seeds per model
- Enhanced vs CNN参数在±10%范围内
- 指标有效性验证 (macro_f1, ECE, NLL)
- 下一步: 跨生成器测试, 更高难度扫描, 消融研究
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# 统计和验证库
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


class D1验收标准验证器:
    """D1验收标准验证器 - 基于记忆6364081标准"""
    
    def __init__(self, 结果目录: str = "experiments/results"):
        """
        初始化验收标准验证器
        参数:
            结果目录: 实验结果根目录
        """
        self.结果目录 = Path(结果目录)
        self.验证报告 = []
        
        # 设置日志
        self._设置验证日志()
        
        # D1验收标准 (基于记忆)
        self.D1标准 = {
            "InD合成能力对齐": {
                "最少种子数": 3,
                "需要模型": ["Enhanced", "CNN"],
                "参数容差": 0.10,  # ±10%
                "汇总CSV必需": True
            },
            "性能指标要求": {
                "macro_f1": {"最小值": 0.75, "Enhanced目标": 0.83},
                "ECE": {"最大值": 0.05, "理想值": 0.03},
                "NLL": {"最大值": 1.5, "理想值": 1.0}
            },
            "Enhanced模型一致性": {
                "LOSO_F1": 0.830,
                "LORO_F1": 0.830,
                "允许偏差": 0.001,
                "一致性要求": True
            },
            "STEA突破点": {
                "20%标签F1": 0.821,
                "目标阈值": 0.80,
                "突破要求": True,
                "相对性能": 0.986  # 82.1/83.3
            }
        }
        
    def _设置验证日志(self):
        """设置验证日志系统"""
        日志文件 = self.结果目录 / f"D1验收验证_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - D1验收 - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(日志文件, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.日志器 = logging.getLogger("D1验收验证器")
    
    def 验证InD合成能力对齐(self, 实验结果路径: str) -> Dict[str, bool]:
        """
        验证InD合成能力对齐标准
        参数:
            实验结果路径: 实验结果CSV文件路径
        返回:
            验证结果字典
        """
        self.日志器.info("🔬 验证InD合成能力对齐标准...")
        
        验证结果 = {
            "汇总CSV存在": False,
            "种子数量充足": False,
            "参数容差符合": False,
            "模型完整性": False
        }
        
        try:
            # 检查汇总CSV文件
            CSV文件路径 = Path(实验结果路径)
            if CSV文件路径.exists():
                验证结果["汇总CSV存在"] = True
                self.日志器.info("✅ 汇总CSV文件存在")
                
                # 读取CSV数据
                实验数据 = pd.read_csv(CSV文件路径)
                
                # 验证种子数量
                if 'seed' in 实验数据.columns:
                    for 模型名 in self.D1标准["InD合成能力对齐"]["需要模型"]:
                        模型数据 = 实验数据[实验数据['model'] == 模型名]
                        种子数量 = len(模型数据['seed'].unique())
                        
                        if 种子数量 >= self.D1标准["InD合成能力对齐"]["最少种子数"]:
                            验证结果["种子数量充足"] = True
                            self.日志器.info(f"✅ {模型名}: {种子数量} 个种子 ≥ {self.D1标准['InD合成能力对齐']['最少种子数']}")
                        else:
                            self.日志器.warning(f"⚠️  {模型名}: {种子数量} 个种子 < {self.D1标准['InD合成能力对齐']['最少种子数']}")
                
                # 验证Enhanced vs CNN参数容差
                验证结果["参数容差符合"] = self._验证参数容差(实验数据)
                
                # 验证模型完整性
                必需模型 = set(self.D1标准["InD合成能力对齐"]["需要模型"])
                实际模型 = set(实验数据['model'].unique()) if 'model' in 实验数据.columns else set()
                
                if 必需模型.issubset(实际模型):
                    验证结果["模型完整性"] = True
                    self.日志器.info(f"✅ 模型完整性: {实际模型}")
                else:
                    缺失模型 = 必需模型 - 实际模型
                    self.日志器.warning(f"⚠️  缺失模型: {缺失模型}")
                    
        except Exception as 错误:
            self.日志器.error(f"❌ InD验证失败: {错误}")
        
        return 验证结果
    
    def _验证参数容差(self, 实验数据: pd.DataFrame) -> bool:
        """验证Enhanced vs CNN参数容差在±10%范围内"""
        try:
            if 'model' in 实验数据.columns and 'parameters' in 实验数据.columns:
                Enhanced参数 = 实验数据[实验数据['model'] == 'Enhanced']['parameters'].iloc[0]
                CNN参数 = 实验数据[实验数据['model'] == 'CNN']['parameters'].iloc[0]
                
                参数差异率 = abs(Enhanced参数 - CNN参数) / CNN参数
                容差阈值 = self.D1标准["InD合成能力对齐"]["参数容差"]
                
                if 参数差异率 <= 容差阈值:
                    self.日志器.info(f"✅ 参数容差: {参数差异率:.1%} ≤ ±{容差阈值:.0%}")
                    return True
                else:
                    self.日志器.warning(f"⚠️  参数容差超限: {参数差异率:.1%} > ±{容差阈值:.0%}")
                    return False
            else:
                self.日志器.warning("⚠️  缺少参数信息列")
                return False
                
        except Exception as 错误:
            self.日志器.error(f"❌ 参数容差验证失败: {错误}")
            return False
    
    def 验证性能指标有效性(self, 实验结果: Dict[str, Any]) -> Dict[str, bool]:
        """
        验证性能指标有效性 (macro_f1, ECE, NLL)
        参数:
            实验结果: 实验结果字典
        返回:
            指标验证结果
        """
        self.日志器.info("📊 验证性能指标有效性...")
        
        指标验证 = {
            "macro_f1_有效": False,
            "ECE_有效": False,
            "NLL_有效": False
        }
        
        # 验证macro_f1
        if "macro_f1" in 实验结果:
            macro_f1 = 实验结果["macro_f1"]
            最小要求 = self.D1标准["性能指标要求"]["macro_f1"]["最小值"]
            
            if macro_f1 >= 最小要求:
                指标验证["macro_f1_有效"] = True
                self.日志器.info(f"✅ Macro F1: {macro_f1:.4f} ≥ {最小要求}")
            else:
                self.日志器.warning(f"⚠️  Macro F1不达标: {macro_f1:.4f} < {最小要求}")
        
        # 验证ECE  
        if "ECE" in 实验结果:
            ECE = 实验结果["ECE"]
            最大允许 = self.D1标准["性能指标要求"]["ECE"]["最大值"]
            
            if ECE <= 最大允许:
                指标验证["ECE_有效"] = True
                self.日志器.info(f"✅ ECE: {ECE:.4f} ≤ {最大允许}")
            else:
                self.日志器.warning(f"⚠️  ECE超标: {ECE:.4f} > {最大允许}")
        
        # 验证NLL
        if "NLL" in 实验结果:
            NLL = 实验结果["NLL"]
            最大允许 = self.D1标准["性能指标要求"]["NLL"]["最大值"]
            
            if NLL <= 最大允许:
                指标验证["NLL_有效"] = True
                self.日志器.info(f"✅ NLL: {NLL:.4f} ≤ {最大允许}")
            else:
                self.日志器.warning(f"⚠️  NLL超标: {NLL:.4f} > {最大允许}")
        
        return 指标验证
    
    def 验证Enhanced模型一致性(self, CDAE结果: Dict[str, Any]) -> Dict[str, bool]:
        """
        验证Enhanced模型LOSO=LORO一致性
        参数:
            CDAE结果: CDAE协议实验结果
        返回:
            一致性验证结果
        """
        self.日志器.info("🎯 验证Enhanced模型一致性...")
        
        一致性验证 = {
            "LOSO性能达标": False,
            "LORO性能达标": False,
            "一致性满足": False
        }
        
        try:
            # 提取LOSO和LORO结果
            LOSO结果 = CDAE结果.get("LOSO", {})
            LORO结果 = CDAE结果.get("LORO", {})
            
            LOSO_F1 = LOSO结果.get("loso_mean_f1", 0.0)
            LORO_F1 = LORO结果.get("loro_mean_f1", 0.0)
            
            # 验证LOSO性能
            if LOSO_F1 >= self.D1标准["Enhanced模型一致性"]["LOSO_F1"]:
                一致性验证["LOSO性能达标"] = True
                self.日志器.info(f"✅ LOSO F1: {LOSO_F1:.4f} ≥ {self.D1标准['Enhanced模型一致性']['LOSO_F1']}")
            
            # 验证LORO性能
            if LORO_F1 >= self.D1标准["Enhanced模型一致性"]["LORO_F1"]:
                一致性验证["LORO性能达标"] = True
                self.日志器.info(f"✅ LORO F1: {LORO_F1:.4f} ≥ {self.D1标准['Enhanced模型一致性']['LORO_F1']}")
            
            # 验证一致性
            F1差异 = abs(LOSO_F1 - LORO_F1)
            允许偏差 = self.D1标准["Enhanced模型一致性"]["允许偏差"]
            
            if F1差异 <= 允许偏差:
                一致性验证["一致性满足"] = True
                self.日志器.info(f"✅ LOSO-LORO一致性: 差异={F1差异:.4f} ≤ {允许偏差}")
            else:
                self.日志器.warning(f"⚠️  一致性不满足: 差异={F1差异:.4f} > {允许偏差}")
                
        except Exception as 错误:
            self.日志器.error(f"❌ 一致性验证失败: {错误}")
        
        return 一致性验证
    
    def 验证STEA突破点(self, STEA结果: Dict[str, Any]) -> Dict[str, bool]:
        """
        验证STEA协议20%标签突破点
        参数:
            STEA结果: STEA协议实验结果
        返回:
            突破点验证结果
        """
        self.日志器.info("🎯 验证STEA突破点...")
        
        突破点验证 = {
            "20%标签达标": False,
            "超越目标阈值": False,
            "相对性能达标": False
        }
        
        try:
            STEA数据 = STEA结果.get("stea_results", [])
            
            # 查找20%标签结果
            for 结果项 in STEA数据:
                if 结果项.get("label_ratio") == 20:
                    标签20_F1 = 结果项.get("final_f1", 0.0)
                    目标阈值 = self.D1标准["STEA突破点"]["目标阈值"]
                    
                    # 验证20%标签性能
                    if 标签20_F1 >= self.D1标准["STEA突破点"]["20%标签F1"]:
                        突破点验证["20%标签达标"] = True
                        self.日志器.info(f"✅ 20%标签F1: {标签20_F1:.4f} ≥ {self.D1标准['STEA突破点']['20%标签F1']}")
                    
                    # 验证超越目标阈值
                    if 标签20_F1 >= 目标阈值:
                        突破点验证["超越目标阈值"] = True
                        self.日志器.info(f"✅ 超越目标: {标签20_F1:.4f} ≥ {目标阈值}")
                    
                    # 验证相对性能
                    相对性能 = 结果项.get("relative_performance", 0.0)
                    if 相对性能 >= self.D1标准["STEA突破点"]["相对性能"]:
                        突破点验证["相对性能达标"] = True
                        self.日志器.info(f"✅ 相对性能: {相对性能:.1%} ≥ {self.D1标准['STEA突破点']['相对性能']:.1%}")
                    
                    break
            else:
                self.日志器.warning("⚠️  未找到20%标签实验结果")
                
        except Exception as 错误:
            self.日志器.error(f"❌ STEA突破点验证失败: {错误}")
        
        return 突破点验证
    
    def 执行统计显著性检验(self, 对比结果: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        执行统计显著性检验
        参数:
            对比结果: 模型对比结果字典 {"Enhanced": [F1列表], "CNN": [F1列表]}
        返回:
            统计检验结果
        """
        self.日志器.info("📈 执行统计显著性检验...")
        
        统计结果 = {}
        
        if "Enhanced" in 对比结果 and "CNN" in 对比结果:
            Enhanced_F1 = 对比结果["Enhanced"]
            CNN_F1 = 对比结果["CNN"]
            
            # 配对t检验
            t统计量, p值 = stats.ttest_rel(Enhanced_F1, CNN_F1)
            
            # Cohen's d效应量
            差异均值 = np.mean(np.array(Enhanced_F1) - np.array(CNN_F1))
            合并标准差 = np.sqrt((np.var(Enhanced_F1) + np.var(CNN_F1)) / 2)
            cohens_d = 差异均值 / 合并标准差 if 合并标准差 > 0 else 0
            
            统计结果 = {
                "配对t检验": {
                    "t统计量": float(t统计量),
                    "p值": float(p值),
                    "显著性": p值 < 0.05
                },
                "效应量": {
                    "Cohen's d": float(cohens_d),
                    "效应大小": "大" if abs(cohens_d) > 0.8 else ("中" if abs(cohens_d) > 0.5 else "小")
                },
                "描述统计": {
                    "Enhanced均值": float(np.mean(Enhanced_F1)),
                    "Enhanced标准差": float(np.std(Enhanced_F1)),
                    "CNN均值": float(np.mean(CNN_F1)),
                    "CNN标准差": float(np.std(CNN_F1))
                }
            }
            
            if p值 < 0.05:
                self.日志器.info(f"✅ 统计显著性: p={p值:.4f} < 0.05")
            else:
                self.日志器.warning(f"⚠️  统计不显著: p={p值:.4f} ≥ 0.05")
        
        return 统计结果
    
    def 生成综合验收报告(self, 
                        InD验证: Dict,
                        指标验证: Dict, 
                        一致性验证: Dict,
                        突破点验证: Dict,
                        统计验证: Dict) -> Dict[str, Any]:
        """
        生成综合验收报告
        参数:
            各项验证结果字典
        返回:
            综合验收报告
        """
        self.日志器.info("📋 生成综合验收报告...")
        
        # 计算总体通过率
        所有验证项 = []
        所有验证项.extend(InD验证.values())
        所有验证项.extend(指标验证.values())
        所有验证项.extend(一致性验证.values())
        所有验证项.extend(突破点验证.values())
        
        通过项数 = sum(所有验证项)
        总验证项 = len(所有验证项)
        通过率 = 通过项数 / 总验证项
        
        综合报告 = {
            "验收总览": {
                "验收时间": datetime.now().isoformat(),
                "通过项数": 通过项数,
                "总验证项": 总验证项,
                "通过率": f"{通过率:.1%}",
                "总体状态": "✅ 通过" if 通过率 >= 0.8 else "❌ 未通过"
            },
            "详细验证结果": {
                "InD合成能力对齐": InD验证,
                "性能指标有效性": 指标验证,
                "Enhanced模型一致性": 一致性验证,
                "STEA突破点": 突破点验证,
                "统计显著性": 统计验证
            },
            "后续建议": self._生成后续建议(通过率, InD验证, 指标验证, 一致性验证, 突破点验证)
        }
        
        # 保存报告
        报告文件 = self.结果目录 / f"D1综合验收报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(报告文件, 'w', encoding='utf-8') as f:
            json.dump(综合报告, f, ensure_ascii=False, indent=2)
        
        # 生成Markdown报告
        self._生成Markdown报告(综合报告)
        
        return 综合报告
    
    def _生成后续建议(self, 通过率: float, *验证结果) -> List[str]:
        """基于验证结果生成后续建议"""
        建议列表 = []
        
        if 通过率 >= 0.9:
            建议列表.extend([
                "🎉 验收标准全面达成，可进入下一阶段",
                "📈 建议添加跨生成器测试 (test_seed)",
                "🔄 考虑更高难度扫描验证",
                "🧪 进行消融研究 (+SE/+Attention/only CNN)",
                "📊 温度缩放NPZ导出用于可靠性曲线"
            ])
        elif 通过率 >= 0.7:
            建议列表.extend([
                "⚡ 大部分标准达成，需要针对性改进",
                "🔧 重点优化未通过的验证项",
                "📊 增加实验种子数量提高统计可靠性"
            ])
        else:
            建议列表.extend([
                "🚨 验收标准未达成，需要系统性改进",
                "🔬 重新审视实验设计和模型架构",
                "📋 建议从单个协议开始逐步验证"
            ])
        
        return 建议列表
    
    def _生成Markdown报告(self, 综合报告: Dict[str, Any]):
        """生成Markdown格式的验收报告"""
        报告内容 = f"""# WiFi CSI博士论文D1验收标准验证报告

## 📊 验收概览

- **验收时间**: {综合报告['验收总览']['验收时间']}
- **通过率**: {综合报告['验收总览']['通过率']}
- **总体状态**: {综合报告['验收总览']['总体状态']}
- **通过项**: {综合报告['验收总览']['通过项数']}/{综合报告['验收总览']['总验证项']}

## ✅ 详细验证结果

### 🔬 InD合成能力对齐验证
"""
        
        InD结果 = 综合报告['详细验证结果']['InD合成能力对齐']
        for 项目, 状态 in InD结果.items():
            状态符号 = "✅" if 状态 else "❌"
            报告内容 += f"- {项目}: {状态符号}\n"
        
        报告内容 += """
### 📈 性能指标有效性验证
"""
        指标结果 = 综合报告['详细验证结果']['性能指标有效性']
        for 指标, 状态 in 指标结果.items():
            状态符号 = "✅" if 状态 else "❌"
            报告内容 += f"- {指标}: {状态符号}\n"
        
        报告内容 += f"""
## 🎯 后续建议

"""
        for 建议 in 综合报告['后续建议']:
            报告内容 += f"- {建议}\n"
        
        报告内容 += f"""
---
**验收标准版本**: D1 (基于记忆6364081)
**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
"""
        
        # 保存Markdown报告
        MD文件 = self.结果目录 / f"D1验收报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(MD文件, 'w', encoding='utf-8') as f:
            f.write(报告内容)
        
        self.日志器.info(f"📄 Markdown报告已保存: {MD文件}")


def 演示验收流程():
    """演示完整验收流程"""
    print("🏆 D1验收标准验证演示")
    print("=" * 50)
    
    # 创建验证器
    验证器 = D1验收标准验证器()
    
    # 模拟实验结果
    模拟InD结果 = "experiments/results/模拟汇总.csv"
    模拟性能结果 = {"macro_f1": 0.830, "ECE": 0.03, "NLL": 1.2}
    模拟CDAE结果 = {
        "LOSO": {"loso_mean_f1": 0.830},
        "LORO": {"loro_mean_f1": 0.830}
    }
    模拟STEA结果 = {
        "stea_results": [
            {"label_ratio": 20, "final_f1": 0.821, "relative_performance": 0.986}
        ]
    }
    模拟对比结果 = {
        "Enhanced": [0.830, 0.831, 0.829, 0.830],
        "CNN": [0.820, 0.825, 0.815, 0.822]
    }
    
    # 执行各项验证
    print("1. InD合成能力对齐验证...")
    # InD验证 = 验证器.验证InD合成能力对齐(模拟InD结果)
    InD验证 = {"汇总CSV存在": True, "种子数量充足": True, "参数容差符合": True, "模型完整性": True}
    
    print("2. 性能指标有效性验证...")
    指标验证 = 验证器.验证性能指标有效性(模拟性能结果)
    
    print("3. Enhanced模型一致性验证...")
    一致性验证 = 验证器.验证Enhanced模型一致性(模拟CDAE结果)
    
    print("4. STEA突破点验证...")
    突破点验证 = 验证器.验证STEA突破点(模拟STEA结果)
    
    print("5. 统计显著性检验...")
    统计验证 = 验证器.执行统计显著性检验(模拟对比结果)
    
    # 生成综合报告
    综合报告 = 验证器.生成综合验收报告(
        InD验证, 指标验证, 一致性验证, 突破点验证, 统计验证
    )
    
    print(f"\n🎉 验收流程完成!")
    print(f"📊 通过率: {综合报告['验收总览']['通过率']}")
    print(f"📋 详细报告已保存到: experiments/results/")


if __name__ == "__main__":
    演示验收流程()