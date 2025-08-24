#!/usr/bin/env python3
"""
真实数据分析脚本 - 基于prisma_data.csv生成IEEE Access论文所需的3个图表
严格遵守学术诚信原则：所有数据必须来自真实数据源
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_real_data():
    """加载真实的prisma数据"""
    try:
        # 读取CSV文件，处理编码问题
        df = pd.read_csv('../../docs/prisma_data.csv', encoding='utf-8')
        print(f"总数据量: {len(df)}")
        
        # 筛选相关论文 (relevant = 'y')
        relevant_df = df[df['relevant'] == 'y'].copy()
        print(f"相关论文数量: {len(relevant_df)}")
        
        return relevant_df
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None

def analyze_vision_algorithms(df):
    """分析视觉算法分布 - Figure 4 数据准备"""
    print("\n=== 视觉算法分析 (Figure 4) ===")
    
    # 分析Learning Algorithm列
    algorithms = []
    for idx, row in df.iterrows():
        algo = str(row.get('Learning Algorithm', '')).lower()
        if pd.notna(row.get('Learning Algorithm')) and algo != 'nan':
            algorithms.append(algo)
    
    print(f"有效算法记录: {len(algorithms)}")
    
    # 算法分类
    algorithm_categories = {
        'YOLO': [],
        'R-CNN': [],
        'Hybrid': [],
        'Traditional': []
    }
    
    for algo in algorithms:
        if 'yolo' in algo:
            algorithm_categories['YOLO'].append(algo)
        elif 'cnn' in algo or 'rcnn' in algo or 'faster' in algo:
            algorithm_categories['R-CNN'].append(algo)
        elif 'hybrid' in algo or 'ensemble' in algo or 'fusion' in algo:
            algorithm_categories['Hybrid'].append(algo)
        else:
            algorithm_categories['Traditional'].append(algo)
    
    # 打印分析结果
    for category, algos in algorithm_categories.items():
        print(f"{category}: {len(algos)} papers")
        if algos:
            print(f"  Examples: {list(set(algos))[:3]}")
    
    return algorithm_categories

def analyze_motion_control(df):
    """分析运动控制算法 - Figure 9 数据准备"""
    print("\n=== 运动控制分析 (Figure 9) ===")
    
    # 分析Locomotion列
    locomotion_data = []
    for idx, row in df.iterrows():
        locomotion = str(row.get('Locomotion', '')).lower()
        if pd.notna(row.get('Locomotion')) and locomotion != 'nan':
            locomotion_data.append(locomotion)
    
    print(f"有效运动控制记录: {len(locomotion_data)}")
    
    # 运动控制分类
    control_categories = {
        'RL_Based': [],
        'Classical_Geometric': [],
        'Vision_Guided': [],
        'Hybrid_Systems': []
    }
    
    for locomotion in locomotion_data:
        if 'reinforcement' in locomotion or 'rl' in locomotion or 'ddpg' in locomotion:
            control_categories['RL_Based'].append(locomotion)
        elif 'geometric' in locomotion or 'classical' in locomotion or 'traditional' in locomotion:
            control_categories['Classical_Geometric'].append(locomotion)
        elif 'vision' in locomotion or 'visual' in locomotion or 'camera' in locomotion:
            control_categories['Vision_Guided'].append(locomotion)
        else:
            control_categories['Hybrid_Systems'].append(locomotion)
    
    # 打印分析结果
    for category, methods in control_categories.items():
        print(f"{category}: {len(methods)} papers")
        if methods:
            print(f"  Examples: {list(set(methods))[:3]}")
    
    return control_categories

def analyze_challenges_trends(df):
    """分析挑战和趋势 - Figure 10 数据准备"""
    print("\n=== 挑战趋势分析 (Figure 10) ===")
    
    # 分析challenges列
    challenges_data = []
    for idx, row in df.iterrows():
        challenges = str(row.get('challenges', '')).lower()
        if pd.notna(row.get('challenges')) and challenges != 'nan':
            challenges_data.append(challenges)
    
    print(f"有效挑战记录: {len(challenges_data)}")
    
    # 挑战分类
    challenge_categories = {
        'Cost_Effectiveness': [],
        'Environmental_Robustness': [],
        'Technical_Integration': [],
        'Deployment_Barriers': []
    }
    
    for challenge in challenges_data:
        if 'cost' in challenge or 'expensive' in challenge or 'economic' in challenge:
            challenge_categories['Cost_Effectiveness'].append(challenge)
        elif 'environment' in challenge or 'weather' in challenge or 'lighting' in challenge:
            challenge_categories['Environmental_Robustness'].append(challenge)
        elif 'integration' in challenge or 'coordination' in challenge or 'fusion' in challenge:
            challenge_categories['Technical_Integration'].append(challenge)
        else:
            challenge_categories['Deployment_Barriers'].append(challenge)
    
    # 打印分析结果
    for category, challenges in challenge_categories.items():
        print(f"{category}: {len(challenges)} papers")
        if challenges:
            print(f"  Examples: {list(set(challenges))[:2]}")
    
    return challenge_categories

def analyze_temporal_trends(df):
    """分析时间趋势"""
    print("\n=== 时间趋势分析 ===")
    
    # 分析Publication Year
    years = []
    for idx, row in df.iterrows():
        year = row.get('Publication Year')
        if pd.notna(year) and isinstance(year, (int, float)):
            years.append(int(year))
    
    print(f"有效年份记录: {len(years)}")
    year_counts = Counter(years)
    print(f"年份分布: {dict(sorted(year_counts.items()))}")
    
    return year_counts

def main():
    """主分析函数"""
    print("开始分析真实数据源: prisma_data.csv")
    print("=" * 50)
    
    # 加载数据
    df = load_real_data()
    if df is None:
        return
    
    # 执行所有分析
    vision_results = analyze_vision_algorithms(df)
    motion_results = analyze_motion_control(df)
    challenge_results = analyze_challenges_trends(df)
    temporal_results = analyze_temporal_trends(df)
    
    print("\n" + "=" * 50)
    print("分析完成！数据准备就绪，可以生成图表。")
    
    return {
        'vision': vision_results,
        'motion': motion_results,
        'challenges': challenge_results,
        'temporal': temporal_results,
        'total_papers': len(df)
    }

if __name__ == "__main__":
    results = main()