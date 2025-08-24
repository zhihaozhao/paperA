#!/usr/bin/env python3
"""
简化的真实数据分析脚本 - 不依赖pandas
基于prisma_data.csv生成IEEE Access论文所需的3个图表
严格遵守学术诚信原则：所有数据必须来自真实数据源
"""

import csv
import sys
from collections import Counter, defaultdict

def parse_csv_line(line):
    """简单的CSV行解析"""
    # 处理包含逗号的引用字段
    parts = []
    current = ""
    in_quotes = False
    
    for char in line:
        if char == '"' and not in_quotes:
            in_quotes = True
        elif char == '"' and in_quotes:
            in_quotes = False
        elif char == ',' and not in_quotes:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    
    parts.append(current.strip())
    return parts

def load_real_data():
    """加载真实的prisma数据"""
    try:
        relevant_papers = []
        with open('/workspace/benchmarks/docs/prisma_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        print(f"总数据行数: {len(lines)}")
        
        # 获取header
        header_parts = parse_csv_line(lines[0])
        print(f"数据列数: {len(header_parts)}")
        print("列名:", header_parts[:10])  # 打印前10个列名
        
        # 处理数据行
        for i, line in enumerate(lines[1:], 1):
            if line.strip():
                parts = parse_csv_line(line)
                if len(parts) > 0 and parts[0].strip().lower() == 'y':
                    relevant_papers.append({
                        'relevant': parts[0] if len(parts) > 0 else '',
                        'source': parts[1] if len(parts) > 1 else '',
                        'title': parts[2] if len(parts) > 2 else '',
                        'citations': parts[3] if len(parts) > 3 else '',
                        'year': parts[4] if len(parts) > 4 else '',
                        'highly_cited': parts[5] if len(parts) > 5 else '',
                        'publisher': parts[6] if len(parts) > 6 else '',
                        'citation_key': parts[7] if len(parts) > 7 else '',
                        'doc_type': parts[8] if len(parts) > 8 else '',
                        'contribution': parts[9] if len(parts) > 9 else '',
                        'fruit_veg': parts[10] if len(parts) > 10 else '',
                        'data_modality': parts[11] if len(parts) > 11 else '',
                        'learning_algorithm': parts[12] if len(parts) > 12 else '',
                        'locomotion': parts[13] if len(parts) > 13 else '',
                        'performance': parts[14] if len(parts) > 14 else '',
                        'challenges': parts[15] if len(parts) > 15 else '',
                        'authors': parts[16] if len(parts) > 16 else '',
                        'abstract': parts[17] if len(parts) > 17 else ''
                    })
        
        print(f"相关论文数量: {len(relevant_papers)}")
        return relevant_papers
    
    except Exception as e:
        print(f"数据加载错误: {e}")
        return []

def analyze_vision_algorithms(papers):
    """分析视觉算法分布 - Figure 4 数据准备"""
    print("\n=== 视觉算法分析 (Figure 4) ===")
    
    algorithm_categories = {
        'YOLO': [],
        'R-CNN': [],
        'Hybrid': [],
        'Traditional': []
    }
    
    valid_algorithms = 0
    for paper in papers:
        algo = paper.get('learning_algorithm', '').lower().strip()
        if algo and algo != 'nan' and algo != '':
            valid_algorithms += 1
            
            if 'yolo' in algo:
                algorithm_categories['YOLO'].append(paper)
            elif 'cnn' in algo or 'rcnn' in algo or 'faster' in algo:
                algorithm_categories['R-CNN'].append(paper)
            elif 'hybrid' in algo or 'ensemble' in algo or 'fusion' in algo:
                algorithm_categories['Hybrid'].append(paper)
            else:
                algorithm_categories['Traditional'].append(paper)
    
    print(f"有效算法记录: {valid_algorithms}")
    for category, papers_list in algorithm_categories.items():
        print(f"{category}: {len(papers_list)} papers")
        # 显示一些示例
        for i, paper in enumerate(papers_list[:2]):
            print(f"  - {paper.get('title', 'No title')[:60]}...")
    
    return algorithm_categories

def analyze_motion_control(papers):
    """分析运动控制算法 - Figure 9 数据准备"""
    print("\n=== 运动控制分析 (Figure 9) ===")
    
    control_categories = {
        'RL_Based': [],
        'Classical_Geometric': [],
        'Vision_Guided': [],
        'Hybrid_Systems': []
    }
    
    valid_locomotion = 0
    for paper in papers:
        locomotion = paper.get('locomotion', '').lower().strip()
        if locomotion and locomotion != 'nan' and locomotion != '':
            valid_locomotion += 1
            
            if 'reinforcement' in locomotion or 'rl' in locomotion or 'ddpg' in locomotion:
                control_categories['RL_Based'].append(paper)
            elif 'geometric' in locomotion or 'classical' in locomotion or 'traditional' in locomotion:
                control_categories['Classical_Geometric'].append(paper)
            elif 'vision' in locomotion or 'visual' in locomotion or 'camera' in locomotion:
                control_categories['Vision_Guided'].append(paper)
            else:
                control_categories['Hybrid_Systems'].append(paper)
    
    print(f"有效运动控制记录: {valid_locomotion}")
    for category, papers_list in control_categories.items():
        print(f"{category}: {len(papers_list)} papers")
        # 显示一些示例
        for i, paper in enumerate(papers_list[:2]):
            print(f"  - {paper.get('title', 'No title')[:60]}...")
    
    return control_categories

def analyze_challenges_trends(papers):
    """分析挑战和趋势 - Figure 10 数据准备"""
    print("\n=== 挑战趋势分析 (Figure 10) ===")
    
    challenge_categories = {
        'Cost_Effectiveness': [],
        'Environmental_Robustness': [],
        'Technical_Integration': [],
        'Deployment_Barriers': []
    }
    
    valid_challenges = 0
    for paper in papers:
        challenges = paper.get('challenges', '').lower().strip()
        if challenges and challenges != 'nan' and challenges != '':
            valid_challenges += 1
            
            if 'cost' in challenges or 'expensive' in challenges or 'economic' in challenges:
                challenge_categories['Cost_Effectiveness'].append(paper)
            elif 'environment' in challenges or 'weather' in challenges or 'lighting' in challenges:
                challenge_categories['Environmental_Robustness'].append(paper)
            elif 'integration' in challenges or 'coordination' in challenges or 'fusion' in challenges:
                challenge_categories['Technical_Integration'].append(paper)
            else:
                challenge_categories['Deployment_Barriers'].append(paper)
    
    print(f"有效挑战记录: {valid_challenges}")
    for category, papers_list in challenge_categories.items():
        print(f"{category}: {len(papers_list)} papers")
        # 显示一些示例
        for i, paper in enumerate(papers_list[:2]):
            print(f"  - {paper.get('title', 'No title')[:60]}...")
    
    return challenge_categories

def analyze_temporal_trends(papers):
    """分析时间趋势"""
    print("\n=== 时间趋势分析 ===")
    
    year_counts = Counter()
    valid_years = 0
    
    for paper in papers:
        year_str = paper.get('year', '').strip()
        try:
            if year_str and year_str != 'nan':
                year = int(float(year_str))
                if 2000 <= year <= 2025:  # 合理的年份范围
                    year_counts[year] += 1
                    valid_years += 1
        except (ValueError, TypeError):
            continue
    
    print(f"有效年份记录: {valid_years}")
    print("年份分布:")
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]} papers")
    
    return dict(year_counts)

def main():
    """主分析函数"""
    print("开始分析真实数据源: prisma_data.csv")
    print("=" * 50)
    
    # 加载数据
    papers = load_real_data()
    if not papers:
        print("数据加载失败!")
        return
    
    # 执行所有分析
    vision_results = analyze_vision_algorithms(papers)
    motion_results = analyze_motion_control(papers)
    challenge_results = analyze_challenges_trends(papers)
    temporal_results = analyze_temporal_trends(papers)
    
    print("\n" + "=" * 50)
    print("分析完成！数据准备就绪。")
    print(f"总相关论文数: {len(papers)}")
    
    return {
        'vision': vision_results,
        'motion': motion_results,
        'challenges': challenge_results,
        'temporal': temporal_results,
        'total_papers': len(papers)
    }

if __name__ == "__main__":
    results = main()