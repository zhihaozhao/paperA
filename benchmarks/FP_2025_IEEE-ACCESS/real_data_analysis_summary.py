#!/usr/bin/env python3
"""
真实数据统计分析脚本
统计每个图表和表格的真实论文支撑数量
严格基于prisma_data.csv，绝不编造数据
"""

import csv
import re
from collections import Counter, defaultdict

def analyze_real_data():
    """分析真实数据，统计支撑数量"""
    try:
        with open('/workspace/benchmarks/docs/prisma_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print("=== 真实数据统计分析报告 ===")
        print(f"数据源: prisma_data.csv ({len(lines)} 总行数)")
        
        # 统计相关论文
        relevant_papers = []
        vision_papers = []
        robotics_papers = []
        challenges_papers = []
        
        for i, line in enumerate(lines[1:], 1):  # 跳过header
            if line.strip():
                parts = line.split(',')
                if len(parts) > 0 and parts[0].strip().lower() == 'y':
                    paper_data = {
                        'line_num': i,
                        'title': parts[2] if len(parts) > 2 else '',
                        'year': parts[4] if len(parts) > 4 else '',
                        'learning_algo': parts[12] if len(parts) > 12 else '',
                        'locomotion': parts[13] if len(parts) > 13 else '',
                        'challenges': parts[15] if len(parts) > 15 else '',
                        'full_line': line[:200] + "..." if len(line) > 200 else line
                    }
                    relevant_papers.append(paper_data)
                    
                    # 分类统计
                    title_lower = paper_data['title'].lower()
                    algo_lower = paper_data['learning_algo'].lower()
                    loco_lower = paper_data['locomotion'].lower()
                    challenge_lower = paper_data['challenges'].lower()
                    
                    # 视觉算法论文
                    if any(keyword in title_lower + algo_lower for keyword in 
                           ['yolo', 'rcnn', 'r-cnn', 'cnn', 'detection', 'vision', 'deep', 'neural']):
                        vision_papers.append(paper_data)
                    
                    # 机器人控制论文  
                    if any(keyword in title_lower + loco_lower for keyword in 
                           ['robot', 'motion', 'control', 'planning', 'navigation', 'harvesting', 'picking']):
                        robotics_papers.append(paper_data)
                    
                    # 挑战/趋势论文
                    if any(keyword in title_lower + challenge_lower for keyword in 
                           ['challenge', 'problem', 'limitation', 'future', 'trend', 'review']):
                        challenges_papers.append(paper_data)
        
        print(f"\n📊 数据统计结果:")
        print(f"总相关论文: {len(relevant_papers)}")
        print(f"视觉算法论文: {len(vision_papers)}")  
        print(f"机器人控制论文: {len(robotics_papers)}")
        print(f"挑战趋势论文: {len(challenges_papers)}")
        
        # 年份分布
        years = []
        for paper in relevant_papers:
            try:
                year = int(float(paper['year']))
                if 2000 <= year <= 2025:
                    years.append(year)
            except:
                continue
        
        year_dist = Counter(years)
        print(f"\n📅 年份分布:")
        for year in sorted(year_dist.keys()):
            print(f"  {year}: {year_dist[year]} papers")
        
        # 具体论文示例
        print(f"\n📝 视觉算法论文示例 (前5篇):")
        for i, paper in enumerate(vision_papers[:5], 1):
            print(f"  {i}. {paper['title'][:80]}... ({paper['year']})")
        
        print(f"\n🤖 机器人控制论文示例 (前5篇):")
        for i, paper in enumerate(robotics_papers[:5], 1):
            print(f"  {i}. {paper['title'][:80]}... ({paper['year']})")
            
        print(f"\n⚠️  挑战趋势论文示例 (前5篇):")
        for i, paper in enumerate(challenges_papers[:5], 1):
            print(f"  {i}. {paper['title'][:80]}... ({paper['year']})")
        
        return {
            'total_papers': len(relevant_papers),
            'vision_papers': len(vision_papers),
            'robotics_papers': len(robotics_papers), 
            'challenges_papers': len(challenges_papers),
            'year_distribution': dict(year_dist),
            'vision_examples': vision_papers[:10],
            'robotics_examples': robotics_papers[:10],
            'challenges_examples': challenges_papers[:10]
        }
        
    except Exception as e:
        print(f"数据分析错误: {e}")
        return None

if __name__ == "__main__":
    results = analyze_real_data()