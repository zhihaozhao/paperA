#!/usr/bin/env python3
"""
机器人控制论文深度分析脚本 - Figure 9数据准备
基于77篇机器人控制论文的真实数据进行技术成果分析
"""

import csv
import re
from collections import Counter, defaultdict

def extract_robotics_performance_data():
    """提取机器人控制相关的性能数据和技术方法"""
    try:
        with open('/workspace/benchmarks/docs/prisma_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        robotics_papers = []
        control_methods = defaultdict(list)
        performance_data = defaultdict(list)
        environments = defaultdict(int)
        
        print("=== 机器人控制论文深度分析 (Figure 9) ===")
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip():
                parts = line.split(',')
                if len(parts) > 0 and parts[0].strip().lower() == 'y':
                    title = parts[2] if len(parts) > 2 else ''
                    year = parts[4] if len(parts) > 4 else ''
                    locomotion = parts[13] if len(parts) > 13 else ''
                    performance = parts[14] if len(parts) > 14 else ''
                    challenges = parts[15] if len(parts) > 15 else ''
                    
                    title_lower = title.lower()
                    loco_lower = locomotion.lower()
                    perf_lower = performance.lower()
                    
                    # 机器人控制相关论文
                    if any(keyword in title_lower + loco_lower for keyword in 
                           ['robot', 'motion', 'control', 'planning', 'navigation', 'harvesting', 'picking', 'manipulator', 'arm']):
                        
                        paper_data = {
                            'title': title,
                            'year': year,
                            'locomotion': locomotion,
                            'performance': performance,
                            'challenges': challenges
                        }
                        robotics_papers.append(paper_data)
                        
                        # 分析控制方法
                        if 'reinforcement' in loco_lower or 'rl' in loco_lower or 'ddpg' in loco_lower:
                            control_methods['RL-Based'].append(paper_data)
                        elif 'geometric' in loco_lower or 'rrt' in loco_lower or 'a*' in loco_lower:
                            control_methods['Traditional-Planning'].append(paper_data)
                        elif 'vision' in loco_lower or 'visual' in loco_lower or 'camera' in loco_lower:
                            control_methods['Vision-Guided'].append(paper_data)
                        elif 'pid' in loco_lower or 'control' in loco_lower:
                            control_methods['Classical-Control'].append(paper_data)
                        else:
                            control_methods['Other/Hybrid'].append(paper_data)
                        
                        # 提取性能数据
                        success_rate = extract_percentage(perf_lower)
                        if success_rate:
                            performance_data['success_rates'].append((success_rate, year, title[:50]))
                        
                        cycle_time = extract_time(perf_lower)
                        if cycle_time:
                            performance_data['cycle_times'].append((cycle_time, year, title[:50]))
                        
                        # 分析环境类型
                        if 'greenhouse' in title_lower or 'controlled' in title_lower:
                            environments['Greenhouse'] += 1
                        elif 'orchard' in title_lower or 'field' in title_lower:
                            environments['Orchard/Field'] += 1
                        elif 'laboratory' in title_lower or 'lab' in title_lower:
                            environments['Laboratory'] += 1
                        else:
                            environments['General'] += 1
        
        print(f"机器人控制论文总数: {len(robotics_papers)}")
        print(f"\n🎯 控制方法分布:")
        for method, papers in control_methods.items():
            print(f"  {method}: {len(papers)} papers")
            if papers:
                print(f"    示例: {papers[0]['title'][:60]}...")
        
        print(f"\n📈 性能数据统计:")
        print(f"  成功率数据: {len(performance_data['success_rates'])} entries")
        print(f"  循环时间数据: {len(performance_data['cycle_times'])} entries")
        
        if performance_data['success_rates']:
            rates = [r[0] for r in performance_data['success_rates']]
            print(f"  成功率范围: {min(rates):.1f}%-{max(rates):.1f}%")
        
        if performance_data['cycle_times']:
            times = [t[0] for t in performance_data['cycle_times']]
            print(f"  循环时间范围: {min(times):.1f}s-{max(times):.1f}s")
        
        print(f"\n🌍 环境分布:")
        for env, count in environments.items():
            print(f"  {env}: {count} papers")
        
        # 年度趋势分析
        year_analysis = defaultdict(lambda: {'papers': 0, 'methods': set()})
        for paper in robotics_papers:
            try:
                year = int(float(paper['year']))
                if 2015 <= year <= 2024:
                    year_analysis[year]['papers'] += 1
                    
                    # 确定主要方法
                    loco_lower = paper['locomotion'].lower()
                    if 'reinforcement' in loco_lower or 'rl' in loco_lower:
                        year_analysis[year]['methods'].add('RL')
                    elif 'vision' in loco_lower:
                        year_analysis[year]['methods'].add('Vision')
                    else:
                        year_analysis[year]['methods'].add('Traditional')
            except:
                continue
        
        print(f"\n📅 年度发展趋势:")
        for year in sorted(year_analysis.keys()):
            data = year_analysis[year]
            methods_str = ', '.join(sorted(data['methods']))
            print(f"  {year}: {data['papers']} papers, 主要方法: [{methods_str}]")
        
        return {
            'total_papers': len(robotics_papers),
            'control_methods': dict(control_methods),
            'performance_data': dict(performance_data),
            'environments': dict(environments),
            'year_trends': dict(year_analysis),
            'sample_papers': robotics_papers[:10]
        }
        
    except Exception as e:
        print(f"机器人分析错误: {e}")
        return None

def extract_percentage(text):
    """从文本中提取百分比数据"""
    patterns = [
        r'(\d+\.?\d*)\s*%',
        r'success rate.{0,20}(\d+\.?\d*)',
        r'accuracy.{0,20}(\d+\.?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                if 0 <= value <= 100:
                    return value
            except:
                continue
    return None

def extract_time(text):
    """从文本中提取时间数据（秒）"""
    patterns = [
        r'(\d+\.?\d*)\s*s(?:ec)?',
        r'(\d+\.?\d*)\s*ms',  # 毫秒转秒
        r'cycle time.{0,20}(\d+\.?\d*)',
        r'processing time.{0,20}(\d+\.?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                if 'ms' in pattern:
                    value = value / 1000  # 转换为秒
                if 0 < value < 1000:  # 合理的时间范围
                    return value
            except:
                continue
    return None

if __name__ == "__main__":
    results = extract_robotics_performance_data()