#!/usr/bin/env python3
"""
批判性挑战分析脚本 - Figure 10数据准备
基于24篇挑战趋势论文进行深度批判性分析
重点关注：研究问题、性能局限、模型缺陷、改进方向
"""

import csv
import re
from collections import Counter, defaultdict

def extract_critical_challenges_data():
    """提取批判性挑战和研究局限数据"""
    try:
        with open('/workspace/benchmarks/docs/prisma_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        challenge_papers = []
        research_problems = defaultdict(list)
        performance_limitations = defaultdict(list)
        model_defects = defaultdict(list)
        future_gaps = defaultdict(list)
        
        print("=== 批判性挑战分析 (Figure 10) ===")
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip():
                parts = line.split(',')
                if len(parts) > 0 and parts[0].strip().lower() == 'y':
                    title = parts[2] if len(parts) > 2 else ''
                    year = parts[4] if len(parts) > 4 else ''
                    challenges = parts[15] if len(parts) > 15 else ''
                    abstract = parts[17] if len(parts) > 17 else ''
                    
                    title_lower = title.lower()
                    challenge_lower = challenges.lower()
                    abstract_lower = abstract.lower()
                    
                    # 批判性分析相关论文
                    is_critical_paper = any(keyword in title_lower + challenge_lower + abstract_lower for keyword in 
                                          ['challenge', 'problem', 'limitation', 'future', 'trend', 'review', 
                                           'survey', 'gap', 'barrier', 'constraint', 'failure', 'difficulty'])
                    
                    if is_critical_paper:
                        paper_data = {
                            'title': title,
                            'year': year,
                            'challenges': challenges,
                            'abstract': abstract[:500] + "..." if len(abstract) > 500 else abstract
                        }
                        challenge_papers.append(paper_data)
                        
                        # 分析研究问题类别
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['cost', 'expensive', 'economic', 'commercial', 'investment']):
                            research_problems['Economic-Viability'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['accuracy', 'precision', 'detection', 'error', 'false']):
                            performance_limitations['Detection-Accuracy'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['environment', 'weather', 'lighting', 'robust', 'variable']):
                            performance_limitations['Environmental-Robustness'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['speed', 'time', 'real-time', 'processing', 'latency']):
                            performance_limitations['Processing-Speed'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['generalization', 'transfer', 'adaptability', 'specific']):
                            model_defects['Limited-Generalization'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['occlusion', 'hidden', 'overlap', 'dense', 'foliage']):
                            model_defects['Occlusion-Handling'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['integration', 'coordination', 'fusion', 'system']):
                            future_gaps['System-Integration'].append(paper_data)
                        
                        if any(keyword in challenge_lower + abstract_lower for keyword in 
                               ['field', 'deployment', 'practical', 'commercial']):
                            future_gaps['Lab-to-Field-Gap'].append(paper_data)
        
        print(f"批判性分析论文总数: {len(challenge_papers)}")
        
        print(f"\n🚨 核心研究问题:")
        for problem, papers in research_problems.items():
            print(f"  {problem}: {len(papers)} papers")
            if papers:
                print(f"    示例: {papers[0]['title'][:60]}...")
        
        print(f"\n⚡ 性能局限分析:")
        for limitation, papers in performance_limitations.items():
            print(f"  {limitation}: {len(papers)} papers")
            if papers:
                print(f"    示例: {papers[0]['title'][:60]}...")
        
        print(f"\n🔧 模型缺陷分析:")
        for defect, papers in model_defects.items():
            print(f"  {defect}: {len(papers)} papers")
            if papers:
                print(f"    示例: {papers[0]['title'][:60]}...")
        
        print(f"\n🔮 未来研究缺口:")
        for gap, papers in future_gaps.items():
            print(f"  {gap}: {len(papers)} papers")
            if papers:
                print(f"    示例: {papers[0]['title'][:60]}...")
        
        # 年度趋势 - 问题持续性分析
        persistent_issues = analyze_persistent_issues(challenge_papers)
        
        # 识别未成熟但活跃的研究方向
        emerging_directions = identify_emerging_directions(challenge_papers)
        
        return {
            'total_papers': len(challenge_papers),
            'research_problems': dict(research_problems),
            'performance_limitations': dict(performance_limitations),
            'model_defects': dict(model_defects),
            'future_gaps': dict(future_gaps),
            'persistent_issues': persistent_issues,
            'emerging_directions': emerging_directions,
            'sample_papers': challenge_papers[:10]
        }
        
    except Exception as e:
        print(f"批判性分析错误: {e}")
        return None

def analyze_persistent_issues(papers):
    """分析持续性问题 - 多年未解决的挑战"""
    issue_by_year = defaultdict(lambda: defaultdict(int))
    
    key_issues = ['cost', 'accuracy', 'environment', 'occlusion', 'commercial', 'field']
    
    for paper in papers:
        try:
            year = int(float(paper['year']))
            if 2015 <= year <= 2024:
                text = (paper['challenges'] + ' ' + paper['abstract']).lower()
                for issue in key_issues:
                    if issue in text:
                        issue_by_year[year][issue] += 1
        except:
            continue
    
    print(f"\n📊 持续性问题分析 (2015-2024):")
    for issue in key_issues:
        years_mentioned = []
        for year in sorted(issue_by_year.keys()):
            if issue_by_year[year][issue] > 0:
                years_mentioned.append(year)
        if len(years_mentioned) >= 3:  # 至少3年提及的持续问题
            print(f"  {issue.title()}: 持续 {len(years_mentioned)} 年 ({years_mentioned[0]}-{years_mentioned[-1]})")
    
    return dict(issue_by_year)

def identify_emerging_directions(papers):
    """识别新兴但未成熟的研究方向"""
    emerging_keywords = [
        'multi-robot', 'swarm', 'distributed', 'federated',
        'explainable', 'interpretable', 'uncertainty', 
        'sim2real', 'digital twin', 'edge computing',
        'sustainability', 'carbon footprint', 'green',
        'human-robot', 'collaborative', 'safety'
    ]
    
    emerging_trends = defaultdict(list)
    
    for paper in papers:
        text = (paper['title'] + ' ' + paper['challenges'] + ' ' + paper['abstract']).lower()
        for keyword in emerging_keywords:
            if keyword in text:
                emerging_trends[keyword].append(paper)
    
    print(f"\n🚀 新兴研究方向 (活跃但未成熟):")
    for direction, papers_list in emerging_trends.items():
        if len(papers_list) >= 1:  # 至少1篇论文提及
            print(f"  {direction.title()}: {len(papers_list)} papers")
            if papers_list:
                print(f"    最新: {papers_list[0]['title'][:50]}...")
    
    return dict(emerging_trends)

if __name__ == "__main__":
    results = extract_critical_challenges_data()