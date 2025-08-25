#!/usr/bin/env python3
"""
提取prisma_data.csv中的真实数据
严格使用真实数据，零编造
"""

import csv
import json
import re
from collections import defaultdict, Counter

def extract_real_prisma_data():
    """从prisma_data.csv提取所有真实数据"""
    
    print("📊 从prisma_data.csv提取真实数据...")
    
    # 读取prisma数据
    papers = []
    bibtex_keys = []
    
    try:
        with open('../docs/prisma_data.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 跳过标题行
            
            print(f"📋 CSV列标题: {headers}")
            
            for row_num, row in enumerate(reader, 2):
                if len(row) >= 8:  # 确保有足够的列
                    paper_data = {
                        'row': row_num,
                        'relevant': row[0] if len(row) > 0 else '',
                        'database': row[1] if len(row) > 1 else '',
                        'title': row[2] if len(row) > 2 else '',
                        'citations': row[3] if len(row) > 3 else '',
                        'year': row[4] if len(row) > 4 else '',
                        'selected': row[5] if len(row) > 5 else '',
                        'journal': row[6] if len(row) > 6 else '',
                        'bibtex_cite': row[7] if len(row) > 7 else '',  # H列
                        'bibtex_entry': row[8] if len(row) > 8 else ''   # I列
                    }
                    
                    # 只收集相关的论文 (relevant = 'y')
                    if paper_data['relevant'].lower() == 'y':
                        papers.append(paper_data)
                        
                        # 提取bibtex key
                        if '\\cite{' in paper_data['bibtex_cite']:
                            match = re.search(r'\\cite\{([^}]+)\}', paper_data['bibtex_cite'])
                            if match:
                                bibtex_keys.append(match.group(1))
    
    except Exception as e:
        print(f"❌ 读取prisma_data.csv失败: {e}")
        return None, None
    
    print(f"✅ 提取到 {len(papers)} 篇相关论文")
    print(f"✅ 提取到 {len(bibtex_keys)} 个真实bibtex引用")
    
    return papers, bibtex_keys

def analyze_vision_algorithms(papers):
    """分析视觉算法论文"""
    
    print("\n🔍 分析视觉算法论文...")
    
    vision_keywords = [
        'vision', 'yolo', 'rcnn', 'r-cnn', 'cnn', 'deep', 'detection', 
        'recognition', 'segmentation', 'mask', 'faster', 'object detection',
        'computer vision', 'image', 'visual', 'neural network'
    ]
    
    vision_papers = []
    algorithm_families = defaultdict(list)
    
    for paper in papers:
        title_lower = paper['title'].lower()
        
        # 检查是否是视觉相关论文
        is_vision = any(keyword in title_lower for keyword in vision_keywords)
        
        if is_vision:
            vision_papers.append(paper)
            
            # 分类算法家族
            if any(word in title_lower for word in ['yolo']):
                algorithm_families['YOLO'].append(paper)
            elif any(word in title_lower for word in ['rcnn', 'r-cnn', 'mask']):
                algorithm_families['R-CNN'].append(paper)
            elif any(word in title_lower for word in ['deep', 'cnn', 'neural']):
                algorithm_families['Deep Learning'].append(paper)
            elif any(word in title_lower for word in ['vision', 'visual', 'image']):
                algorithm_families['Traditional Vision'].append(paper)
            else:
                algorithm_families['Other'].append(paper)
    
    print(f"✅ 找到 {len(vision_papers)} 篇视觉算法论文")
    for family, papers_list in algorithm_families.items():
        print(f"   - {family}: {len(papers_list)} 篇")
    
    return vision_papers, algorithm_families

def analyze_robotics_papers(papers):
    """分析机器人论文"""
    
    print("\n🤖 分析机器人论文...")
    
    robotics_keywords = [
        'robot', 'robotic', 'harvesting', 'picking', 'gripper', 'manipulator',
        'autonomous', 'automation', 'control', 'motion', 'path planning',
        'end effector', 'mechanical'
    ]
    
    robotics_papers = []
    control_methods = defaultdict(list)
    
    for paper in papers:
        title_lower = paper['title'].lower()
        
        # 检查是否是机器人相关论文
        is_robotics = any(keyword in title_lower for keyword in robotics_keywords)
        
        if is_robotics:
            robotics_papers.append(paper)
            
            # 分类控制方法
            if any(word in title_lower for word in ['path', 'planning', 'motion']):
                control_methods['Path Planning'].append(paper)
            elif any(word in title_lower for word in ['control', 'pid', 'feedback']):
                control_methods['Control Systems'].append(paper)
            elif any(word in title_lower for word in ['gripper', 'end effector', 'manipulator']):
                control_methods['End Effectors'].append(paper)
            elif any(word in title_lower for word in ['autonomous', 'automation']):
                control_methods['Autonomous Systems'].append(paper)
            else:
                control_methods['General Robotics'].append(paper)
    
    print(f"✅ 找到 {len(robotics_papers)} 篇机器人论文")
    for method, papers_list in control_methods.items():
        print(f"   - {method}: {len(papers_list)} 篇")
    
    return robotics_papers, control_methods

def analyze_critical_trends(papers):
    """分析关键趋势"""
    
    print("\n📈 分析关键趋势...")
    
    # 按年份分析
    year_distribution = Counter()
    citation_trends = []
    
    for paper in papers:
        year = paper['year']
        if year and year.isdigit():
            year_int = int(year)
            year_distribution[year_int] += 1
            
            # 引用数分析
            citations = paper['citations']
            if citations and citations.isdigit():
                citation_trends.append({
                    'year': year_int,
                    'citations': int(citations),
                    'title': paper['title']
                })
    
    # 排序获取高引用论文
    citation_trends.sort(key=lambda x: x['citations'], reverse=True)
    top_cited = citation_trends[:20]  # 前20篇高引用论文
    
    print(f"✅ 年份分布: {dict(year_distribution)}")
    print(f"✅ 前20篇高引用论文已识别")
    
    return year_distribution, top_cited

def generate_real_figure_data():
    """生成真实的图表数据"""
    
    print("\n📊 生成真实图表数据...")
    
    # 提取真实数据
    papers, bibtex_keys = extract_real_prisma_data()
    
    if not papers:
        print("❌ 无法提取数据")
        return False
    
    # 分析各类论文
    vision_papers, algorithm_families = analyze_vision_algorithms(papers)
    robotics_papers, control_methods = analyze_robotics_papers(papers)
    year_distribution, top_cited = analyze_critical_trends(papers)
    
    # 保存Figure 4数据 (视觉算法)
    figure4_data = {
        'total_vision_papers': len(vision_papers),
        'algorithm_families': {family: len(papers_list) for family, papers_list in algorithm_families.items()},
        'detailed_papers': []
    }
    
    for family, papers_list in algorithm_families.items():
        for paper in papers_list[:5]:  # 每个家族取前5篇
            bibtex_key = ''
            if '\\cite{' in paper['bibtex_cite']:
                match = re.search(r'\\cite\{([^}]+)\}', paper['bibtex_cite'])
                if match:
                    bibtex_key = match.group(1)
            
            figure4_data['detailed_papers'].append({
                'family': family,
                'title': paper['title'],
                'year': paper['year'],
                'citations': paper['citations'],
                'bibtex_key': bibtex_key
            })
    
    # 保存Figure 9数据 (机器人控制)
    figure9_data = {
        'total_robotics_papers': len(robotics_papers),
        'control_methods': {method: len(papers_list) for method, papers_list in control_methods.items()},
        'detailed_papers': []
    }
    
    for method, papers_list in control_methods.items():
        for paper in papers_list[:5]:  # 每个方法取前5篇
            bibtex_key = ''
            if '\\cite{' in paper['bibtex_cite']:
                match = re.search(r'\\cite\{([^}]+)\}', paper['bibtex_cite'])
                if match:
                    bibtex_key = match.group(1)
            
            figure9_data['detailed_papers'].append({
                'method': method,
                'title': paper['title'],
                'year': paper['year'],
                'citations': paper['citations'],
                'bibtex_key': bibtex_key
            })
    
    # 保存Figure 10数据 (关键趋势)
    figure10_data = {
        'year_distribution': dict(year_distribution),
        'top_cited_papers': []
    }
    
    for paper_info in top_cited[:20]:
        # 找到对应的完整论文信息
        for paper in papers:
            if paper['title'] == paper_info['title']:
                bibtex_key = ''
                if '\\cite{' in paper['bibtex_cite']:
                    match = re.search(r'\\cite\{([^}]+)\}', paper['bibtex_cite'])
                    if match:
                        bibtex_key = match.group(1)
                
                figure10_data['top_cited_papers'].append({
                    'title': paper['title'],
                    'year': paper['year'],
                    'citations': paper['citations'],
                    'bibtex_key': bibtex_key
                })
                break
    
    # 保存数据到JSON文件
    with open('REAL_FIGURE4_DATA.json', 'w', encoding='utf-8') as f:
        json.dump(figure4_data, f, indent=2, ensure_ascii=False)
    
    with open('REAL_FIGURE9_DATA.json', 'w', encoding='utf-8') as f:
        json.dump(figure9_data, f, indent=2, ensure_ascii=False)
    
    with open('REAL_FIGURE10_DATA.json', 'w', encoding='utf-8') as f:
        json.dump(figure10_data, f, indent=2, ensure_ascii=False)
    
    # 保存所有真实bibtex引用
    with open('REAL_BIBTEX_KEYS.json', 'w', encoding='utf-8') as f:
        json.dump(bibtex_keys, f, indent=2, ensure_ascii=False)
    
    print("✅ 真实数据已保存到JSON文件")
    print(f"📄 REAL_FIGURE4_DATA.json - {len(vision_papers)}篇视觉论文")
    print(f"📄 REAL_FIGURE9_DATA.json - {len(robotics_papers)}篇机器人论文") 
    print(f"📄 REAL_FIGURE10_DATA.json - {len(top_cited)}篇高引用论文")
    print(f"📄 REAL_BIBTEX_KEYS.json - {len(bibtex_keys)}个真实引用")
    
    return True

if __name__ == "__main__":
    print("🚨 提取prisma_data.csv中的真实数据")
    print("🔒 严格使用真实数据，零编造")
    
    success = generate_real_figure_data()
    
    if success:
        print("\n✅ 真实数据提取完成！")
        print("📊 所有数据来自prisma_data.csv")
        print("🔒 零编造，100%真实可验证")
    else:
        print("\n❌ 数据提取失败！")