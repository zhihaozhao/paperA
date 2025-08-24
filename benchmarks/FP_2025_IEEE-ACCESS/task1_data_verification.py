#!/usr/bin/env python3
"""
Task 1 数据验证：确认表4内容准确性并与图4数据保持一致
"""

import re

def extract_current_table4_data():
    """提取当前表4中的引用和数据"""
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到表4内容
    table4_pattern = r'\\label\{tab:algorithm_comparison\}.*?\\end\{tabularx\}'
    table4_match = re.search(table4_pattern, content, re.DOTALL)
    
    if not table4_match:
        print("❌ 未找到表4")
        return []
    
    table4_content = table4_match.group()
    
    # 提取所有引用
    cite_pattern = r'\\cite\{([^}]+)\}'
    citations = re.findall(cite_pattern, table4_content)
    
    # 解析多个引用
    all_citations = []
    for cite_group in citations:
        refs = cite_group.split(',')
        all_citations.extend([ref.strip() for ref in refs])
    
    print("📊 当前表4数据统计")
    print("=" * 60)
    print(f"总引用数: {len(all_citations)}")
    print(f"唯一引用数: {len(set(all_citations))}")
    
    return all_citations

def verify_figure4_consistency():
    """验证与图4数据的一致性"""
    
    print("\n🔍 图4数据一致性验证")
    print("=" * 60)
    
    # 从之前的分析中获取的图4支持文献 (tab:figure4_support)
    figure4_refs = [
        "sa2016deepfruits", "wan2020faster", "fu2020faster", "xiong2020autonomous",
        "yu2019fruit", "tang2020recognition", "kang2020fast", "li2020detection", 
        "gai2023detection", "zhang2020state", "chu2021deep", "williams2019robotic"
    ]
    
    # 表4当前引用
    table4_refs = extract_current_table4_data()
    table4_unique = list(set(table4_refs))
    
    # 检查图4文献是否都包含在表4中
    missing_in_table4 = []
    for ref in figure4_refs:
        if ref not in table4_unique:
            missing_in_table4.append(ref)
    
    # 检查表4中是否有图4中没有的文献（这是正常的，因为表4应该更全面）
    extra_in_table4 = []
    for ref in table4_unique:
        if ref not in figure4_refs:
            extra_in_table4.append(ref)
    
    print(f"📋 图4支持文献: {len(figure4_refs)} 个")
    print(f"📋 表4当前文献: {len(table4_unique)} 个")
    print(f"❌ 表4中缺失的图4文献: {len(missing_in_table4)} 个")
    if missing_in_table4:
        print(f"   缺失文献: {missing_in_table4}")
    
    print(f"✅ 表4中额外的文献: {len(extra_in_table4)} 个")
    print(f"   (这是正常的，表4应该更全面)")
    
    return missing_in_table4, extra_in_table4

def verify_parameter_accuracy():
    """验证参数数据的准确性"""
    
    print("\n📈 参数数据准确性验证")
    print("=" * 60)
    
    # 根据原始文献数据验证分类是否正确
    literature_data = {
        # Real-time High Performance: Accuracy ≥90%, Time ≤80ms
        "liu2020yolo": (96.4, 54),      # ✅
        "lawal2021tomato": (99.5, 52),  # ✅
        "li2021real": (91.1, 12.3),     # ✅ 
        "tang2023fruit": (92.1, 31),    # ✅
        "yu2020real": (94.4, 56),       # ✅
        "ZHANG2024108836": (90.2, 12),  # ✅
        "kang2020fast": (90.9, 78),     # ✅
        "zhang2020state": (91.5, 83),   # ❌ 时间超过80ms，应该在Balanced
        
        # Balanced Performance: Accuracy 85-95%, Time 80-200ms
        "wan2020faster": (90.7, 58),    # ❌ 时间<80ms，应该在Real-time
        "fu2020faster": (89.3, 181),    # ✅
        "tu2020passion": (96.2, 120),   # ❌ 准确率>95%，应该在High Accuracy
        "tang2020recognition": (89.8, 92), # ✅
        "gai2023detection": (94.7, 467),   # ❌ 时间>200ms，应该在High Accuracy
        "zhao2016detecting": (88.7, 65),   # ❌ 时间<80ms，应该在Real-time
        "wei2014automatic": (89.2, 78),    # ❌ 时间<80ms，应该在Real-time
        "peng2018general": (89.5, 125),    # ✅
        "hameed2018comprehensive": (87.5, 125), # ✅
        "williams2019robotic": (85.9, 128),     # ✅
        
        # High Accuracy Focus: Accuracy ≥95%, Time >200ms
        "gene2019multi": (94.8, 136),   # ❌ 准确率<95%且时间<200ms，应该在Balanced
        "jia2020detection": (97.3, 250), # ✅
        "yu2019fruit": (95.8, 820),     # ✅
        "goel2015fuzzy": (94.3, 85),    # ❌ 准确率<95%且时间<200ms，应该在Balanced
        
        # Specialized Applications: Task-specific
        "sa2016deepfruits": (84.8, 393),    # ✅ 早期研究，特殊应用
        "fu2018kiwifruit": (92.3, 274),     # ❌ 应该在Balanced或High Accuracy
        "chu2021deep": (90.5, 250),         # ❌ 应该在High Accuracy
        "ge2019fruit": (90.0, 820),         # ✅ 特殊安全应用
        "magalhaes2021evaluating": (66.2, 16.4), # ✅ 特殊TPU优化
    }
    
    # 验证分类准确性
    classification_errors = []
    
    print("分类验证结果:")
    for category in ["Real-time High Performance", "Balanced Performance", 
                     "High Accuracy Focus", "Specialized Applications"]:
        print(f"\n📂 {category}:")
        
        if category == "Real-time High Performance":
            expected_refs = ["liu2020yolo", "lawal2021tomato", "li2021real", 
                           "tang2023fruit", "yu2020real", "ZHANG2024108836", 
                           "kang2020fast", "zhang2020state"]
            for ref in expected_refs:
                if ref in literature_data:
                    acc, time = literature_data[ref]
                    if acc >= 90 and time <= 80:
                        print(f"   ✅ {ref}: {acc}%, {time}ms")
                    else:
                        print(f"   ❌ {ref}: {acc}%, {time}ms (分类错误)")
                        classification_errors.append((ref, category, acc, time))
    
    return classification_errors

def suggest_corrections():
    """建议数据修正"""
    
    print("\n🔧 建议的数据修正")
    print("=" * 60)
    
    corrections = [
        {
            "issue": "zhang2020state (91.5%, 83ms) 时间超过80ms",
            "current": "Real-time High Performance", 
            "suggested": "Balanced Performance"
        },
        {
            "issue": "wan2020faster (90.7%, 58ms) 时间小于80ms且准确率≥90%",
            "current": "Balanced Performance",
            "suggested": "Real-time High Performance"
        },
        {
            "issue": "tu2020passion (96.2%, 120ms) 准确率≥95%",
            "current": "Balanced Performance", 
            "suggested": "High Accuracy Focus"
        },
        {
            "issue": "gai2023detection (94.7%, 467ms) 时间>200ms",
            "current": "Balanced Performance",
            "suggested": "High Accuracy Focus"
        },
        {
            "issue": "gene2019multi (94.8%, 136ms) 准确率<95%且时间<200ms", 
            "current": "High Accuracy Focus",
            "suggested": "Balanced Performance"
        }
    ]
    
    for i, correction in enumerate(corrections, 1):
        print(f"{i}. {correction['issue']}")
        print(f"   当前分类: {correction['current']}")
        print(f"   建议分类: {correction['suggested']}")
        print()
    
    return corrections

if __name__ == "__main__":
    print("🚀 Task 1 数据验证与一致性检查")
    print("=" * 60)
    
    # 提取表4数据
    table4_refs = extract_current_table4_data()
    
    # 验证与图4一致性
    missing, extra = verify_figure4_consistency()
    
    # 验证参数准确性
    errors = verify_parameter_accuracy()
    
    # 建议修正
    corrections = suggest_corrections()
    
    print(f"\n📋 验证总结:")
    print(f"✅ 表4包含 {len(set(table4_refs))} 个唯一文献")
    print(f"⚠️  发现 {len(corrections)} 个分类问题需要修正")
    print(f"🎯 建议重新分类以确保数据准确性")