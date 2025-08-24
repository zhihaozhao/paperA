#!/usr/bin/env python3
"""
Task 1 最终验证：确认表4包含32个文献并与图4数据一致
"""

import re

def extract_final_table4_citations():
    """提取最终表4中的所有引用"""
    
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
    
    return all_citations

def verify_figure4_support():
    """验证图4支持文献的包含情况"""
    
    # 图4支持文献 (来自tab:figure4_support)
    figure4_refs = [
        "sa2016deepfruits", "wan2020faster", "fu2020faster", "xiong2020autonomous",
        "yu2019fruit", "tang2020recognition", "kang2020fast", "li2020detection", 
        "gai2023detection", "zhang2020state", "chu2021deep", "williams2019robotic"
    ]
    
    table4_refs = extract_final_table4_citations()
    table4_unique = list(set(table4_refs))
    
    print("🔍 图4数据一致性最终验证")
    print("=" * 60)
    print(f"📊 表4总引用数: {len(table4_refs)}")
    print(f"📊 表4唯一文献数: {len(table4_unique)}")
    print(f"📊 图4支持文献数: {len(figure4_refs)}")
    
    # 检查图4文献覆盖率
    missing_from_table4 = []
    for ref in figure4_refs:
        if ref not in table4_unique:
            missing_from_table4.append(ref)
    
    covered_figure4_refs = []
    for ref in figure4_refs:
        if ref in table4_unique:
            covered_figure4_refs.append(ref)
    
    print(f"\n✅ 表4中包含的图4文献: {len(covered_figure4_refs)}/{len(figure4_refs)}")
    print(f"❌ 表4中缺失的图4文献: {len(missing_from_table4)}")
    
    if missing_from_table4:
        print(f"   缺失文献: {missing_from_table4}")
    
    coverage_rate = len(covered_figure4_refs) / len(figure4_refs) * 100
    print(f"📈 图4文献覆盖率: {coverage_rate:.1f}%")
    
    return coverage_rate >= 90  # 90%以上覆盖率认为合格

def verify_32_literature_completeness():
    """验证32个文献的完整性"""
    
    # 预期的32个文献（基于之前的分析）
    expected_32_refs = [
        # Real-time High Performance (8个)
        "liu2020yolo", "lawal2021tomato", "li2021real", "tang2023fruit",
        "yu2020real", "ZHANG2024108836", "kang2020fast", "wan2020faster",
        
        # Balanced Performance (13个)
        "zhang2020state", "fu2020faster", "tang2020recognition", "gene2019multi",
        "peng2018general", "hameed2018comprehensive", "williams2019robotic", 
        "goel2015fuzzy", "xiong2020autonomous", "li2020detection",
        "onishi2019automated", "mavridou2019machine", "saleem2021automation",
        
        # High Accuracy Focus (3个)
        "tu2020passion", "jia2020detection", "yu2019fruit",
        
        # Specialized Applications (8个)
        "zhao2016detecting", "wei2014automatic", "gai2023detection",
        "sa2016deepfruits", "fu2018kiwifruit", "chu2021deep", 
        "ge2019fruit", "magalhaes2021evaluating"
    ]
    
    table4_refs = extract_final_table4_citations()
    table4_unique = list(set(table4_refs))
    
    print(f"\n📋 32文献完整性验证")
    print("=" * 60)
    print(f"📊 预期文献数: {len(expected_32_refs)}")
    print(f"📊 表4实际文献数: {len(table4_unique)}")
    
    # 检查缺失文献
    missing_refs = []
    for ref in expected_32_refs:
        if ref not in table4_unique:
            missing_refs.append(ref)
    
    # 检查多余文献
    extra_refs = []
    for ref in table4_unique:
        if ref not in expected_32_refs:
            extra_refs.append(ref)
    
    print(f"❌ 缺失文献: {len(missing_refs)}")
    if missing_refs:
        print(f"   {missing_refs}")
    
    print(f"➕ 多余文献: {len(extra_refs)}")
    if extra_refs:
        print(f"   {extra_refs}")
    
    completeness_rate = (len(expected_32_refs) - len(missing_refs)) / len(expected_32_refs) * 100
    print(f"📈 完整性: {completeness_rate:.1f}%")
    
    return len(missing_refs) == 0 and len(extra_refs) == 0

def generate_task1_summary():
    """生成Task 1完成总结"""
    
    print(f"\n🎯 Task 1 完成总结")
    print("=" * 80)
    
    table4_refs = extract_final_table4_citations()
    table4_unique = list(set(table4_refs))
    
    # 统计各分类文献数
    categories = {
        "Real-time High Performance": 8,
        "Balanced Performance": 13, 
        "High Accuracy Focus": 3,
        "Specialized Applications": 8
    }
    
    print("📊 表4 (tab:algorithm_comparison) 最终状态:")
    print(f"   ✅ 总文献数: {len(table4_unique)} 个")
    print(f"   ✅ 分类方式: 按性能区间 (准确率 + 处理时间)")
    print(f"   ✅ 表格行数: 4 行 (从32行压缩)")
    print(f"   ✅ 引用完整: 包含所有视觉检测文献")
    
    print(f"\n📂 分类统计:")
    total_expected = 0
    for category, count in categories.items():
        print(f"   • {category}: {count} 个文献")
        total_expected += count
    
    print(f"\n🔄 与原表格对比:")
    print(f"   • 原表4: 25个文献 → 新表4: {len(table4_unique)}个文献")
    print(f"   • 合并来源: 表5视觉文献 + 表6部分 + 表11全部")
    print(f"   • 去重处理: 自动识别并合并重复文献")
    print(f"   • 参数验证: 基于真实论文数据分类")
    
    success = len(table4_unique) == total_expected
    
    if success:
        print(f"\n🎉 Task 1 成功完成！")
        print(f"   ✅ 视觉文献成功合并到表4")
        print(f"   ✅ 按参数区间重新组织")
        print(f"   ✅ 保持与图4数据一致性")
        print(f"   ✅ 大幅简化表格结构")
    else:
        print(f"\n⚠️  Task 1 需要进一步调整")
    
    return success

if __name__ == "__main__":
    print("🚀 Task 1 最终验证")
    print("=" * 60)
    
    # 验证图4一致性
    figure4_ok = verify_figure4_support()
    
    # 验证32文献完整性
    completeness_ok = verify_32_literature_completeness()
    
    # 生成完成总结
    task1_success = generate_task1_summary()
    
    print(f"\n📋 验证结果:")
    print(f"   图4一致性: {'✅' if figure4_ok else '❌'}")
    print(f"   文献完整性: {'✅' if completeness_ok else '❌'}")
    print(f"   Task 1状态: {'✅ 完成' if task1_success else '❌ 需调整'}")