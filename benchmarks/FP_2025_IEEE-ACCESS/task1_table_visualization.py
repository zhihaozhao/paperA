#!/usr/bin/env python3
"""
Task 1 表格可视化：展示合并后的表4结构
"""

import re
import pandas as pd

def extract_table4_data():
    """从LaTeX文件中提取表4的数据"""
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到表4的内容
    table4_pattern = r'\\label\{tab:algorithm_comparison\}.*?\\end\{tabularx\}'
    table4_match = re.search(table4_pattern, content, re.DOTALL)
    
    if not table4_match:
        print("❌ 未找到表4")
        return []
    
    table4_content = table4_match.group()
    
    # 提取数据行
    data_rows = []
    current_family = ""
    
    # 分析每一行
    lines = table4_content.split('\n')
    for line in lines:
        line = line.strip()
        
        # 检测算法家族
        if '\\multirow' in line:
            family_match = re.search(r'\\textbf\{([^}]+)\}', line)
            if family_match:
                current_family = family_match.group(1).replace('\\\\', ' ').replace('\\', '').strip()
        
        # 提取数据行
        if '&' in line and '\\\\' in line and not '\\toprule' in line and not '\\midrule' in line:
            parts = line.split('&')
            if len(parts) >= 7:
                # 清理数据
                model = parts[1].strip() if len(parts) > 1 else ""
                accuracy = parts[2].strip() if len(parts) > 2 else ""
                time = parts[3].strip() if len(parts) > 3 else ""
                sample = parts[4].strip() if len(parts) > 4 else ""
                env = parts[5].strip() if len(parts) > 5 else ""
                features = parts[6].strip() if len(parts) > 6 else ""
                
                # 提取引用
                cite_match = re.search(r'\\cite\{([^}]+)\}', line)
                citation = cite_match.group(1) if cite_match else ""
                
                if model and accuracy:  # 确保有有效数据
                    data_rows.append({
                        'Algorithm Family': current_family,
                        'Version/Model': model,
                        'Accuracy (%)': accuracy,
                        'Processing Time (ms)': time,
                        'Sample Size': sample,
                        'Environment': env,
                        'Evolution Features': features,
                        'Citation': citation
                    })
    
    return data_rows

def visualize_table_structure():
    """可视化表格结构"""
    
    data = extract_table4_data()
    
    if not data:
        print("❌ 无法提取表格数据")
        return
    
    print("📊 Task 1 合并后的表4 (tab:algorithm_comparison) 结构")
    print("=" * 100)
    
    # 按算法家族分组统计
    family_stats = {}
    for row in data:
        family = row['Algorithm Family']
        if family not in family_stats:
            family_stats[family] = []
        family_stats[family].append(row)
    
    print(f"📈 总计: {len(data)} 个算法条目，分布在 {len(family_stats)} 个算法家族中\n")
    
    # 展示每个算法家族的详细信息
    for i, (family, entries) in enumerate(family_stats.items(), 1):
        print(f"🏷️  {i}. {family} ({len(entries)} 个算法)")
        print("-" * 80)
        
        for j, entry in enumerate(entries, 1):
            print(f"   {j}. {entry['Version/Model']}")
            print(f"      📊 准确率: {entry['Accuracy (%)']} | ⏱️  处理时间: {entry['Processing Time (ms)']} | 📝 样本: {entry['Sample Size']}")
            print(f"      🌍 环境: {entry['Environment']} | 🔬 特征: {entry['Evolution Features']}")
            print(f"      📚 引用: {entry['Citation']}")
            print()
        
        print()
    
    # 统计分析
    print("📊 统计分析")
    print("=" * 50)
    
    # 算法家族分布
    family_counts = [(family, len(entries)) for family, entries in family_stats.items()]
    family_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("🏆 算法家族分布 (按数量排序):")
    for family, count in family_counts:
        print(f"   {family}: {count} 个算法")
    
    # 性能统计
    accuracies = []
    times = []
    for row in data:
        try:
            acc = float(row['Accuracy (%)'])
            accuracies.append(acc)
        except:
            pass
        
        try:
            time_val = row['Processing Time (ms)'].replace('ms', '').strip()
            if time_val and time_val != 'N/A':
                times.append(float(time_val))
        except:
            pass
    
    if accuracies:
        print(f"\n📈 准确率统计:")
        print(f"   最高: {max(accuracies):.1f}%")
        print(f"   最低: {min(accuracies):.1f}%")
        print(f"   平均: {sum(accuracies)/len(accuracies):.1f}%")
    
    if times:
        print(f"\n⏱️  处理时间统计:")
        print(f"   最快: {min(times):.1f}ms")
        print(f"   最慢: {max(times):.1f}ms")
        print(f"   平均: {sum(times)/len(times):.1f}ms")

def create_summary_table():
    """创建汇总表格"""
    
    data = extract_table4_data()
    
    if not data:
        return
    
    print("\n" + "="*100)
    print("📋 Task 1 合并汇总表")
    print("="*100)
    
    # 创建DataFrame用于更好的显示
    df = pd.DataFrame(data)
    
    # 显示前10行作为示例
    print("📝 表格前10行预览:")
    print("-"*100)
    display_cols = ['Algorithm Family', 'Version/Model', 'Accuracy (%)', 'Processing Time (ms)', 'Citation']
    if len(df) > 0:
        preview_df = df[display_cols].head(10)
        for i, row in preview_df.iterrows():
            print(f"{i+1:2d}. {row['Algorithm Family']:<20} | {row['Version/Model']:<25} | {row['Accuracy (%)']:>8} | {row['Processing Time (ms)']:>12} | {row['Citation']:<20}")
    
    print(f"\n... (共 {len(df)} 行)")

if __name__ == "__main__":
    print("🎨 Task 1 表格可视化工具")
    print("="*60)
    
    # 可视化表格结构
    visualize_table_structure()
    
    # 创建汇总表格
    create_summary_table()
    
    print("\n✅ 可视化完成！")
    print("💬 请查看上述结构，我们可以讨论任何需要调整的地方。")