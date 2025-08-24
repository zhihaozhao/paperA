#!/usr/bin/env python3
"""
V5表格优化脚本
解决表格超宽、重叠、换行问题
按照相对百分比重新分配列宽，确保自动换行
"""

import re

def analyze_and_optimize_tables():
    """分析和优化所有表格"""
    
    # 读取LaTeX文件
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("=" * 70)
    print("📊 V5表格优化分析")
    print("=" * 70)
    
    # 找到所有tabular定义
    tabular_pattern = r'\\begin\{tabular\}\{([^}]+)\}'
    matches = list(re.finditer(tabular_pattern, content))
    
    optimizations = []
    
    for i, match in enumerate(matches, 1):
        tabular_def = match.group(1)
        print(f"\n🔍 表格 {i}:")
        print(f"   当前定义: {tabular_def}")
        
        # 提取p{width}定义
        p_widths = re.findall(r'p\{([0-9.]+)\\textwidth\}', tabular_def)
        
        if p_widths:
            total_width = sum(float(w) for w in p_widths)
            print(f"   📏 列宽总和: {total_width:.3f}")
            
            if total_width > 0.98:
                # 需要优化的表格
                print(f"   ⚠️  接近边界，需要优化")
                optimizations.append((i, tabular_def, total_width, "边界问题"))
            elif total_width > 1.0:
                print(f"   ❌ 超宽 {(total_width-1.0)*100:.1f}%")
                optimizations.append((i, tabular_def, total_width, "超宽"))
            else:
                print(f"   ✅ 宽度合理")
        else:
            print("   ⚪ 无p{width}定义")
    
    # 提供具体的优化建议
    print("\n" + "=" * 70)
    print("🔧 表格优化建议")
    print("=" * 70)
    
    # 常见表格优化方案
    optimization_suggestions = {
        # Table 3 - Algorithm Comparison (88%)
        r'p\{0\.10\\textwidth\}p\{0\.12\\textwidth\}p\{0\.08\\textwidth\}p\{0\.08\\textwidth\}p\{0\.10\\textwidth\}p\{0\.08\\textwidth\}p\{0\.12\\textwidth\}p\{0\.20\\textwidth\}': {
            'name': 'Table 3 (Algorithm Comparison)',
            'current_total': 0.88,
            'optimized': 'p{0.09\\textwidth}p{0.11\\textwidth}p{0.08\\textwidth}p{0.08\\textwidth}p{0.10\\textwidth}p{0.08\\textwidth}p{0.11\\textwidth}p{0.22\\textwidth}',
            'new_total': 0.87,
            'reason': '增大References列宽，减少重叠'
        },
        
        # Figure 4 support table (88%)
        r'p\{0\.12\\textwidth\}p\{0\.10\\textwidth\}p\{0\.08\\textwidth\}p\{0\.10\\textwidth\}p\{0\.08\\textwidth\}p\{0\.10\\textwidth\}p\{0\.30\\textwidth\}': {
            'name': 'Table - Figure 4 Support',
            'current_total': 0.88,
            'optimized': 'p{0.11\\textwidth}p{0.09\\textwidth}p{0.08\\textwidth}p{0.09\\textwidth}p{0.08\\textwidth}p{0.09\\textwidth}p{0.33\\textwidth}',
            'new_total': 0.87,
            'reason': '平衡列宽，增大References列'
        },
        
        # Motion control enhanced (100% - 需要缩减)
        r'p\{0\.15\\textwidth\}p\{0\.08\\textwidth\}p\{0\.12\\textwidth\}p\{0\.15\\textwidth\}p\{0\.18\\textwidth\}p\{0\.12\\textwidth\}p\{0\.20\\textwidth\}': {
            'name': 'Motion Control Enhanced (100%)',
            'current_total': 1.00,
            'optimized': 'p{0.14\\textwidth}p{0.07\\textwidth}p{0.11\\textwidth}p{0.14\\textwidth}p{0.17\\textwidth}p{0.11\\textwidth}p{0.19\\textwidth}',
            'new_total': 0.93,
            'reason': '缩减7%避免边界问题，保持比例'
        },
        
        # Motion control KPIs (100% - 需要缩减)  
        r'p\{0\.20\\textwidth\}p\{0\.18\\textwidth\}p\{0\.22\\textwidth\}p\{0\.40\\textwidth\}': {
            'name': 'Motion Control KPIs (100%)',
            'current_total': 1.00,
            'optimized': 'p{0.18\\textwidth}p{0.17\\textwidth}p{0.20\\textwidth}p{0.37\\textwidth}',
            'new_total': 0.92,
            'reason': '缩减8%避免边界问题，保持内容可读性'
        }
    }
    
    print("\n🎯 具体优化方案:")
    for pattern, suggestion in optimization_suggestions.items():
        print(f"\n📋 {suggestion['name']}:")
        print(f"   当前: {suggestion['current_total']:.2f} → 优化: {suggestion['new_total']:.2f}")
        print(f"   原因: {suggestion['reason']}")
        print(f"   新定义: {suggestion['optimized']}")
    
    # 通用优化规则
    print(f"\n📏 通用优化规则:")
    print(f"   ✅ 总宽度控制在 85-95% 之间")
    print(f"   ✅ 使用 p{{width}} 确保自动换行") 
    print(f"   ✅ 长内容列给予更多空间")
    print(f"   ✅ 使用 \\arraystretch{{1.2}} 增加行间距")
    print(f"   ✅ 复杂表格使用 \\small 或 \\footnotesize")
    
    return optimization_suggestions

def generate_optimization_latex():
    """生成优化后的LaTeX代码片段"""
    
    print(f"\n" + "=" * 70)
    print("📝 优化代码示例")
    print("=" * 70)
    
    # 示例：优化后的表格设置
    optimized_example = """
% 优化的表格设置模板
\\begin{table*}[htbp]
\\centering
\\small  % 使用小字体
\\renewcommand{\\arraystretch}{1.2}  % 增加行间距
\\caption{您的表格标题}
\\label{tab:your_label}
\\begin{tabular}{p{0.12\\textwidth}p{0.15\\textwidth}p{0.20\\textwidth}p{0.25\\textwidth}p{0.23\\textwidth}}
\\toprule
\\textbf{列1} & \\textbf{列2} & \\textbf{列3} & \\textbf{列4} & \\textbf{列5} \\\\
\\midrule
内容自动换行 & 较长的内容会在p列类型中自动换行，避免重叠 & 中等长度内容 & 更长的内容可以分配更宽的列 & 参考文献列通常需要最宽 \\\\
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    print(optimized_example)
    
    return optimized_example

if __name__ == "__main__":
    print("🚀 启动V5表格优化分析...")
    
    # 分析表格
    suggestions = analyze_and_optimize_tables()
    
    # 生成优化代码
    generate_optimization_latex()
    
    print(f"\n" + "=" * 70)
    print("✅ 分析完成！")
    print("🔧 根据以上建议调整表格列宽，确保:")
    print("   1. 总宽度 < 95%")
    print("   2. 内容自动换行")
    print("   3. 避免跨格重叠")
    print("   4. 便于手动调节")
    print("=" * 70)