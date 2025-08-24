#!/usr/bin/env python3
"""
TabularX相对百分比列宽修复脚本
结合tabularx和p{百分比宽度}，彻底解决越界和跨行问题
"""
import re

def analyze_current_tabularx():
    """分析当前tabularx表格的列定义"""
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔍 当前TabularX表格分析")
    print("=" * 70)
    
    # 找到所有tabularx定义
    tabularx_pattern = r'\\begin\{tabularx\}\{\\textwidth\}\{([^}]+)\}'
    matches = list(re.finditer(tabularx_pattern, content))
    
    for i, match in enumerate(matches, 1):
        col_def = match.group(1)
        print(f"\n📋 TabularX表格 {i}:")
        print(f"   列定义: {col_def}")
        
        # 分析列类型
        l_count = col_def.count('l')
        c_count = col_def.count('c') 
        r_count = col_def.count('r')
        x_count = col_def.count('X')
        p_count = len(re.findall(r'p\{[^}]+\}', col_def))
        
        print(f"   📊 列类型统计:")
        print(f"      l(左对齐): {l_count}")
        print(f"      c(居中): {c_count}")
        print(f"      r(右对齐): {r_count}")
        print(f"      X(自适应): {x_count}")
        print(f"      p{{width}}: {p_count}")
        
        total_cols = l_count + c_count + r_count + x_count + p_count
        print(f"   📏 总列数: {total_cols}")
        
        # 评估潜在问题
        if x_count == 0:
            print("   ⚠️  警告: 无X列，可能无法自动调整")
        if l_count + c_count + r_count > 3:
            print("   ⚠️  警告: 固定列过多，可能导致挤压")

def generate_optimized_definitions():
    """生成优化的tabularx列定义"""
    
    print(f"\n" + "=" * 70)
    print("🔧 优化的TabularX列宽定义")
    print("=" * 70)
    
    # 推荐的列定义模板
    templates = {
        'Algorithm Comparison (8列)': {
            'current': 'l X c c c c X X',
            'optimized': '>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.15\\linewidth}cc>{\\raggedright\\arraybackslash}p{0.10\\linewidth}c>{\\raggedright\\arraybackslash}p{0.18\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}',
            'reason': '精确控制每列宽度，确保References列足够宽'
        },
        
        'Motion Control Enhanced (7列)': {
            'current': 'l c c c X X X', 
            'optimized': '>{\\raggedright\\arraybackslash}p{0.15\\linewidth}ccc>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}',
            'reason': '数值列居中，文字列使用p{百分比}自动换行'
        },
        
        'Motion Control KPIs (4列)': {
            'current': 'l c X X',
            'optimized': '>{\\raggedright\\arraybackslash}p{0.20\\linewidth}c>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.45\\linewidth}',
            'reason': '最后一列Technical Significance需要最大空间'
        },
        
        'Dataset Table 6 (8列)': {
            'current': 'c c X c c X X X',
            'optimized': 'cc>{\\raggedright\\arraybackslash}p{0.15\\linewidth}cc>{\\raggedright\\arraybackslash}p{0.20\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}',
            'reason': '前面数值列紧凑，后面描述列充分展开'
        },
        
        'Performance Metrics (5列)': {
            'current': 'l X X X c',
            'optimized': '>{\\raggedright\\arraybackslash}p{0.15\\linewidth}>{\\raggedright\\arraybackslash}p{0.20\\linewidth}>{\\raggedright\\arraybackslash}p{0.30\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}c',
            'reason': 'Strengths列需要最大空间显示详细描述'
        }
    }
    
    for name, template in templates.items():
        print(f"\n📋 {name}:")
        print(f"   当前: {template['current']}")
        print(f"   优化: {template['optimized']}")  
        print(f"   原因: {template['reason']}")

def create_fix_commands():
    """生成具体的修复命令"""
    
    print(f"\n" + "=" * 70)
    print("📝 具体修复方案")
    print("=" * 70)
    
    fixes = [
        {
            'table': 'Algorithm Comparison',
            'search': r'\\begin\{tabularx\}\{\\textwidth\}\{l X c c c c X X\}',
            'replace': r'\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.15\\linewidth}cc>{\\raggedright\\arraybackslash}p{0.10\\linewidth}c>{\\raggedright\\arraybackslash}p{0.18\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}}'
        },
        {
            'table': 'Motion Control Enhanced', 
            'search': r'\\begin\{tabularx\}\{\\textwidth\}\{l c c c X X X\}',
            'replace': r'\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{0.15\\linewidth}ccc>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}}'
        },
        {
            'table': 'Motion Control KPIs',
            'search': r'\\begin\{tabularx\}\{\\textwidth\}\{l c X X\}', 
            'replace': r'\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{0.20\\linewidth}c>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.45\\linewidth}}'
        }
    ]
    
    print("🔧 推荐使用以下列定义特性:")
    print("   ✅ p{百分比\\linewidth} - 精确控制列宽")
    print("   ✅ >{\\raggedright\\arraybackslash} - 左对齐+自动换行")
    print("   ✅ 总宽度控制在95%以内避免越界")
    print("   ✅ 文字列宽松，数值列紧凑")
    
    return fixes

if __name__ == "__main__":
    print("🚀 TabularX相对百分比列宽分析...")
    
    # 分析当前状态
    analyze_current_tabularx()
    
    # 生成优化定义
    generate_optimized_definitions()
    
    # 创建修复方案
    fixes = create_fix_commands()
    
    print(f"\n" + "=" * 70)
    print("✅ 分析完成！建议:")
    print("1. 使用p{百分比\\linewidth}替代简单的X列")
    print("2. 添加>{\\raggedright\\arraybackslash}确保左对齐和换行")  
    print("3. 数值列使用c，文字列使用p{宽度}")
    print("4. 控制总列宽在95%以内")
    print("=" * 70)