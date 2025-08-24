#!/usr/bin/env python3
"""
LaTeX表格宽度检查脚本（修复版）
检查所有表格的列宽总和是否超出页面宽度
"""

import re

def check_table_widths(tex_file):
    """检查LaTeX文件中所有表格的列宽"""
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有tabular环境的列定义（更完整的正则表达式）
    tabular_pattern = r'\\begin\{tabular\}\{([^}]*(?:\}[^}]*)*)\}'
    matches = re.findall(tabular_pattern, content)
    
    print("=" * 70)
    print("📊 LaTeX表格宽度分析报告")
    print("=" * 70)
    
    problem_tables = []
    
    for i, match in enumerate(matches, 1):
        print(f"\n🔍 表格 {i}:")
        print(f"   定义: {match}")
        
        # 提取所有p{width}定义
        p_widths = re.findall(r'p\{([^}]+)\}', match)
        
        if not p_widths:
            print("   ✅ 无p{width}列定义 (使用默认列宽)")
            continue
        
        total_width = 0
        width_details = []
        has_issues = False
        
        for width in p_widths:
            if '\\textwidth' in width:
                # 提取textwidth系数
                coef_match = re.search(r'([0-9.]+)\\textwidth', width)
                if coef_match:
                    coef = float(coef_match.group(1))
                    total_width += coef
                    width_details.append(f"{coef:.3f}tw")
            elif '\\linewidth' in width:
                coef_match = re.search(r'([0-9.]+)\\linewidth', width)
                if coef_match:
                    coef = float(coef_match.group(1))
                    total_width += coef
                    width_details.append(f"{coef:.3f}lw")
            elif 'cm' in width:
                has_issues = True
                cm_match = re.search(r'([0-9.]+)cm', width)
                if cm_match:
                    cm_val = float(cm_match.group(1))
                    # 假设页面宽度约17cm (IEEE ACCESS标准)
                    coef = cm_val / 17.0
                    total_width += coef
                    width_details.append(f"{cm_val}cm")
            else:
                width_details.append(f"'{width}'")
        
        # 显示分析结果
        print(f"   📏 列宽: {' + '.join(width_details)}")
        print(f"   📐 总宽度: {total_width:.3f} (1.0 = 100%页面)")
        
        # 问题判断
        status = "✅"
        if total_width > 1.0:
            status = "❌"
            problem_tables.append((i, total_width, "超宽"))
            print(f"   ❌ 超宽！超出页面 {(total_width-1.0)*100:.1f}%")
        elif total_width > 0.95:
            status = "⚠️"
            problem_tables.append((i, total_width, "接近边界"))
            print(f"   ⚠️  接近边界 ({total_width*100:.1f}%页面宽度)")
        elif has_issues:
            status = "⚠️"  
            print(f"   ⚠️  使用cm单位，可能在不同设备上不一致")
        else:
            print(f"   ✅ 宽度合理 ({total_width*100:.1f}%页面宽度)")
    
    # 总结报告
    print("\n" + "=" * 70)
    print("📋 检查总结:")
    print("=" * 70)
    
    if len(matches) == 0:
        print("⚠️  未找到tabular表格定义")
    else:
        print(f"✅ 共检查 {len(matches)} 个表格")
        
        if problem_tables:
            print(f"\n❌ 发现 {len(problem_tables)} 个问题表格:")
            for table_num, width, issue in problem_tables:
                print(f"   - 表格 {table_num}: {issue} (宽度: {width:.3f})")
        else:
            print("\n🎉 所有表格宽度都在合理范围内！")
    
    print("=" * 70)

if __name__ == "__main__":
    tex_file = "FP_2025_IEEE-ACCESS_v5.tex"
    check_table_widths(tex_file)