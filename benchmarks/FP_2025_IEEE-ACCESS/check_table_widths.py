#!/usr/bin/env python3
"""
LaTeX表格宽度检查脚本
检查所有表格的列宽总和是否超出页面宽度
"""

import re

def check_table_widths(tex_file):
    """检查LaTeX文件中所有表格的列宽"""
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有tabular环境的列定义
    tabular_pattern = r'\\begin\{tabular\}\{([^}]+)\}'
    matches = re.findall(tabular_pattern, content)
    
    print("=" * 60)
    print("📊 LaTeX表格宽度分析报告")
    print("=" * 60)
    
    for i, match in enumerate(matches, 1):
        print(f"\n🔍 表格 {i}: {match}")
        
        # 提取p{width}定义
        p_widths = re.findall(r'p\{([^}]+)\}', match)
        
        if not p_widths:
            print("   ✅ 无固定列宽定义")
            continue
        
        total_width = 0
        width_details = []
        has_cm_units = False
        
        for width in p_widths:
            if 'textwidth' in width:
                # 提取textwidth系数
                coef_match = re.search(r'([0-9.]+)\\textwidth', width)
                if coef_match:
                    coef = float(coef_match.group(1))
                    total_width += coef
                    width_details.append(f"{coef:.3f}")
            elif 'linewidth' in width:
                # linewidth通常等于textwidth
                coef_match = re.search(r'([0-9.]+)\\linewidth', width)
                if coef_match:
                    coef = float(coef_match.group(1))
                    total_width += coef
                    width_details.append(f"{coef:.3f}(lw)")
            elif 'cm' in width:
                has_cm_units = True
                cm_match = re.search(r'([0-9.]+)cm', width)
                if cm_match:
                    cm_val = float(cm_match.group(1))
                    # 假设页面宽度约17cm
                    coef = cm_val / 17.0
                    total_width += coef
                    width_details.append(f"{cm_val}cm({coef:.3f})")
        
        # 显示分析结果
        print(f"   📏 列宽: {' + '.join(width_details)}")
        print(f"   📐 总宽度: {total_width:.3f} \\textwidth")
        
        if has_cm_units:
            print("   ⚠️  包含cm单位 - 可能在不同设备上显示不一致")
        
        if total_width > 1.0:
            print(f"   ❌ 超宽！超出 {(total_width-1.0)*100:.1f}%")
        elif total_width > 0.95:
            print(f"   ⚠️  接近边界 ({total_width*100:.1f}%)")
        else:
            print(f"   ✅ 宽度合理 ({total_width*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("📋 检查完成！")
    if len(matches) == 0:
        print("⚠️  未找到tabular表格定义")
    else:
        print(f"✅ 共检查 {len(matches)} 个表格")
    print("=" * 60)

if __name__ == "__main__":
    tex_file = "FP_2025_IEEE-ACCESS_v5.tex"
    check_table_widths(tex_file)