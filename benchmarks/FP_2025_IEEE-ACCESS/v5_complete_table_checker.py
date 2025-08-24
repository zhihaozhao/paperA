#!/usr/bin/env python3
"""
完整表格宽度检查脚本 - 精确版
"""
import re

def extract_complete_table_definitions():
    """提取所有完整的表格定义"""
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("=" * 70)
    print("🔍 完整表格定义分析")
    print("=" * 70)
    
    # 手动搜索关键表格
    critical_tables = {
        'Algorithm Comparison (Table 3)': r'\\label\{tab:algorithm_comparison\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Dataset (Table 6)': r'\\label\{tab:dataset\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Literature Support Summary': r'\\label\{tab:literature_support_summary\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'IEEE Meta Summary': r'\\label\{tab:ieee_meta_summary\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'R-CNN Based': r'\\label\{tab:RCNN-based\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'YOLO Based': r'\\label\{tab:yolo-based\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Performance Metrics': r'\\label\{tab:performance-metrics\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Motion Control Enhanced': r'\\label\{tab:motion_control_enhanced\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Motion Control KPIs': r'\\label\{tab:motion_control_kpis\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Motion Control Based': r'\\label\{tab:motion-control-based\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Figure 4 Support': r'\\label\{tab:figure4_support\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Figure 9 Support': r'\\label\{tab:figure9_support\}.*?\\begin\{tabular\}\{([^}]+)\}',
        'Figure 10 Support': r'\\label\{tab:figure10_support\}.*?\\begin\{tabular\}\{([^}]+)\}'
    }
    
    problems_found = []
    
    for table_name, pattern in critical_tables.items():
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            tabular_def = match.group(1)
            print(f"\n📋 {table_name}:")
            print(f"   完整定义: {tabular_def}")
            
            # 提取p{width}
            p_widths = re.findall(r'p\{([0-9.]+)\\textwidth\}', tabular_def)
            
            if p_widths:
                total_width = sum(float(w) for w in p_widths)
                print(f"   📏 总宽度: {total_width:.3f} = {total_width*100:.1f}%")
                
                if total_width > 1.0:
                    print(f"   🔴 超宽 {(total_width-1.0)*100:.1f}%")
                    problems_found.append((table_name, total_width, "超宽"))
                elif total_width > 0.98:
                    print(f"   ⚠️  接近边界")
                    problems_found.append((table_name, total_width, "接近边界"))
                elif total_width < 0.30:
                    print(f"   🟡 过窄，可能重叠")
                    problems_found.append((table_name, total_width, "过窄"))
                else:
                    print(f"   ✅ 宽度合理")
            else:
                # 检查其他列类型
                other_patterns = ['@{}', 'c', 'l', 'r', '|']
                found_others = any(pat in tabular_def for pat in other_patterns)
                if found_others:
                    print(f"   ⚪ 使用其他列类型: {tabular_def}")
                else:
                    print(f"   ❓ 无法解析")
        else:
            print(f"\n❌ {table_name}: 未找到")
    
    return problems_found

def check_specific_problems():
    """检查特定的已知问题表格"""
    
    print(f"\n" + "=" * 70)
    print("🔧 特定问题检查")  
    print("=" * 70)
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查具体的表格定义
    specific_checks = [
        ('Algorithm Comparison', r'p\{0\.10\\textwidth\}p\{0\.12\\textwidth\}p\{0\.08\\textwidth\}p\{0\.08\\textwidth\}p\{0\.10\\textwidth\}p\{0\.08\\textwidth\}p\{0\.12\\textwidth\}p\{0\.20\\textwidth\}'),
        ('Motion Control Enhanced', r'p\{0\.14\\textwidth\}p\{0\.07\\textwidth\}p\{0\.11\\textwidth\}p\{0\.14\\textwidth\}p\{0\.17\\textwidth\}p\{0\.11\\textwidth\}p\{0\.19\\textwidth\}'),
        ('Motion Control KPIs', r'p\{0\.18\\textwidth\}p\{0\.17\\textwidth\}p\{0\.20\\textwidth\}p\{0\.37\\textwidth\}'),
        ('R-CNN Based', r'p\{0\.03\\textwidth\}p\{0\.08\\textwidth\}p\{0\.08\\textwidth\}p\{0\.12\\textwidth\}p\{0\.22\\textwidth\}p\{0\.25\\textwidth\}'),
    ]
    
    for name, pattern in specific_checks:
        if re.search(pattern, content):
            # 计算宽度
            widths = re.findall(r'([0-9.]+)', pattern)
            total = sum(float(w) for w in widths)
            print(f"✅ {name}: {total:.2f} = {total*100:.0f}% - 已修复")
        else:
            print(f"❌ {name}: 未找到修复的定义")

if __name__ == "__main__":
    print("🚀 完整表格检查分析...")
    
    # 完整分析
    problems = extract_complete_table_definitions()
    
    # 特定检查
    check_specific_problems()
    
    print(f"\n" + "=" * 70)
    print(f"📊 检查结果总结")
    print(f"发现 {len(problems)} 个问题表格")
    for name, width, issue in problems:
        print(f"   🔍 {name}: {width:.2f} ({issue})")
    print("=" * 70)