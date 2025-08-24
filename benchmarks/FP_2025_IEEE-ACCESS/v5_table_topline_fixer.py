#!/usr/bin/env python3
"""
V5表格topline修复脚本
专门解决 \toprule 线条超出页面宽度的问题
"""

import re

def analyze_table_widths():
    """分析所有表格的实际宽度"""
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("=" * 70)
    print("🔍 表格Topline宽度分析")
    print("=" * 70)
    
    # 更精确的表格模式匹配
    table_patterns = [
        # 匹配完整的tabular定义，包括换行
        r'\\begin\{tabular\}\{([^}]+)\}',
        # 匹配可能跨行的定义
        r'\\begin\{tabular\}\{([^}]*(?:\n[^}]*)*)\}'
    ]
    
    problematic_tables = []
    
    # 找所有表格定义和它们的上下文
    all_matches = []
    for pattern in table_patterns:
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
        all_matches.extend(matches)
    
    # 去重并排序
    seen_positions = set()
    unique_matches = []
    for match in sorted(all_matches, key=lambda x: x.start()):
        if match.start() not in seen_positions:
            unique_matches.append(match)
            seen_positions.add(match.start())
    
    for i, match in enumerate(unique_matches, 1):
        # 获取表格定义
        tabular_def = match.group(1).replace('\n', '').replace(' ', '')
        
        # 获取上下文，寻找表格标签
        start_pos = max(0, match.start() - 200)
        end_pos = min(len(content), match.end() + 100)
        context = content[start_pos:end_pos]
        
        # 提取表格标签
        label_match = re.search(r'\\label\{([^}]+)\}', context)
        table_label = label_match.group(1) if label_match else f"Table_{i}"
        
        print(f"\n📋 {table_label}:")
        print(f"   定义: {tabular_def}")
        
        # 提取所有p{width}定义
        p_widths = re.findall(r'p\{([0-9.]+)\\textwidth\}', tabular_def)
        
        if p_widths:
            total_width = sum(float(w) for w in p_widths)
            print(f"   📏 列宽总和: {total_width:.3f} = {total_width*100:.1f}%")
            
            # 分类问题严重程度
            if total_width > 1.05:
                status = "🔴 严重超宽"
                problematic_tables.append((table_label, tabular_def, total_width, "严重超宽"))
            elif total_width > 1.0:
                status = "🟡 超宽"
                problematic_tables.append((table_label, tabular_def, total_width, "轻微超宽"))
            elif total_width > 0.98:
                status = "⚠️  接近边界"
                problematic_tables.append((table_label, tabular_def, total_width, "接近边界"))
            else:
                status = "✅ 宽度安全"
            
            print(f"   状态: {status}")
            
            # 检查表格环境类型
            table_env_match = re.search(r'\\begin\{(table\*?)\}', context)
            if table_env_match:
                env_type = table_env_match.group(1)
                print(f"   环境: {env_type}")
                
                if env_type == "table" and total_width > 0.48:
                    print("   💡 建议: 单栏表格可能需要table*环境")
                elif env_type == "table*" and total_width > 0.95:
                    print("   💡 建议: 双栏表格需要缩减列宽")
        else:
            # 检查是否有其他列类型
            other_cols = re.findall(r'[lcr]', tabular_def)
            if other_cols:
                print(f"   ⚪ 使用其他列类型: {len(other_cols)}列")
            else:
                print("   ❓ 未识别的列定义")
    
    return problematic_tables

def generate_width_fixes():
    """生成具体的宽度修复方案"""
    
    print(f"\n" + "=" * 70)
    print("🔧 Topline修复方案")
    print("=" * 70)
    
    # 具体的修复建议
    fixes = {
        # Motion Control Enhanced: 100% -> 93%
        r'p\{0\.15\\textwidth\}p\{0\.08\\textwidth\}p\{0\.12\\textwidth\}p\{0\.15\\textwidth\}p\{0\.18\\textwidth\}p\{0\.12\\textwidth\}p\{0\.20\\textwidth\}': {
            'original_width': 1.00,
            'fixed_def': 'p{0.14\\textwidth}p{0.07\\textwidth}p{0.11\\textwidth}p{0.14\\textwidth}p{0.17\\textwidth}p{0.11\\textwidth}p{0.19\\textwidth}',
            'new_width': 0.93,
            'reason': '缩减7%避免topline超长'
        },
        
        # Motion Control KPIs: 100% -> 92%
        r'p\{0\.20\\textwidth\}p\{0\.18\\textwidth\}p\{0\.22\\textwidth\}p\{0\.40\\textwidth\}': {
            'original_width': 1.00,
            'fixed_def': 'p{0.18\\textwidth}p{0.17\\textwidth}p{0.20\\textwidth}p{0.37\\textwidth}',
            'new_width': 0.92,
            'reason': '缩减8%防止topline超页面'
        },
        
        # R-CNN based table: 70% (细列问题)
        r'p\{0\.018\\textwidth\}p\{0\.065\\textwidth\}p\{0\.060\\textwidth\}p\{0\.105\\textwidth\}p\{0\.185\\textwidth\}p\{0\.195\\textwidth\}': {
            'original_width': 0.628,
            'fixed_def': 'p{0.03\\textwidth}p{0.08\\textwidth}p{0.08\\textwidth}p{0.12\\textwidth}p{0.22\\textwidth}p{0.25\\textwidth}',
            'new_width': 0.78,
            'reason': '增宽过窄列，改善可读性'
        }
    }
    
    print("🎯 具体修复方案:")
    for pattern, fix_info in fixes.items():
        print(f"\n📋 宽度修复:")
        print(f"   原始: {fix_info['original_width']:.2f} → 修复: {fix_info['new_width']:.2f}")
        print(f"   原因: {fix_info['reason']}")
        print(f"   修复定义: {fix_info['fixed_def']}")
    
    return fixes

def create_universal_table_wrapper():
    """创建通用的表格包装器建议"""
    
    print(f"\n" + "=" * 70)
    print("📦 通用表格包装器")
    print("=" * 70)
    
    wrapper_code = """
% 防止topline超长的通用包装器
\\newcommand{\\safetable}[3]{%
    % #1: 表格标题
    % #2: 表格标签  
    % #3: 表格内容
    \\begin{table*}[htbp]
    \\centering
    \\small
    \\renewcommand{\\arraystretch}{1.1}
    \\captionsetup{width=0.9\\textwidth}
    \\caption{#1}
    \\label{#2}
    \\resizebox{0.95\\textwidth}{!}{%
        #3
    }
    \\end{table*}
}

% 使用示例:
\\safetable{您的表格标题}{tab:your_label}{%
    \\begin{tabular}{p{0.15\\textwidth}p{0.25\\textwidth}p{0.35\\textwidth}p{0.25\\textwidth}}
    \\toprule
    列1 & 列2 & 列3 & 列4 \\\\
    \\midrule
    内容 & 自动换行内容 & 更长的内容会自动换行 & 引用内容 \\\\
    \\bottomrule
    \\end{tabular}%
}
"""
    
    print(wrapper_code)
    
    return wrapper_code

if __name__ == "__main__":
    print("🚀 启动Topline超长修复分析...")
    
    # 分析表格宽度
    problematic = analyze_table_widths()
    
    # 生成修复方案
    fixes = generate_width_fixes()
    
    # 创建通用包装器
    wrapper = create_universal_table_wrapper()
    
    print(f"\n" + "=" * 70)
    print("✅ 修复分析完成！")
    print(f"发现 {len(problematic)} 个问题表格")
    print("🔧 立即修复建议:")
    print("   1. 超宽表格缩减列宽至 < 95%")
    print("   2. 使用 \\resizebox{0.95\\textwidth}{!}{...} 包装")
    print("   3. 过窄列适当增宽，改善可读性")
    print("   4. 统一使用 table* 环境")
    print("=" * 70)