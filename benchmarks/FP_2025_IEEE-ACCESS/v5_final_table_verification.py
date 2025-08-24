#!/usr/bin/env python3
"""
最终表格状态验证
检查tabularx转换完成情况
"""
import re

with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔍 最终表格状态验证")
print("=" * 70)

# 检查tabularx转换
tabularx_count = len(re.findall(r'\\begin\{tabularx\}', content))
tabular_count = len(re.findall(r'\\begin\{tabular\}', content))

print(f"📊 表格环境统计:")
print(f"   ✅ tabularx 表格: {tabularx_count}")
print(f"   ⚪ tabular 表格: {tabular_count}")

# 检查关键表格是否已转换
key_tables = [
    ('Algorithm Comparison', r'tab:algorithm_comparison'),
    ('Dataset (Table 6)', r'tab:dataset'),
    ('Performance Metrics', r'tab:performance-metrics'),
    ('Motion Control Enhanced', r'tab:motion_control_enhanced'),
    ('Motion Control KPIs', r'tab:motion_control_kpis'),
    ('IEEE Meta Summary', r'tab:ieee_meta_summary'),
]

print(f"\n📋 关键表格转换状态:")
converted_count = 0

for name, label_pattern in key_tables:
    # 查找表格标签后的tabularx环境
    pattern = rf'{label_pattern}.*?\\begin\{{(tabularx|tabular)\}}'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        env_type = match.group(1)
        if env_type == 'tabularx':
            print(f"   ✅ {name}: 已转换为tabularx")
            converted_count += 1
        else:
            print(f"   ❌ {name}: 仍使用tabular")
    else:
        print(f"   ❓ {name}: 未找到")

# 检查是否还有固定宽度定义
fixed_width_patterns = re.findall(r'p\{[0-9.]+\\textwidth\}', content)
if fixed_width_patterns:
    print(f"\n⚠️  发现 {len(fixed_width_patterns)} 个固定宽度列定义:")
    unique_patterns = list(set(fixed_width_patterns))[:5]  # 显示前5个
    for pattern in unique_patterns:
        print(f"      {pattern}")
    if len(fixed_width_patterns) > 5:
        print(f"      ... 还有 {len(fixed_width_patterns)-5} 个")
else:
    print(f"\n✅ 未发现固定宽度列定义")

# 检查X列类型使用情况
x_columns = len(re.findall(r'[^a-zA-Z]X[^a-zA-Z]', content))
print(f"\n📏 X列类型使用: {x_columns} 个自适应列")

print(f"\n" + "=" * 70)
print(f"📈 转换进度: {converted_count}/{len(key_tables)} 关键表格已转换")

if converted_count == len(key_tables) and len(fixed_width_patterns) < 10:
    print("🎉 tabularx转换基本完成！表格应已解决重叠和超宽问题")
elif converted_count >= len(key_tables) * 0.8:
    print("👍 大部分关键表格已转换，应已大幅改善重叠问题")  
else:
    print("⚠️  仍需完成更多表格转换")

print("=" * 70)