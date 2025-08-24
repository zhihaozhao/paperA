#!/usr/bin/env python3
"""
简单列宽验证
计算实际的百分比总和
"""
import re

with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔍 相对百分比列宽验证")
print("=" * 60)

# 查找所有p{百分比\linewidth}定义
linewidth_patterns = re.findall(r'p\{([0-9.]+)\\linewidth\}', content)

if linewidth_patterns:
    widths = [float(w) for w in linewidth_patterns]
    total_width = sum(widths)
    
    print(f"📊 发现 {len(widths)} 个百分比列定义:")
    for i, width in enumerate(widths, 1):
        print(f"   列{i}: {width*100:.0f}%")
    
    print(f"\n📏 各表格大致宽度分配:")
    # 按表格分组（每个表格大概6-8列）
    table_widths = []
    start = 0
    for table_cols in [8, 8, 6, 5, 7, 4]:  # 预估每个表格的列数
        if start < len(widths):
            end = min(start + table_cols, len(widths))
            table_width = sum(widths[start:end])
            table_widths.append(table_width)
            print(f"   表格{len(table_widths)}: {table_width*100:.0f}% ({end-start}列)")
            start = end
    
    print(f"\n📈 宽度统计:")
    print(f"   最大表格宽度: {max(table_widths)*100:.0f}%")
    print(f"   平均表格宽度: {sum(table_widths)/len(table_widths)*100:.0f}%")
    print(f"   总使用百分比: {total_width*100:.0f}%")
else:
    print("❌ 未找到百分比列宽定义")

# 检查tabularx环境
tabularx_count = len(re.findall(r'\\begin\{tabularx\}', content))
print(f"\n🔧 TabularX环境: {tabularx_count} 个")

# 检查自动换行设置
raggedright_count = len(re.findall(r'raggedright\\\\arraybackslash', content))
print(f"📝 自动换行设置: {raggedright_count} 处")

print(f"\n" + "=" * 60)
if linewidth_patterns and max(table_widths) < 0.95:
    print("✅ 列宽配置合理，应已解决越界问题")
elif linewidth_patterns:
    print("⚠️  部分表格可能仍有越界风险")
else:
    print("❌ 需要使用相对百分比列宽")
print("=" * 60)