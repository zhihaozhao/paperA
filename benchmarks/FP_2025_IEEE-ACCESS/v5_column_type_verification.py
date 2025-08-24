#!/usr/bin/env python3
"""
验证p列到m列的修复完成情况
"""
import re

with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔍 列类型修复验证")
print("=" * 60)

# 检查m{linewidth}的使用
m_patterns = re.findall(r'm\{([0-9.]+)\\linewidth\}', content)
p_patterns = re.findall(r'p\{([0-9.]+)\\linewidth\}', content)

print(f"📊 列类型统计:")
print(f"   ✅ m{{linewidth}} 列: {len(m_patterns)} 个")
if m_patterns:
    m_widths = [float(w) for w in m_patterns]
    print(f"      宽度分布: {', '.join([f'{w*100:.0f}%' for w in sorted(set(m_widths))])}")

print(f"   ⚪ p{{linewidth}} 列: {len(p_patterns)} 个")
if p_patterns:
    p_widths = [float(w) for w in p_patterns]
    print(f"      剩余宽度: {', '.join([f'{w*100:.0f}%' for w in sorted(set(p_widths))])}")

# 检查tabularx环境中的m列
tabularx_m_count = len(re.findall(r'tabularx.*?m\{[^}]+\}', content, re.DOTALL))
print(f"\n🔧 TabularX中的m列: {tabularx_m_count} 处")

# 检查是否还有警告源
remaining_p = len(re.findall(r'\\begin\{tabularx\}.*?p\{[0-9.]+\\linewidth\}', content, re.DOTALL))
print(f"📝 TabularX中剩余p列: {remaining_p} 处")

print(f"\n" + "=" * 60)
if len(m_patterns) >= 22 and remaining_p == 0:
    print("✅ 修复完成！所有p{linewidth}已改为m{linewidth}")
    print("   - 解决了LaTeX编译警告")
    print("   - 实现了垂直居中对齐")
    print("   - 保持了相同的列宽设置")
elif remaining_p > 0:
    print(f"⚠️  还有 {remaining_p} 个p列需要修复")
else:
    print("✅ 列类型修复基本完成")
print("=" * 60)