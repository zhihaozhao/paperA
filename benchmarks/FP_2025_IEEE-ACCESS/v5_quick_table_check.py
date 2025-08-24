#!/usr/bin/env python3
"""
快速表格宽度验证
"""
import re

with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
    content = f.read()

print("🔍 当前表格宽度验证")
print("=" * 60)

# 直接匹配所有表格定义
patterns = [
    ('Algorithm Comparison', r'p\{0\.09\\textwidth\}p\{0\.11\\textwidth\}p\{0\.08\\textwidth\}p\{0\.08\\textwidth\}p\{0\.10\\textwidth\}p\{0\.08\\textwidth\}p\{0\.11\\textwidth\}p\{0\.22\\textwidth\}'),
    ('Dataset Table 6', r'p\{0\.04\\textwidth\}p\{0\.04\\textwidth\}p\{0\.14\\textwidth\}p\{0\.06\\textwidth\}p\{0\.08\\textwidth\}p\{0\.18\\textwidth\}p\{0\.24\\textwidth\}p\{0\.14\\textwidth\}'),  
    ('IEEE Meta Summary', r'p\{0\.16\\textwidth\}p\{0\.06\\textwidth\}p\{0\.17\\textwidth\}p\{0\.15\\textwidth\}p\{0\.16\\textwidth\}p\{0\.22\\textwidth\}'),
    ('Performance Metrics', r'p\{0\.11\\textwidth\}p\{0\.16\\textwidth\}p\{0\.32\\textwidth\}p\{0\.23\\textwidth\}p\{0\.10\\textwidth\}'),
    ('Motion Control Enhanced', r'p\{0\.14\\textwidth\}p\{0\.07\\textwidth\}p\{0\.11\\textwidth\}p\{0\.14\\textwidth\}p\{0\.17\\textwidth\}p\{0\.11\\textwidth\}p\{0\.19\\textwidth\}'),
    ('Motion Control KPIs', r'p\{0\.18\\textwidth\}p\{0\.17\\textwidth\}p\{0\.20\\textwidth\}p\{0\.37\\textwidth\}'),
]

for name, pattern in patterns:
    if re.search(pattern, content):
        widths = re.findall(r'([0-9.]+)', pattern)
        total = sum(float(w) for w in widths)
        status = "✅" if total < 0.95 else "⚠️" if total < 1.0 else "❌"
        print(f"{status} {name}: {total:.2f} = {total*100:.0f}%")
    else:
        print(f"❓ {name}: 未找到")

print("=" * 60)
print("✅ = 安全(<95%)  ⚠️ = 边界(95-100%)  ❌ = 超宽(>100%)")