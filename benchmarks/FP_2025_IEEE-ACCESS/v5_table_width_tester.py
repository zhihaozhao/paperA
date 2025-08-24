#!/usr/bin/env python3
"""
表格宽度测试工具
帮助诊断tabularx宽度不变的问题
"""
import re

def analyze_tabularx_issues():
    """分析可能导致tabularx宽度不变的问题"""
    
    with open('FP_2025_IEEE-ACCESS_v5.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔍 TabularX宽度不变问题诊断")
    print("=" * 70)
    
    # 1. 检查tabularx包是否正确引入
    tabularx_imports = re.findall(r'\\usepackage.*tabularx', content)
    print(f"📦 TabularX包导入: {len(tabularx_imports)} 处")
    for imp in tabularx_imports:
        print(f"   {imp}")
    
    # 2. 检查表格环境
    tabularx_envs = re.findall(r'\\begin\{tabularx\}\{\\textwidth\}', content)
    print(f"\n📋 TabularX环境: {len(tabularx_envs)} 个")
    
    # 3. 检查列定义中的问题
    print(f"\n🔧 潜在问题检查:")
    
    # 检查是否有过长的列定义
    long_tabularx = re.findall(r'\\begin\{tabularx\}\{\\textwidth\}\{[^}]{100,}\}', content)
    if long_tabularx:
        print(f"   ⚠️  发现 {len(long_tabularx)} 个过长的列定义，可能导致解析问题")
    
    # 检查是否有缺失的包
    array_pkg = re.findall(r'\\usepackage.*array', content)
    if not array_pkg:
        print(f"   ⚠️  可能缺少array包，影响列类型解析")
    else:
        print(f"   ✅ array包已导入")
    
    # 检查换行符问题
    broken_tabularx = re.findall(r'\\begin\{tabularx\}[^{]*\n[^{]*\{', content)
    if broken_tabularx:
        print(f"   ⚠️  发现 {len(broken_tabularx)} 个跨行的tabularx定义")
    
    return True

def create_width_test_table():
    """创建宽度测试表格"""
    
    print(f"\n" + "=" * 70)
    print("📝 表格宽度测试代码")
    print("=" * 70)
    
    test_code = r"""
% 表格宽度测试 - 复制到LaTeX文档中测试
\begin{table*}[htbp]
\centering
\small
\renewcommand{\arraystretch}{1.2}
\caption{表格宽度测试}
\label{tab:width_test}

% 测试1: 简单的宽度设置
\begin{tabularx}{\textwidth}{m{0.20\linewidth}m{0.30\linewidth}m{0.40\linewidth}}
\toprule
\textbf{20\%列} & \textbf{30\%列} & \textbf{40\%列} \\
\midrule
短内容 & 中等长度的内容，用来测试自动换行功能 & 很长的内容，这应该展示40\%的列宽效果，内容会在这个列中自动换行显示 \\
\bottomrule
\end{tabularx}
\end{table*}

% 测试2: 混合列类型
\begin{table*}[htbp]
\centering
\small
\caption{混合列类型测试}
\begin{tabularx}{\textwidth}{cm{0.30\linewidth}m{0.50\linewidth}}
\toprule
\textbf{数值} & \textbf{30\%列} & \textbf{50\%列} \\
\midrule
1 & 中等内容测试 & 长内容测试，这里应该显示50\%的列宽效果 \\
2 & 另一行测试 & 再次测试长内容的自动换行功能 \\
\bottomrule
\end{tabularx}
\end{table*}
"""
    
    print(test_code)
    return test_code

def diagnose_common_issues():
    """诊断常见的tabularx问题"""
    
    print(f"\n" + "=" * 70)
    print("🚨 常见TabularX宽度问题及解决方案")
    print("=" * 70)
    
    issues = [
        {
            "问题": "表格宽度不变",
            "原因": [
                "LaTeX没有重新编译",
                "缓存文件干扰",
                "PDF查看器没有刷新",
                "列定义语法错误"
            ],
            "解决方案": [
                "清理*.aux, *.log等缓存文件",
                "强制重新编译LaTeX",
                "刷新PDF查看器",
                "检查列定义语法"
            ]
        },
        {
            "问题": "列宽不生效",
            "原因": [
                "tabularx包未正确导入",
                "使用了错误的表格环境",
                "列类型定义错误",
                "内容过长导致强制拉伸"
            ],
            "解决方案": [
                "确保\\usepackage{tabularx}",
                "使用table*环境而非table",
                "检查m{width}语法正确性",
                "适当增加\\arraystretch"
            ]
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"\n🔍 问题 {i}: {issue['问题']}")
        print("   可能原因:")
        for cause in issue["原因"]:
            print(f"     - {cause}")
        print("   解决方案:")
        for solution in issue["解决方案"]:
            print(f"     ✅ {solution}")

if __name__ == "__main__":
    print("🚀 开始TabularX宽度问题诊断...")
    
    # 诊断分析
    analyze_tabularx_issues()
    
    # 创建测试代码
    create_width_test_table()
    
    # 诊断常见问题
    diagnose_common_issues()
    
    print(f"\n" + "=" * 70)
    print("💡 建议的调试步骤:")
    print("1. 在本地LaTeX编辑器中测试上面的测试代码")
    print("2. 清理所有缓存文件后重新编译")
    print("3. 确保PDF查看器刷新显示最新版本") 
    print("4. 如果仍有问题，逐个检查表格列定义")
    print("=" * 70)