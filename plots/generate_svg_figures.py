#!/usr/bin/env python3
"""
IEEE IoTJ Figure 3 & 4 SVG生成脚本 - 无依赖版本
用于预览图表布局、文字大小、图例位置等设计问题

Author: Generated for PaperA submission  
Date: 2025
"""

import math

def create_svg_figure3():
    """生成Figure 3 SVG预览 - Cross-Domain Generalization"""
    print("生成Figure 3 SVG预览...")
    
    # IEEE IoTJ规范: 17.1cm × 10cm = 648px × 378px @ 96dpi
    width, height = 648, 378
    margin = 50
    plot_width = width - 2*margin
    plot_height = height - 2*margin - 40  # 标题空间
    
    # 数据
    models = ['Enhanced', 'CNN', 'BiLSTM', 'Conformer']
    loso_scores = [0.830, 0.842, 0.803, 0.403]
    loso_errors = [0.001, 0.025, 0.022, 0.386]
    loro_scores = [0.830, 0.796, 0.789, 0.841]
    loro_errors = [0.001, 0.097, 0.044, 0.040]
    
    # 颜色
    colors = {
        'Enhanced': '#2E86AB',
        'CNN': '#E84855', 
        'BiLSTM': '#3CB371',
        'Conformer': '#DC143C'
    }
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <!-- 背景 -->
    <rect width="{width}" height="{height}" fill="white" stroke="black" stroke-width="1"/>
    
    <!-- 标题 -->
    <text x="{width//2}" y="25" text-anchor="middle" font-family="Times New Roman" font-size="12" font-weight="bold">
        Cross-Domain Generalization Performance
    </text>
    
    <!-- 绘图区域边框 -->
    <rect x="{margin}" y="40" width="{plot_width}" height="{plot_height}" 
          fill="none" stroke="black" stroke-width="0.5"/>
    
    <!-- Y轴网格线和标签 -->'''
    
    # Y轴网格和标签
    for i in range(6):  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        y_val = i * 0.2
        y_pos = 40 + plot_height - (y_val * plot_height)
        svg_content += f'''
    <line x1="{margin}" y1="{y_pos}" x2="{margin + plot_width}" y2="{y_pos}" 
          stroke="gray" stroke-width="0.25" opacity="0.3"/>
    <text x="{margin-5}" y="{y_pos+3}" text-anchor="end" font-family="Times New Roman" font-size="9">
        {y_val:.1f}
    </text>'''
    
    # Y轴标题
    svg_content += f'''
    <text x="20" y="{40 + plot_height//2}" text-anchor="middle" font-family="Times New Roman" 
          font-size="10" transform="rotate(-90, 20, {40 + plot_height//2})">
        Macro F1 Score
    </text>'''
    
    # 柱状图
    bar_width = 30
    spacing = plot_width / len(models)
    
    for i, model in enumerate(models):
        x_center = margin + spacing * (i + 0.5)
        x_loso = x_center - bar_width//2 - 8
        x_loro = x_center + bar_width//2 + 8
        
        # LOSO柱状
        loso_height = loso_scores[i] * plot_height
        loso_y = 40 + plot_height - loso_height
        
        # 特殊处理Enhanced模型 - 边框加粗
        stroke_width = "1.5" if model == "Enhanced" else "0.5"
        
        svg_content += f'''
    <!-- {model} LOSO柱状 -->
    <rect x="{x_loso-bar_width//2}" y="{loso_y}" width="{bar_width}" height="{loso_height}"
          fill="{colors[model]}" stroke="black" stroke-width="{stroke_width}" opacity="0.9"/>
    
    <!-- {model} LOSO误差棒 -->
    <line x1="{x_loso}" y1="{loso_y - loso_errors[i]*plot_height}" 
          x2="{x_loso}" y2="{loso_y + loso_errors[i]*plot_height}" 
          stroke="black" stroke-width="0.5"/>
    <line x1="{x_loso-3}" y1="{loso_y - loso_errors[i]*plot_height}" 
          x2="{x_loso+3}" y2="{loso_y - loso_errors[i]*plot_height}" 
          stroke="black" stroke-width="0.5"/>
    <line x1="{x_loso-3}" y1="{loso_y + loso_errors[i]*plot_height}" 
          x2="{x_loso+3}" y2="{loso_y + loso_errors[i]*plot_height}" 
          stroke="black" stroke-width="0.5"/>
          
    <!-- {model} LOSO数值标签 -->
    <text x="{x_loso}" y="{loso_y - loso_errors[i]*plot_height - 5}" 
          text-anchor="middle" font-family="Times New Roman" font-size="8">
        {loso_scores[i]:.3f}±{loso_errors[i]:.3f}
    </text>'''
        
        # LORO柱状
        loro_height = loro_scores[i] * plot_height  
        loro_y = 40 + plot_height - loro_height
        
        svg_content += f'''
    <!-- {model} LORO柱状 -->
    <rect x="{x_loro-bar_width//2}" y="{loro_y}" width="{bar_width}" height="{loro_height}"
          fill="{colors[model]}" stroke="black" stroke-width="{stroke_width}" opacity="0.7"/>
    
    <!-- {model} LORO误差棒 -->
    <line x1="{x_loro}" y1="{loro_y - loro_errors[i]*plot_height}" 
          x2="{x_loro}" y2="{loro_y + loro_errors[i]*plot_height}" 
          stroke="black" stroke-width="0.5"/>
    <line x1="{x_loro-3}" y1="{loro_y - loro_errors[i]*plot_height}" 
          x2="{x_loro+3}" y2="{loro_y - loro_errors[i]*plot_height}" 
          stroke="black" stroke-width="0.5"/>
    <line x1="{x_loro-3}" y1="{loro_y + loro_errors[i]*plot_height}" 
          x2="{x_loro+3}" y2="{loro_y + loro_errors[i]*plot_height}" 
          stroke="black" stroke-width="0.5"/>
          
    <!-- {model} LORO数值标签 -->
    <text x="{x_loro}" y="{loro_y - loro_errors[i]*plot_height - 5}" 
          text-anchor="middle" font-family="Times New Roman" font-size="8">
        {loro_scores[i]:.3f}±{loro_errors[i]:.3f}
    </text>
    
    <!-- X轴标签 -->
    <text x="{x_center}" y="{40 + plot_height + 20}" text-anchor="middle" 
          font-family="Times New Roman" font-size="9">
        {model}
    </text>'''
    
    # X轴标题和图例
    svg_content += f'''
    
    <!-- X轴标题 -->
    <text x="{width//2}" y="{height-10}" text-anchor="middle" 
          font-family="Times New Roman" font-size="10">
        Model Architecture
    </text>
    
    <!-- 图例 -->
    <g transform="translate({width-120}, 60)">
        <rect width="110" height="40" fill="white" stroke="black" stroke-width="0.5" opacity="0.9"/>
        <!-- LOSO图例 -->
        <rect x="5" y="8" width="15" height="10" fill="#2E86AB" opacity="0.9"/>
        <text x="25" y="16" font-family="Times New Roman" font-size="9">LOSO</text>
        <!-- LORO图例 -->  
        <rect x="5" y="23" width="15" height="10" fill="#2E86AB" opacity="0.7"/>
        <text x="25" y="31" font-family="Times New Roman" font-size="9">LORO</text>
    </g>
    
</svg>'''
    
    return svg_content

def create_svg_figure4():
    """生成Figure 4 SVG预览 - Sim2Real Label Efficiency"""
    print("生成Figure 4 SVG预览...")
    
    # IEEE IoTJ规范: 17.1cm × 12cm = 648px × 454px @ 96dpi
    width, height = 648, 454
    margin = 50
    plot_width = width - 2*margin
    plot_height = height - 2*margin - 60  # 标题和标注空间
    
    # 数据
    label_ratios = [1.0, 5.0, 10.0, 20.0, 100.0]
    f1_scores = [0.455, 0.780, 0.730, 0.821, 0.833]
    std_errors = [0.050, 0.016, 0.104, 0.003, 0.000]
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <!-- 背景 -->
    <rect width="{width}" height="{height}" fill="white" stroke="black" stroke-width="1"/>
    
    <!-- 标题 -->
    <text x="{width//2}" y="25" text-anchor="middle" font-family="Times New Roman" 
          font-size="12" font-weight="bold">
        Sim2Real Label Efficiency Breakthrough
    </text>
    
    <!-- 绘图区域边框 -->
    <rect x="{margin}" y="50" width="{plot_width}" height="{plot_height}" 
          fill="none" stroke="black" stroke-width="0.5"/>
    
    <!-- 效率区域背景 (0-25%标签) -->
    <rect x="{margin}" y="50" width="{plot_width * 0.25}" height="{plot_height}" 
          fill="lightgreen" opacity="0.2"/>
    
    <!-- Y轴网格线和标签 (0.1-0.95) -->'''
    
    # Y轴网格
    y_min, y_max = 0.1, 0.95
    for i in range(9):  # 0.1, 0.2, ..., 0.9
        y_val = 0.1 + i * 0.1
        y_pos = 50 + plot_height - ((y_val - y_min) / (y_max - y_min) * plot_height)
        svg_content += f'''
    <line x1="{margin}" y1="{y_pos}" x2="{margin + plot_width}" y2="{y_pos}" 
          stroke="gray" stroke-width="0.25" opacity="0.3"/>
    <text x="{margin-5}" y="{y_pos+3}" text-anchor="end" font-family="Times New Roman" font-size="9">
        {y_val:.1f}
    </text>'''
    
    # 参考线
    target_y = 50 + plot_height - ((0.80 - y_min) / (y_max - y_min) * plot_height)
    ideal_y = 50 + plot_height - ((0.90 - y_min) / (y_max - y_min) * plot_height)
    zero_y = 50 + plot_height - ((0.151 - y_min) / (y_max - y_min) * plot_height)
    
    svg_content += f'''
    
    <!-- 参考线 -->
    <line x1="{margin}" y1="{target_y}" x2="{margin + plot_width}" y2="{target_y}" 
          stroke="red" stroke-width="1.5" stroke-dasharray="5,5" opacity="0.8"/>
    <line x1="{margin}" y1="{ideal_y}" x2="{margin + plot_width}" y2="{ideal_y}" 
          stroke="orange" stroke-width="1.0" stroke-dasharray="2,2"/>
    <line x1="{margin}" y1="{zero_y}" x2="{margin + plot_width}" y2="{zero_y}" 
          stroke="gray" stroke-width="1.0"/>
    
    <!-- 主曲线数据点 -->
    <g stroke="#2E86AB" stroke-width="2.5" fill="#2E86AB">'''
    
    # 绘制数据点和连线
    points = []
    for i, (ratio, score, error) in enumerate(zip(label_ratios, f1_scores, std_errors)):
        # 对数坐标转换 (近似)
        x_log = margin + (math.log10(ratio) / 2) * plot_width  # log10(1) to log10(100) = 0 to 2
        y_pos = 50 + plot_height - ((score - y_min) / (y_max - y_min) * plot_height)
        points.append((x_log, y_pos))
        
        # 数据点
        svg_content += f'''
        <circle cx="{x_log}" cy="{y_pos}" r="4" stroke="black" stroke-width="0.5"/>'''
        
        # 误差棒
        error_height = error / (y_max - y_min) * plot_height
        svg_content += f'''
        <line x1="{x_log}" y1="{y_pos - error_height}" x2="{x_log}" y2="{y_pos + error_height}" 
              stroke="black" stroke-width="0.5"/>
        <line x1="{x_log-3}" y1="{y_pos - error_height}" x2="{x_log+3}" y2="{y_pos - error_height}" 
              stroke="black" stroke-width="0.5"/>
        <line x1="{x_log-3}" y1="{y_pos + error_height}" x2="{x_log+3}" y2="{y_pos + error_height}" 
              stroke="black" stroke-width="0.5"/>'''
        
        # 数据标签
        if ratio == 20.0:  # 关键点特殊标记
            svg_content += f'''
        <text x="{x_log}" y="{y_pos - error_height - 10}" text-anchor="middle" 
              font-family="Times New Roman" font-size="8" font-weight="bold" fill="#2E86AB">
            {score:.3f}±{error:.3f} ★
        </text>'''
        else:
            svg_content += f'''
        <text x="{x_log}" y="{y_pos - error_height - 8}" text-anchor="middle" 
              font-family="Times New Roman" font-size="8" fill="#2E86AB">
            {score:.3f}±{error:.3f}
        </text>'''
    
    # 连接线
    svg_content += '''
        <polyline points="'''
    for x, y in points:
        svg_content += f"{x},{y} "
    svg_content += '''" fill="none"/>
    </g>
    
    <!-- 关键点标注 (20%, 0.821) -->
    <g>
        <!-- 标注框 -->
        <rect x="400" y="80" width="180" height="35" fill="#FFFACD" 
              stroke="#FF6B6B" stroke-width="1" rx="3"/>
        <text x="490" y="95" text-anchor="middle" font-family="Times New Roman" 
              font-size="10" font-weight="bold">
            Key Achievement:
        </text>
        <text x="490" y="108" text-anchor="middle" font-family="Times New Roman" 
              font-size="10" font-weight="bold">
            82.1% F1 @ 20% Labels
        </text>
        
        <!-- 指向箭头 -->'''
    
    # 找到20%点的位置
    key_x = margin + (math.log10(20) / 2) * plot_width
    key_y = 50 + plot_height - ((0.821 - y_min) / (y_max - y_min) * plot_height)
    
    svg_content += f'''
        <line x1="490" y1="115" x2="{key_x}" y2="{key_y}" 
              stroke="#FF6B6B" stroke-width="1.5" marker-end="url(#arrowhead)"/>
    </g>
    
    <!-- 箭头标记定义 -->
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#FF6B6B"/>
        </marker>
    </defs>
    
    <!-- X轴标签 -->'''
    
    for ratio in [1, 5, 10, 20, 50, 100]:
        x_pos = margin + (math.log10(ratio) / 2) * plot_width
        svg_content += f'''
    <text x="{x_pos}" y="{50 + plot_height + 15}" text-anchor="middle" 
          font-family="Times New Roman" font-size="9">
        {ratio}
    </text>'''
    
    svg_content += f'''
    
    <!-- X轴标题 -->
    <text x="{width//2}" y="{height-10}" text-anchor="middle" 
          font-family="Times New Roman" font-size="10">
        Label Ratio (%)
    </text>
    
    <!-- 图例 -->
    <g transform="translate(450, 300)">
        <rect width="180" height="80" fill="white" stroke="black" stroke-width="0.5" opacity="0.9"/>
        <!-- 目标线图例 -->
        <line x1="10" y1="15" x2="25" y2="15" stroke="red" stroke-width="1.5" stroke-dasharray="5,5"/>
        <text x="30" y="18" font-family="Times New Roman" font-size="9">Target (80%)</text>
        <!-- 理想线图例 -->
        <line x1="10" y1="30" x2="25" y2="30" stroke="orange" stroke-width="1.0" stroke-dasharray="2,2"/>
        <text x="30" y="33" font-family="Times New Roman" font-size="9">Ideal (90%)</text>
        <!-- Zero-shot图例 -->
        <line x1="10" y1="45" x2="25" y2="45" stroke="gray" stroke-width="1.0"/>
        <text x="30" y="48" font-family="Times New Roman" font-size="9">Zero-shot</text>
        <!-- 主曲线图例 -->
        <circle cx="17.5" cy="60" r="4" fill="#2E86AB" stroke="black" stroke-width="0.5"/>
        <text x="30" y="63" font-family="Times New Roman" font-size="9">Enhanced Model</text>
    </g>
    
</svg>'''
    
    return svg_content

def main():
    """主函数 - 生成SVG预览"""
    print("🚀 IEEE IoTJ Figure 3 & 4 SVG预览生成")
    print("=" * 50)
    
    # 生成Figure 3 SVG
    svg3 = create_svg_figure3()
    with open('figure3_cross_domain_preview.svg', 'w', encoding='utf-8') as f:
        f.write(svg3)
    print("✓ Figure 3 SVG预览已保存: figure3_cross_domain_preview.svg")
    
    # 生成Figure 4 SVG  
    svg4 = create_svg_figure4()
    with open('figure4_sim2real_preview.svg', 'w', encoding='utf-8') as f:
        f.write(svg4)
    print("✓ Figure 4 SVG预览已保存: figure4_sim2real_preview.svg")
    
    print("\n📋 设计检查要点:")
    print("1. 文字大小检查:")
    print("   - 标题: 12pt (bold)")
    print("   - 坐标轴标签: 10pt")  
    print("   - 数据标签: 8pt")
    print("   - 图例: 9pt")
    
    print("\n2. 图例重叠检查:")
    print("   - Figure 3: 右上角，避开数据柱状")
    print("   - Figure 4: 右下角，避开关键标注")
    
    print("\n3. 视觉层次检查:")
    print("   - Enhanced模型边框加粗突出") 
    print("   - 关键点(20%, 82.1%)有特殊标注")
    print("   - 误差棒清晰可见")
    
    print("\n4. IEEE IoTJ规范:")
    print("   - Times New Roman字体")
    print("   - 色盲友好颜色")
    print("   - 适当的线宽和间距")
    
    print("\n🎉 SVG预览生成完成！请使用浏览器或矢量图编辑器查看效果。")

if __name__ == "__main__":
    main()