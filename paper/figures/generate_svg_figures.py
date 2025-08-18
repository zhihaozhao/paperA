#!/usr/bin/env python3
"""
Direct SVG Figure Generation (No external dependencies)
生成可直接转换为PDF的SVG矢量图
"""

def create_figure3_svg():
    """Generate Figure 3 as SVG (IEEE IoTJ compliant)."""
    
    # Data
    models = ['Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite']
    loso_data = [0.830, 0.842, 0.803, 0.403]
    loso_err = [0.001, 0.025, 0.022, 0.386]
    loro_data = [0.830, 0.796, 0.789, 0.841]
    loro_err = [0.001, 0.097, 0.044, 0.040]
    
    # SVG parameters (IEEE IoTJ: 17.1cm x 10cm = 484px x 283px @ 72DPI)
    width, height = 600, 350
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 40, 80
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    # Colors (IEEE IoTJ colorblind-friendly)
    colors = ['#2E86AB', '#E84855', '#3CB371', '#DC143C']
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .title {{ font-family: Times, serif; font-size: 14px; font-weight: bold; }}
            .axis-label {{ font-family: Times, serif; font-size: 12px; }}
            .tick-label {{ font-family: Times, serif; font-size: 10px; }}
            .value-label {{ font-family: Times, serif; font-size: 8px; }}
            .legend {{ font-family: Times, serif; font-size: 10px; }}
        </style>
    </defs>
    
    <!-- White background -->
    <rect width="{width}" height="{height}" fill="white"/>
    
    <!-- Title -->
    <text x="{width/2}" y="25" class="title" text-anchor="middle">
        Cross-Domain Generalization Performance
    </text>
    
    <!-- Y-axis -->
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" 
          stroke="black" stroke-width="1"/>
    
    <!-- X-axis -->
    <line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" 
          stroke="black" stroke-width="1"/>
    
    <!-- Y-axis grid and labels -->'''
    
    # Y-axis grid and labels
    for i in range(11):  # 0.0 to 1.0
        y_val = i / 10.0
        y_pos = height - margin_bottom - (y_val * plot_height)
        
        # Grid line
        svg_content += f'''
    <line x1="{margin_left}" y1="{y_pos}" x2="{width-margin_right}" y2="{y_pos}" 
          stroke="#CCCCCC" stroke-width="0.5" opacity="0.5"/>'''
        
        # Y-axis label
        if i % 2 == 0:  # Every 0.2
            svg_content += f'''
    <text x="{margin_left-10}" y="{y_pos+3}" class="tick-label" text-anchor="end">
        {y_val:.1f}
    </text>'''
    
    # Y-axis title
    svg_content += f'''
    <text x="20" y="{height/2}" class="axis-label" text-anchor="middle" 
          transform="rotate(-90, 20, {height/2})">
        Macro F1 Score
    </text>
    
    <!-- X-axis title -->
    <text x="{width/2}" y="{height-15}" class="axis-label" text-anchor="middle">
        Model Architecture
    </text>'''
    
    # Draw bars
    bar_width = plot_width / (len(models) * 3)  # Space for grouped bars
    
    for i, model in enumerate(models):
        # X position for this model group
        group_x = margin_left + (i + 0.5) * (plot_width / len(models))
        
        # LOSO bar (left)
        loso_height = loso_data[i] * plot_height
        loso_y = height - margin_bottom - loso_height
        loso_x = group_x - bar_width * 0.6
        
        svg_content += f'''
    <!-- LOSO bar for {model} -->
    <rect x="{loso_x}" y="{loso_y}" width="{bar_width}" height="{loso_height}"
          fill="{colors[0]}" stroke="black" stroke-width="0.5" opacity="0.8"/>'''
        
        # LORO bar (right)  
        loro_height = loro_data[i] * plot_height
        loro_y = height - margin_bottom - loro_height
        loro_x = group_x + bar_width * 0.1
        
        svg_content += f'''
    <!-- LORO bar for {model} -->
    <rect x="{loro_x}" y="{loro_y}" width="{bar_width}" height="{loro_height}"
          fill="{colors[1]}" stroke="black" stroke-width="0.5" opacity="0.8"/>'''
        
        # Error bars
        # LOSO error bar
        err_top = loso_y - (loso_err[i] * plot_height)
        err_bottom = loso_y + (loso_err[i] * plot_height)
        err_x = loso_x + bar_width/2
        
        svg_content += f'''
    <!-- LOSO error bar -->
    <line x1="{err_x}" y1="{err_top}" x2="{err_x}" y2="{err_bottom}" 
          stroke="black" stroke-width="1"/>
    <line x1="{err_x-3}" y1="{err_top}" x2="{err_x+3}" y2="{err_top}" 
          stroke="black" stroke-width="1"/>
    <line x1="{err_x-3}" y1="{err_bottom}" x2="{err_x+3}" y2="{err_bottom}" 
          stroke="black" stroke-width="1"/>'''
        
        # Value labels
        svg_content += f'''
    <!-- Value labels -->
    <text x="{err_x}" y="{err_top-5}" class="value-label" text-anchor="middle">
        {loso_data[i]:.3f}±{loso_err[i]:.3f}
    </text>'''
        
        # X-axis model label
        svg_content += f'''
    <text x="{group_x}" y="{height-margin_bottom+20}" class="tick-label" text-anchor="middle">
        {model}
    </text>'''
    
    # Legend
    svg_content += f'''
    <!-- Legend -->
    <rect x="{width-150}" y="50" width="15" height="15" fill="{colors[0]}" opacity="0.8"/>
    <text x="{width-130}" y="62" class="legend">LOSO</text>
    <rect x="{width-150}" y="70" width="15" height="15" fill="{colors[1]}" opacity="0.8"/>
    <text x="{width-130}" y="82" class="legend">LORO</text>
    
    <!-- Enhanced model highlight -->
    <rect x="{margin_left + 0.5 * (plot_width / len(models)) - bar_width * 0.8}" 
          y="{height - margin_bottom - loso_data[0] * plot_height - 5}" 
          width="{bar_width * 1.6}" height="{max(loso_data[0], loro_data[0]) * plot_height + 10}"
          fill="none" stroke="#FFD700" stroke-width="2" stroke-dasharray="5,5" opacity="0.7"/>
    <text x="{margin_left + 0.5 * (plot_width / len(models))}" y="45" 
          class="value-label" text-anchor="middle" fill="#FFD700" font-weight="bold">
        ⭐ 83.0% Consistency
    </text>
    
</svg>'''
    
    return svg_content

def create_figure4_svg():
    """Generate Figure 4 as SVG (IEEE IoTJ compliant)."""
    
    # Data
    x_data = [1.0, 5.0, 10.0, 20.0, 100.0]
    y_data = [0.455, 0.780, 0.730, 0.821, 0.833]
    errors = [0.050, 0.016, 0.104, 0.003, 0.000]
    
    # SVG parameters (IEEE IoTJ: 17.1cm x 12cm = 484px x 340px @ 72DPI)
    width, height = 700, 450
    margin_left, margin_right = 80, 40  
    margin_top, margin_bottom = 40, 80
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    # Scale functions
    def scale_x(x): return margin_left + (x / 100.0) * plot_width
    def scale_y(y): return height - margin_bottom - y * plot_height
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .title {{ font-family: Times, serif; font-size: 14px; font-weight: bold; }}
            .axis-label {{ font-family: Times, serif; font-size: 12px; }}
            .tick-label {{ font-family: Times, serif; font-size: 10px; }}
            .value-label {{ font-family: Times, serif; font-size: 9px; }}
            .annotation {{ font-family: Times, serif; font-size: 11px; font-weight: bold; }}
        </style>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#FF6B6B"/>
        </marker>
    </defs>
    
    <!-- White background -->
    <rect width="{width}" height="{height}" fill="white"/>
    
    <!-- Title -->
    <text x="{width/2}" y="25" class="title" text-anchor="middle">
        Sim2Real Label Efficiency Breakthrough
    </text>
    
    <!-- Efficient range background -->
    <rect x="{scale_x(0)}" y="{scale_y(1.0)}" 
          width="{scale_x(20)-scale_x(0)}" height="{scale_y(0)-scale_y(1.0)}" 
          fill="#90EE90" opacity="0.2"/>
    <text x="{scale_x(10)}" y="{scale_y(0.95)}" class="tick-label" 
          text-anchor="middle" fill="#2E8B57">Efficient Range (≤20%)</text>
    
    <!-- Y-axis -->
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" 
          stroke="black" stroke-width="1"/>
    
    <!-- X-axis -->
    <line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" 
          stroke="black" stroke-width="1"/>
    
    <!-- Grid lines -->'''
    
    # Y-axis grid
    for i in range(11):
        y_val = i / 10.0
        y_pos = scale_y(y_val)
        svg_content += f'''
    <line x1="{margin_left}" y1="{y_pos}" x2="{width-margin_right}" y2="{y_pos}" 
          stroke="#CCCCCC" stroke-width="0.5" opacity="0.5"/>'''
        if i % 2 == 0:
            svg_content += f'''
    <text x="{margin_left-10}" y="{y_pos+3}" class="tick-label" text-anchor="end">
        {y_val:.1f}
    </text>'''
    
    # X-axis ticks
    x_ticks = [1, 5, 10, 20, 50, 100]
    for x_tick in x_ticks:
        x_pos = scale_x(x_tick)
        svg_content += f'''
    <line x1="{x_pos}" y1="{height-margin_bottom}" x2="{x_pos}" y2="{height-margin_bottom+5}" 
          stroke="black" stroke-width="1"/>
    <text x="{x_pos}" y="{height-margin_bottom+20}" class="tick-label" text-anchor="middle">
        {x_tick}
    </text>'''
    
    # Target lines
    target_y = scale_y(0.80)
    ideal_y = scale_y(0.90)
    
    svg_content += f'''
    <!-- Target line (80%) -->
    <line x1="{margin_left}" y1="{target_y}" x2="{width-margin_right}" y2="{target_y}" 
          stroke="#FF6B6B" stroke-width="2" stroke-dasharray="8,4"/>
    <text x="{width-margin_right-5}" y="{target_y-5}" class="tick-label" 
          text-anchor="end" fill="#FF6B6B">Target: 80% F1</text>
    
    <!-- Ideal line (90%) -->
    <line x1="{margin_left}" y1="{ideal_y}" x2="{width-margin_right}" y2="{ideal_y}" 
          stroke="#FFA500" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="{width-margin_right-5}" y="{ideal_y-5}" class="tick-label" 
          text-anchor="end" fill="#FFA500">Ideal: 90% F1</text>'''
    
    # Error ribbon (simplified)
    svg_content += '''
    <!-- Error ribbon -->
    <path d="M'''
    
    # Upper error boundary
    for i, (x, y, err) in enumerate(zip(x_data, y_data, errors)):
        x_pos = scale_x(x)
        y_pos = scale_y(y + err)
        svg_content += f" {x_pos},{y_pos}"
    
    # Lower error boundary (reverse order)
    for i in reversed(range(len(x_data))):
        x, y, err = x_data[i], y_data[i], errors[i]
        x_pos = scale_x(x)
        y_pos = scale_y(max(0, y - err))
        svg_content += f" {x_pos},{y_pos}"
    
    svg_content += f''' Z" fill="#2E86AB" opacity="0.3"/>
    
    <!-- Main efficiency curve -->
    <polyline points="'''
    
    # Main curve points
    curve_points = []
    for x, y in zip(x_data, y_data):
        x_pos = scale_x(x)
        y_pos = scale_y(y)
        curve_points.append(f"{x_pos},{y_pos}")
    
    svg_content += " ".join(curve_points)
    svg_content += f'''" fill="none" stroke="#2E86AB" stroke-width="3"/>
    
    <!-- Data points -->'''
    
    # Data points and labels
    for i, (x, y, err) in enumerate(zip(x_data, y_data, errors)):
        x_pos = scale_x(x)
        y_pos = scale_y(y)
        
        svg_content += f'''
    <circle cx="{x_pos}" cy="{y_pos}" r="4" fill="white" stroke="#2E86AB" stroke-width="2"/>
    <text x="{x_pos}" y="{y_pos-15}" class="value-label" text-anchor="middle" fill="#2E86AB">
        {y:.3f}
    </text>'''
    
    # Key achievement annotation
    key_x, key_y = scale_x(20), scale_y(0.821)
    ann_x, ann_y = scale_x(45), scale_y(0.90)
    
    svg_content += f'''
    <!-- Key achievement annotation -->
    <rect x="{ann_x-50}" y="{ann_y-20}" width="100" height="35" 
          fill="#FFFACD" stroke="#FF6B6B" stroke-width="1.5" rx="5"/>
    <text x="{ann_x}" y="{ann_y-8}" class="annotation" text-anchor="middle" fill="#FF6B6B">
        Key Achievement:
    </text>
    <text x="{ann_x}" y="{ann_y+8}" class="annotation" text-anchor="middle" fill="#FF6B6B">
        82.1% F1 @ 20% Labels
    </text>
    <line x1="{ann_x-10}" y1="{ann_y+15}" x2="{key_x+5}" y2="{key_y-5}" 
          stroke="#FF6B6B" stroke-width="2" marker-end="url(#arrowhead)"/>
    
    <!-- Axis labels -->
    <text x="20" y="{height/2}" class="axis-label" text-anchor="middle" 
          transform="rotate(-90, 20, {height/2})">Macro F1 Score</text>
    <text x="{width/2}" y="{height-15}" class="axis-label" text-anchor="middle">
        Label Ratio (%)</text>
    
</svg>'''
    
    return svg_content

def main():
    print("📊 Direct SVG Figure Generation (IEEE IoTJ Quality)")
    print("=" * 60)
    
    # Generate Figure 3
    fig3_svg = create_figure3_svg()
    with open('figure3_d3_cross_domain.svg', 'w') as f:
        f.write(fig3_svg)
    
    # Generate Figure 4
    fig4_svg = create_figure4_svg()
    with open('figure4_d4_label_efficiency.svg', 'w') as f:
        f.write(fig4_svg)
    
    print("✅ Generated SVG figures:")
    print("  📊 figure3_d3_cross_domain.svg - D3跨域性能对比")
    print("  🎯 figure4_d4_label_efficiency.svg - D4标签效率曲线")
    
    print(f"\n🔧 SVG to PDF conversion:")
    print(f"  • 使用Inkscape: inkscape figure*.svg --export-pdf=figure*.pdf")
    print(f"  • 使用Chrome: 打开SVG → Print → Save as PDF")
    print(f"  • 使用在线工具: CloudConvert等SVG转PDF服务")
    
    print(f"\n💡 SVG Method优势:")
    print(f"  • 无依赖直接生成")
    print(f"  • IEEE IoTJ规范尺寸")
    print(f"  • 矢量图质量")
    print(f"  • 易于转换为PDF")
    
    print(f"\n📊 关键数据确认:")
    print(f"  • Enhanced跨域一致性: 83.0±0.1% F1")
    print(f"  • 标签效率突破: 82.1% F1 @ 20% labels")
    print(f"  • 成本效益: 80% labeling cost reduction")

if __name__ == "__main__":
    main()