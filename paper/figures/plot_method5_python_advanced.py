#!/usr/bin/env python3
"""
Method 5: Advanced Python Data Processing + Export
生成多种格式的图表数据，适配不同的专业绘图工具
"""

import csv
import json
import math

def create_latex_tikz_figure3():
    """Generate LaTeX TikZ code for Figure 3."""
    
    tikz_code = r"""
% Figure 3: D3 Cross-Domain Performance (LaTeX TikZ)
\begin{figure}[ht]
\centering
\begin{tikzpicture}[scale=0.8]
    % Define colors
    \definecolor{enhanced}{RGB}{46,134,171}
    \definecolor{cnn}{RGB}{232,72,85}
    \definecolor{bilstm}{RGB}{60,179,113}
    \definecolor{conformer}{RGB}{220,20,60}
    
    % Axes
    \draw[->] (0,0) -- (10,0) node[right] {Model Architecture};
    \draw[->] (0,0) -- (0,8) node[above] {Macro F1 Score};
    
    % Y-axis labels
    \foreach \y in {0,1,2,3,4,5,6,7,8}
        \draw[gray!30] (0,\y) -- (10,\y);
    \foreach \y in {0,2,4,6,8}
        \node[left] at (0,\y) {\pgfmathparse{\y/8}\pgfmathprintnumber{\pgfmathresult}};
    
    % LOSO bars (left group)
    \fill[enhanced] (1,0) rectangle (1.3,6.64);    % 0.830
    \fill[cnn] (2,0) rectangle (2.3,6.736);        % 0.842  
    \fill[bilstm] (3,0) rectangle (3.3,6.424);     % 0.803
    \fill[conformer] (4,0) rectangle (4.3,3.224);  % 0.403
    
    % LORO bars (right group)
    \fill[enhanced] (1.4,0) rectangle (1.7,6.64);  % 0.830
    \fill[cnn] (2.4,0) rectangle (2.7,6.368);      % 0.796
    \fill[bilstm] (3.4,0) rectangle (3.7,6.312);   % 0.789
    \fill[conformer] (4.4,0) rectangle (4.7,6.728); % 0.841
    
    % Error bars and labels
    % Enhanced (key highlight)
    \draw[thick] (1.15,6.64) -- (1.15,6.648) -- (1.55,6.648) -- (1.55,6.64);
    \node[above] at (1.35,6.7) {\textbf{83.0±0.1\%}};
    
    % Model labels
    \node[below] at (1.35,-0.3) {Enhanced};
    \node[below] at (2.35,-0.3) {CNN};
    \node[below] at (3.35,-0.3) {BiLSTM};
    \node[below] at (4.35,-0.3) {Conformer};
    
    % Legend
    \fill[enhanced] (6,7) rectangle (6.3,7.3) node[right] at (6.4,7.15) {LOSO};
    \fill[cnn] (6,6.5) rectangle (6.3,6.8) node[right] at (6.4,6.65) {LORO};
    
\end{tikzpicture}
\caption{Cross-domain generalization performance across LOSO and LORO protocols.}
\label{fig:cross_domain}
\end{figure}
"""
    
    return tikz_code

def create_svg_figure4():
    """Generate SVG code for Figure 4."""
    
    # D4 data
    x_data = [1.0, 5.0, 10.0, 20.0, 100.0]
    y_data = [0.455, 0.780, 0.730, 0.821, 0.833]
    errors = [0.050, 0.016, 0.104, 0.003, 0.000]
    
    # Scale for SVG (600x400 pixels, IEEE IoTJ aspect ratio)
    width, height = 600, 400
    margin = 60
    
    # Scale data to SVG coordinates
    x_min, x_max = 0, 105
    y_min, y_max = 0, 1.0
    
    def scale_x(x): return margin + (x - x_min) / (x_max - x_min) * (width - 2*margin)
    def scale_y(y): return height - margin - (y - y_min) / (y_max - y_min) * (height - 2*margin)
    
    svg_code = f'''
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <!-- IEEE IoTJ Figure 4: Label Efficiency Curve -->
    
    <!-- Background and grid -->
    <rect width="{width}" height="{height}" fill="white"/>
    
    <!-- Grid lines -->
    <!-- Y-axis grid -->
    '''
    
    for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
        svg_y = scale_y(y)
        svg_code += f'<line x1="{margin}" y1="{svg_y}" x2="{width-margin}" y2="{svg_y}" stroke="#CCCCCC" stroke-width="0.5"/>\n'
    
    # Efficient range background
    svg_code += f'''
    <!-- Efficient range background -->
    <rect x="{scale_x(0)}" y="{scale_y(1.0)}" width="{scale_x(20)-scale_x(0)}" height="{scale_y(0)-scale_y(1.0)}" 
          fill="#90EE90" opacity="0.2"/>
    
    <!-- Target line -->
    <line x1="{scale_x(0)}" y1="{scale_y(0.80)}" x2="{scale_x(100)}" y2="{scale_y(0.80)}" 
          stroke="#FF6B6B" stroke-width="2" stroke-dasharray="5,5"/>
    <text x="{scale_x(50)}" y="{scale_y(0.80)-5}" font-family="Times" font-size="12" 
          text-anchor="middle" fill="#FF6B6B">Target: 80% F1</text>
    
    <!-- Efficiency curve -->
    <polyline points="'''
    
    # Add curve points
    point_coords = []
    for i in range(len(x_data)):
        x_coord = scale_x(x_data[i])
        y_coord = scale_y(y_data[i])
        point_coords.append(f"{x_coord},{y_coord}")
    
    svg_code += " ".join(point_coords)
    svg_code += f'''" fill="none" stroke="#2E86AB" stroke-width="3"/>
    
    <!-- Data points -->
    '''
    
    # Add data points with labels
    for i in range(len(x_data)):
        x_coord = scale_x(x_data[i])
        y_coord = scale_y(y_data[i])
        
        svg_code += f'''<circle cx="{x_coord}" cy="{y_coord}" r="4" fill="#2E86AB" stroke="white" stroke-width="2"/>
    <text x="{x_coord}" y="{y_coord-10}" font-family="Times" font-size="10" 
          text-anchor="middle" fill="#2E86AB">{y_data[i]:.3f}</text>
    '''
    
    # Key achievement annotation
    key_x, key_y = scale_x(20), scale_y(0.821)
    ann_x, ann_y = scale_x(35), scale_y(0.90)
    
    svg_code += f'''
    <!-- Key achievement annotation -->
    <rect x="{ann_x-40}" y="{ann_y-15}" width="80" height="30" 
          fill="#FFFACD" stroke="#FF6B6B" stroke-width="1" rx="3"/>
    <text x="{ann_x}" y="{ann_y-5}" font-family="Times" font-size="10" font-weight="bold"
          text-anchor="middle" fill="#FF6B6B">Key Achievement:</text>
    <text x="{ann_x}" y="{ann_y+8}" font-family="Times" font-size="10" font-weight="bold"
          text-anchor="middle" fill="#FF6B6B">82.1% F1 @ 20% Labels</text>
    <line x1="{ann_x}" y1="{ann_y+15}" x2="{key_x}" y2="{key_y}" 
          stroke="#FF6B6B" stroke-width="2" marker-end="url(#arrowhead)"/>
    
    <!-- Arrow marker definition -->
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#FF6B6B"/>
        </marker>
    </defs>
    
    <!-- Axes -->
    <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" 
          stroke="black" stroke-width="1"/>
    <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" 
          stroke="black" stroke-width="1"/>
    
    <!-- Axis labels -->
    <text x="{width/2}" y="{height-20}" font-family="Times" font-size="12" 
          text-anchor="middle">Label Ratio (%)</text>
    <text x="20" y="{height/2}" font-family="Times" font-size="12" 
          text-anchor="middle" transform="rotate(-90, 20, {height/2})">Macro F1 Score</text>
    
    <!-- Title -->
    <text x="{width/2}" y="25" font-family="Times" font-size="14" font-weight="bold"
          text-anchor="middle">Sim2Real Label Efficiency Breakthrough</text>
    
</svg>
'''
    
    return svg_code

def generate_origin_pro_script():
    """Generate OriginPro script for publication-quality figures."""
    
    origin_script = """
// Method 4: OriginPro Publication Script
// 适用于IEEE期刊的最高质量图表制作

// ============ Figure 3: D3 Cross-Domain Performance ============

// Create new graph and import data
newbook;
impASC fname:="figure3_d3_cross_domain_data.csv";

// Create grouped column plot
plotxy (1,2:5) plot:=200;  // Grouped column plot

// Format for IEEE IoTJ
page.width = 17.1;     // cm
page.height = 10;      // cm
layer.x.from = 0;
layer.x.to = 5;
layer.y.from = 0;
layer.y.to = 1.0;

// Set colors (IEEE IoTJ palette)
set %c -c 1 color(46,134,171);    // Enhanced: Blue
set %c -c 2 color(232,72,85);     // CNN: Red-Orange  
set %c -c 3 color(60,179,113);    // BiLSTM: Green
set %c -c 4 color(220,20,60);     // Conformer: Crimson

// Add error bars
plotxy (1,2:5) plot:=181;  // Error bars

// Labels and formatting
xb.text$ = "Model Architecture";
yl.text$ = "Macro F1 Score";
layer.x.title.font = 1;           // Times font
layer.x.title.size = 10;
layer.y.title.font = 1;
layer.y.title.size = 10;

// Add value labels
label -a;  // Auto-label with values

// Export at 300 DPI
expGraph type:=pdf path:="figure3_d3_cross_domain_origin.pdf" resolution:=300;

// ============ Figure 4: D4 Label Efficiency ============

// New graph for Figure 4
newsheet;
impASC fname:="figure4_d4_label_efficiency_data.csv";

// Create line plot with error bars
plotxy (1,2) plot:=200;           // Line plot
plotxy (1,2:3) plot:=181;         // Error bars

// IEEE IoTJ formatting
page.width = 17.1;
page.height = 12;
layer.x.from = 0;
layer.x.to = 105;
layer.y.from = 0;
layer.y.to = 1.0;

// Line formatting
set %c -l w 3;                    // Line width 3pt
set %c -l c color(46,134,171);    // Enhanced blue
set %c -k 1;                      // Circle symbols
set %c -z 8;                      // Symbol size

// Add target line at 80%
draw -l -h {0,0.8} {105,0.8};
set %c -l c color(255,107,107);   // Red dashed
set %c -l d 2;                    // Dashed line

// Labels
xb.text$ = "Label Ratio (%)";
yl.text$ = "Macro F1 Score"; 
layer.title.text$ = "Sim2Real Label Efficiency Breakthrough";

// Key achievement annotation
label -sa -x 35 -y 0.87 "Key Achievement:\n82.1% F1 @ 20% Labels";

// Export
expGraph type:=pdf path:="figure4_d4_label_efficiency_origin.pdf" resolution:=300;

type "✓ OriginPro figures exported at IEEE IoTJ quality";
type "• Figure 3: Cross-domain comparison with precise error bars";
type "• Figure 4: Label efficiency with professional annotations";
type "• Resolution: 300 DPI, Times font, IEEE compliant sizing";
"""
    
    return origin_script

def create_excel_plotting_data():
    """Generate Excel/LibreOffice compatible plotting data."""
    
    # Figure 3 data for Excel
    excel_fig3_data = [
        ["Model", "LOSO_F1", "LOSO_Error", "LORO_F1", "LORO_Error", "Color_Code"],
        ["Enhanced", 0.830, 0.001, 0.830, 0.001, "#2E86AB"],
        ["CNN", 0.842, 0.025, 0.796, 0.097, "#E84855"],
        ["BiLSTM", 0.803, 0.022, 0.789, 0.044, "#3CB371"],
        ["Conformer-lite", 0.403, 0.386, 0.841, 0.040, "#DC143C"]
    ]
    
    # Figure 4 data for Excel
    excel_fig4_data = [
        ["Label_Percent", "F1_Score", "Error_Bar", "Performance_Level"],
        [1.0, 0.455, 0.050, "Progress"],
        [5.0, 0.780, 0.016, "Good"],
        [10.0, 0.730, 0.104, "Good"],
        [20.0, 0.821, 0.003, "Target Achieved"],
        [100.0, 0.833, 0.000, "Target Achieved"]
    ]
    
    return excel_fig3_data, excel_fig4_data

def detailed_data_analysis():
    """Provide detailed statistical analysis for the figures."""
    
    print("📊 Advanced Python Data Analysis (Method 5)")
    print("=" * 60)
    
    # Figure 3 analysis
    print("\n🎯 Figure 3: Cross-Domain Analysis")
    print("-" * 40)
    
    # Calculate enhanced model advantage
    enhanced_loso = 0.830
    enhanced_loro = 0.830
    consistency = abs(enhanced_loso - enhanced_loro)
    
    print(f"Enhanced Model Cross-Domain Consistency:")
    print(f"  LOSO: {enhanced_loso:.3f} ± 0.001")
    print(f"  LORO: {enhanced_loro:.3f} ± 0.001") 
    print(f"  Difference: {consistency:.3f} (Perfect consistency!)")
    print(f"  Coefficient of Variation: <0.2% (Exceptional stability)")
    
    # Compare with baselines
    baselines = {
        'CNN': {'loso': (0.842, 0.025), 'loro': (0.796, 0.097)},
        'BiLSTM': {'loso': (0.803, 0.022), 'loro': (0.789, 0.044)},
        'Conformer': {'loso': (0.403, 0.386), 'loro': (0.841, 0.040)}
    }
    
    print(f"\nBaseline Comparison:")
    for model, data in baselines.items():
        loso_mean, loso_std = data['loso']
        loro_mean, loro_std = data['loro']
        gap = abs(loso_mean - loro_mean)
        avg_cv = ((loso_std/loso_mean) + (loro_std/loro_mean)) / 2 * 100
        
        print(f"  {model:12}: Gap={gap:.3f}, Avg CV={avg_cv:.1f}%")
    
    # Figure 4 analysis
    print(f"\n🎯 Figure 4: Label Efficiency Analysis")
    print("-" * 40)
    
    x_data = [1.0, 5.0, 10.0, 20.0, 100.0]
    y_data = [0.455, 0.780, 0.730, 0.821, 0.833]
    
    # Calculate efficiency metrics
    target_performance = 0.821  # @ 20% labels
    full_performance = 0.833    # @ 100% labels
    efficiency_gap = full_performance - target_performance
    efficiency_retention = target_performance / full_performance
    
    print(f"Label Efficiency Breakthrough:")
    print(f"  Target (20% labels): {target_performance:.3f} F1")
    print(f"  Full supervision: {full_performance:.3f} F1")
    print(f"  Performance gap: {efficiency_gap:.3f} ({efficiency_gap/full_performance*100:.1f}%)")
    print(f"  Efficiency retention: {efficiency_retention:.1%}")
    print(f"  Cost reduction: 80% (5x less labeling required)")
    
    # Calculate improvement rates
    print(f"\nEfficiency Curve Analysis:")
    for i in range(len(x_data)-1):
        x1, y1 = x_data[i], y_data[i]
        x2, y2 = x_data[i+1], y_data[i+1]
        
        label_increase = x2 - x1
        perf_increase = y2 - y1
        efficiency = perf_increase / label_increase if label_increase > 0 else 0
        
        print(f"  {x1}% → {x2}%: Δ{perf_increase:+.3f} F1 (efficiency: {efficiency:.4f} F1/% label)")
    
    return target_performance, efficiency_retention

def main():
    print("🎨 Multi-Tool Figure Generation for PaperA")
    print("Generating 4 different plotting approaches for comparison")
    
    # Method 5: Advanced data processing
    detailed_data_analysis()
    
    # Generate plotting scripts
    tikz_code = create_latex_tikz_figure3()
    with open('figure3_latex_tikz.tex', 'w') as f:
        f.write(tikz_code)
    
    svg_code = create_svg_figure4()
    with open('figure4_web_svg.svg', 'w') as f:
        f.write(svg_code)
    
    origin_script = generate_origin_pro_script()
    with open('figures_origin_pro.ogs', 'w') as f:
        f.write(origin_script)
    
    excel_fig3, excel_fig4 = create_excel_plotting_data()
    
    # Save Excel data
    with open('figure3_excel_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(excel_fig3)
    
    with open('figure4_excel_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(excel_fig4)
    
    print(f"\n✅ Generated plotting files:")
    print(f"  📊 figure3_latex_tikz.tex - LaTeX TikZ图表")
    print(f"  🌐 figure4_web_svg.svg - SVG矢量图")
    print(f"  🔬 figures_origin_pro.ogs - OriginPro脚本")
    print(f"  📈 figure*_excel_data.csv - Excel绘图数据")
    
    print(f"\n💡 Method 5 Advantages:")
    print(f"  • 多格式输出，适配不同工具")
    print(f"  • 详细数据分析和统计")
    print(f"  • IEEE IoTJ标准规范")
    print(f"  • 专业图表代码生成")

if __name__ == "__main__":
    main()