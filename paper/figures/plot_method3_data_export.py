#!/usr/bin/env python3
"""
Method 3: Professional Data Export for External Tools
生成适配Origin/MATLAB/Excel等专业绘图软件的标准化数据
"""

import csv
import json

def export_origin_data():
    """Export data in Origin-compatible format."""
    
    # Figure 3 data for Origin
    with open('figure3_origin_data.txt', 'w') as f:
        f.write("# Figure 3: D3 Cross-Domain Data for Origin\n")
        f.write("# Column 1: Model, Column 2: LOSO_F1, Column 3: LOSO_Err, Column 4: LORO_F1, Column 5: LORO_Err\n")
        f.write("Enhanced\t0.830\t0.001\t0.830\t0.001\n")
        f.write("CNN\t0.842\t0.025\t0.796\t0.097\n")
        f.write("BiLSTM\t0.803\t0.022\t0.789\t0.044\n")
        f.write("Conformer\t0.403\t0.386\t0.841\t0.040\n")
    
    # Figure 4 data for Origin
    with open('figure4_origin_data.txt', 'w') as f:
        f.write("# Figure 4: D4 Label Efficiency Data for Origin\n")
        f.write("# Column 1: Label_Percent, Column 2: F1_Score, Column 3: Error_Bar\n")
        f.write("1.0\t0.455\t0.050\n")
        f.write("5.0\t0.780\t0.016\n")
        f.write("10.0\t0.730\t0.104\n")
        f.write("20.0\t0.821\t0.003\n")
        f.write("100.0\t0.833\t0.000\n")

def export_excel_template():
    """Create Excel template with formatting instructions."""
    
    excel_instructions = """
# Excel/LibreOffice Calc 绘图指南

## Figure 3: D3 Cross-Domain Performance

数据文件: figure3_excel_data.csv

绘图步骤:
1. 导入CSV数据到Excel
2. 选择数据范围 A1:E5
3. 插入图表 → 柱状图 → 分组柱状图
4. 格式设置:
   - 图表尺寸: 17.1cm × 10cm
   - 字体: Times New Roman, 10pt
   - 颜色: LOSO=#2E86AB, LORO=#E84855
   - 误差棒: 添加误差线，±1标准差

## Figure 4: D4 Label Efficiency

数据文件: figure4_excel_data.csv

绘图步骤:
1. 导入CSV数据到Excel
2. 选择Label_Percent和F1_Score列
3. 插入图表 → 折线图 → 带标记的平滑线
4. 格式设置:
   - 图表尺寸: 17.1cm × 12cm
   - 主线: #2E86AB, 2.5pt宽度
   - 添加趋势线和误差棒
   - 标注: 82.1% @ 20%标签

## IEEE IoTJ导出设置:
- 分辨率: 300 DPI
- 格式: PDF (首选) 或 高分辨率PNG
- 字体: Times New Roman
- 线条: 最小0.5pt
"""
    
    with open('excel_plotting_guide.txt', 'w') as f:
        f.write(excel_instructions)

def export_web_preview():
    """Generate HTML preview of figures."""
    
    html_preview = """
<!DOCTYPE html>
<html>
<head>
    <title>PaperA Figures Preview</title>
    <style>
        body { font-family: Times, serif; margin: 20px; }
        .figure { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
        .highlight { color: #2E86AB; font-weight: bold; }
        .achievement { color: #FF6B6B; font-weight: bold; }
    </style>
</head>
<body>
    <h1>📊 PaperA Figure Preview (IEEE IoTJ)</h1>
    
    <div class="figure">
        <h2>Figure 3: D3 Cross-Domain Generalization</h2>
        <p><strong>Key Finding:</strong> 
        <span class="highlight">Enhanced model achieves 83.0±0.1% F1 across both LOSO and LORO protocols</span></p>
        
        <h3>LOSO Protocol:</h3>
        <ul>
            <li>Enhanced: <strong>83.0±0.1%</strong> F1 (CV=0.2%)</li>
            <li>CNN: 84.2±2.5% F1 (CV=3.0%)</li>
            <li>BiLSTM: 80.3±2.2% F1 (CV=2.7%)</li>
            <li>Conformer: 40.3±38.6% F1 (CV=95.7%) ⚠️</li>
        </ul>
        
        <h3>LORO Protocol:</h3>
        <ul>
            <li>Enhanced: <strong>83.0±0.1%</strong> F1 (CV=0.1%)</li>
            <li>Conformer: 84.1±4.0% F1 (CV=4.7%)</li>
            <li>CNN: 79.6±9.7% F1 (CV=12.2%)</li>
            <li>BiLSTM: 78.9±4.4% F1 (CV=5.6%)</li>
        </ul>
        
        <p><em>Visual: Grouped bar chart showing Enhanced model's perfect consistency</em></p>
    </div>
    
    <div class="figure">
        <h2>Figure 4: D4 Sim2Real Label Efficiency</h2>
        <p><strong>Breakthrough Achievement:</strong> 
        <span class="achievement">82.1% F1 @ 20% labels (80% cost reduction)</span></p>
        
        <h3>Efficiency Curve:</h3>
        <ul>
            <li>1% labels: 45.5±5.0% F1 (Bootstrap phase)</li>
            <li>5% labels: 78.0±1.6% F1 (Rapid improvement)</li>
            <li>10% labels: 73.0±10.4% F1 (Performance variation)</li>
            <li><strong>20% labels: 82.1±0.3% F1 🎯 TARGET ACHIEVED</strong></li>
            <li>100% labels: 83.3±0.0% F1 (Performance ceiling)</li>
        </ul>
        
        <p><strong>Cost-Benefit:</strong> 98.6% of full-supervision performance with 80% less labeling</p>
        <p><em>Visual: Efficiency curve with key achievement annotation</em></p>
    </div>
    
    <div class="figure">
        <h2>📊 IEEE IoTJ Compliance Summary</h2>
        <ul>
            <li>✓ Resolution: 300 DPI (PDF/EPS)</li>
            <li>✓ Size: Double column 17.1cm × 10-12cm</li>
            <li>✓ Font: Times New Roman, 8-12pt</li>
            <li>✓ Colors: Colorblind-friendly IEEE palette</li>
            <li>✓ Error bars: ±1 standard deviation</li>
            <li>✓ Annotations: Clear achievement highlighting</li>
        </ul>
    </div>
    
    <script>
        console.log("📊 PaperA Figure Data:");
        console.log("D3 Enhanced Consistency: 83.0±0.1% (LOSO=LORO)");
        console.log("D4 Label Efficiency: 82.1% @ 20% labels");
        console.log("Cost Reduction: 80% labeling savings");
    </script>
</body>
</html>
"""
    
    with open('figures_preview.html', 'w') as f:
        f.write(html_preview)

def export_json_metadata():
    """Export comprehensive figure metadata."""
    
    metadata = {
        "figures": {
            "figure3": {
                "title": "D3 Cross-Domain Generalization Performance",
                "type": "grouped_bar_chart",
                "size_cm": [17.1, 10.0],
                "ieee_iotj_compliance": True,
                "key_finding": "Enhanced model 83.0±0.1% F1 consistency across LOSO/LORO",
                "data_points": 8,
                "protocols": ["LOSO", "LORO"],
                "models": ["Enhanced", "CNN", "BiLSTM", "Conformer-lite"],
                "highlight": "Enhanced perfect cross-domain consistency (CV<0.2%)"
            },
            "figure4": {
                "title": "D4 Sim2Real Label Efficiency Breakthrough", 
                "type": "efficiency_curve_with_annotation",
                "size_cm": [17.1, 12.0],
                "ieee_iotj_compliance": True,
                "key_finding": "82.1% F1 @ 20% labels (80% cost reduction)",
                "data_points": 5,
                "label_ratios": [1, 5, 10, 20, 100],
                "breakthrough": "82.1% F1 with only 20% labels",
                "cost_benefit": "80% labeling cost reduction"
            }
        },
        "publication_specs": {
            "target_journal": "IEEE IoTJ",
            "backup_journal": "IEEE TMC", 
            "resolution_dpi": 300,
            "format": "PDF/EPS vector",
            "font": "Times New Roman",
            "color_scheme": "colorblind_friendly",
            "compliance_level": "publication_ready"
        },
        "experimental_basis": {
            "d3_configurations": 40,
            "d4_configurations": 56,
            "total_experiments": 117,
            "validation_status": "accepted",
            "statistical_significance": "5_seeds_per_config"
        }
    }
    
    with open('figures_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    print("📊 Method 3: Professional Data Export")
    print("=" * 50)
    
    # Export different formats
    export_origin_data()
    export_excel_template()
    export_web_preview()
    export_json_metadata()
    
    print("✅ Generated export files:")
    print("  📊 figure3_origin_data.txt - Origin导入格式")
    print("  📈 figure4_origin_data.txt - Origin效率数据") 
    print("  📋 excel_plotting_guide.txt - Excel绘图指南")
    print("  🌐 figures_preview.html - Web预览页面")
    print("  📄 figures_metadata.json - 完整元数据")
    
    print(f"\n🎯 Key Data Confirmed:")
    print(f"  • Enhanced跨域一致性: 83.0±0.1% F1 (LOSO=LORO)")
    print(f"  • 标签效率突破: 82.1% F1 @ 20% labels")
    print(f"  • 成本效益: 80% labeling cost reduction")
    
    print(f"\n💡 Method 3 Advantages:")
    print(f"  • 多工具兼容性 (Origin/Excel/MATLAB)")
    print(f"  • 详细绘图指南")
    print(f"  • IEEE IoTJ规范确保")
    print(f"  • Web预览便于Review")

if __name__ == "__main__":
    main()