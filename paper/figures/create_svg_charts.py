#!/usr/bin/env python3
"""
Simple SVG Chart Generation (No dependencies)
直接生成可用的SVG图表，可转换为PDF
"""

def create_figure3_svg():
    """Generate Figure 3: D3 Cross-Domain Performance."""
    
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="350" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font-family: Times, serif; font-size: 14px; font-weight: bold; }
        .axis { font-family: Times, serif; font-size: 12px; }
        .label { font-family: Times, serif; font-size: 10px; }
        .value { font-family: Times, serif; font-size: 8px; }
    </style>
    
    <!-- Background -->
    <rect width="600" height="350" fill="white"/>
    
    <!-- Title -->
    <text x="300" y="25" class="title" text-anchor="middle">Cross-Domain Generalization Performance</text>
    
    <!-- Axes -->
    <line x1="80" y1="50" x2="80" y2="280" stroke="black" stroke-width="1"/>
    <line x1="80" y1="280" x2="550" y2="280" stroke="black" stroke-width="1"/>
    
    <!-- Y-axis labels -->
    <text x="70" y="285" class="label" text-anchor="end">0.0</text>
    <text x="70" y="233" class="label" text-anchor="end">0.2</text>
    <text x="70" y="181" class="label" text-anchor="end">0.4</text>
    <text x="70" y="129" class="label" text-anchor="end">0.6</text>
    <text x="70" y="77" class="label" text-anchor="end">0.8</text>
    <text x="70" y="55" class="label" text-anchor="end">1.0</text>
    
    <!-- Grid -->
    <line x1="80" y1="280" x2="550" y2="280" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="232" x2="550" y2="232" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="184" x2="550" y2="184" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="136" x2="550" y2="136" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="88" x2="550" y2="88" stroke="#CCCCCC" stroke-width="0.5"/>
    
    <!-- Enhanced LOSO bar (83.0%) -->
    <rect x="100" y="90" width="25" height="190" fill="#2E86AB" opacity="0.8" stroke="black" stroke-width="0.5"/>
    <!-- Enhanced LORO bar (83.0%) -->
    <rect x="130" y="90" width="25" height="190" fill="#E84855" opacity="0.8" stroke="black" stroke-width="0.5"/>
    
    <!-- CNN bars -->
    <rect x="180" y="84" width="25" height="196" fill="#2E86AB" opacity="0.8" stroke="black" stroke-width="0.5"/>
    <rect x="210" y="98" width="25" height="182" fill="#E84855" opacity="0.8" stroke="black" stroke-width="0.5"/>
    
    <!-- BiLSTM bars -->  
    <rect x="260" y="95" width="25" height="185" fill="#2E86AB" opacity="0.8" stroke="black" stroke-width="0.5"/>
    <rect x="290" y="99" width="25" height="181" fill="#E84855" opacity="0.8" stroke="black" stroke-width="0.5"/>
    
    <!-- Conformer bars -->
    <rect x="340" y="188" width="25" height="92" fill="#2E86AB" opacity="0.8" stroke="black" stroke-width="0.5"/>
    <rect x="370" y="86" width="25" height="194" fill="#E84855" opacity="0.8" stroke="black" stroke-width="0.5"/>
    
    <!-- Model labels -->
    <text x="127" y="305" class="label" text-anchor="middle">Enhanced</text>
    <text x="207" y="305" class="label" text-anchor="middle">CNN</text>
    <text x="287" y="305" class="label" text-anchor="middle">BiLSTM</text>
    <text x="367" y="305" class="label" text-anchor="middle">Conformer-lite</text>
    
    <!-- Value labels -->
    <text x="127" y="75" class="value" text-anchor="middle" fill="#2E86AB" font-weight="bold">83.0±0.1%</text>
    <text x="207" y="70" class="value" text-anchor="middle">84.2±2.5%</text>
    <text x="287" y="85" class="value" text-anchor="middle">80.3±2.2%</text>
    
    <!-- Legend -->
    <rect x="450" y="60" width="15" height="15" fill="#2E86AB" opacity="0.8"/>
    <text x="470" y="72" class="label">LOSO</text>
    <rect x="450" y="80" width="15" height="15" fill="#E84855" opacity="0.8"/>
    <text x="470" y="92" class="label">LORO</text>
    
    <!-- Highlight Enhanced consistency -->
    <rect x="95" y="85" width="65" height="200" fill="none" stroke="#FFD700" stroke-width="2" stroke-dasharray="5,5"/>
    <text x="127" y="45" class="value" text-anchor="middle" fill="#FFD700" font-weight="bold">⭐ Perfect Consistency</text>
    
    <!-- Axis titles -->
    <text x="315" y="335" class="axis" text-anchor="middle">Model Architecture</text>
    <text x="25" y="165" class="axis" text-anchor="middle" transform="rotate(-90, 25, 165)">Macro F1 Score</text>
    
</svg>'''
    
    return svg

def create_figure4_svg():
    """Generate Figure 4: D4 Label Efficiency."""
    
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="700" height="450" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#FF6B6B"/>
        </marker>
    </defs>
    
    <style>
        .title { font-family: Times, serif; font-size: 14px; font-weight: bold; }
        .axis { font-family: Times, serif; font-size: 12px; }
        .label { font-family: Times, serif; font-size: 10px; }
        .value { font-family: Times, serif; font-size: 9px; }
        .annotation { font-family: Times, serif; font-size: 11px; font-weight: bold; }
    </style>
    
    <!-- Background -->
    <rect width="700" height="450" fill="white"/>
    
    <!-- Title -->
    <text x="350" y="25" class="title" text-anchor="middle">Sim2Real Label Efficiency Breakthrough</text>
    
    <!-- Efficient range background -->
    <rect x="80" y="50" width="120" height="300" fill="#90EE90" opacity="0.2"/>
    <text x="140" y="70" class="label" text-anchor="middle" fill="#2E8B57">Efficient Range (≤20%)</text>
    
    <!-- Axes -->
    <line x1="80" y1="50" x2="80" y2="350" stroke="black" stroke-width="1"/>
    <line x1="80" y1="350" x2="650" y2="350" stroke="black" stroke-width="1"/>
    
    <!-- Grid lines -->
    <line x1="80" y1="350" x2="650" y2="350" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="290" x2="650" y2="290" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="230" x2="650" y2="230" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="170" x2="650" y2="170" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="110" x2="650" y2="110" stroke="#CCCCCC" stroke-width="0.5"/>
    <line x1="80" y1="50" x2="650" y2="50" stroke="#CCCCCC" stroke-width="0.5"/>
    
    <!-- Y-axis labels -->
    <text x="70" y="355" class="label" text-anchor="end">0.0</text>
    <text x="70" y="295" class="label" text-anchor="end">0.2</text>
    <text x="70" y="235" class="label" text-anchor="end">0.4</text>
    <text x="70" y="175" class="label" text-anchor="end">0.6</text>
    <text x="70" y="115" class="label" text-anchor="end">0.8</text>
    <text x="70" y="55" class="label" text-anchor="end">1.0</text>
    
    <!-- X-axis labels -->
    <text x="90" y="370" class="label" text-anchor="middle">1</text>
    <text x="130" y="370" class="label" text-anchor="middle">5</text>
    <text x="170" y="370" class="label" text-anchor="middle">10</text>
    <text x="200" y="370" class="label" text-anchor="middle">20</text>
    <text x="350" y="370" class="label" text-anchor="middle">50</text>
    <text x="620" y="370" class="label" text-anchor="middle">100</text>
    
    <!-- Target line (80%) -->
    <line x1="80" y1="110" x2="650" y2="110" stroke="#FF6B6B" stroke-width="2" stroke-dasharray="8,4"/>
    <text x="640" y="105" class="label" text-anchor="end" fill="#FF6B6B">Target: 80% F1</text>
    
    <!-- Efficiency curve points -->
    <!-- 1% -> 45.5% -->
    <circle cx="90" y="213" r="4" fill="white" stroke="#2E86AB" stroke-width="2"/>
    <!-- 5% -> 78.0% -->  
    <circle cx="130" y="116" r="4" fill="white" stroke="#2E86AB" stroke-width="2"/>
    <!-- 10% -> 73.0% -->
    <circle cx="170" y="131" r="4" fill="white" stroke="#2E86AB" stroke-width="2"/>
    <!-- 20% -> 82.1% -->
    <circle cx="200" y="104" r="5" fill="#FFD700" stroke="#2E86AB" stroke-width="3"/>
    <!-- 100% -> 83.3% -->
    <circle cx="620" y="100" r="4" fill="white" stroke="#2E86AB" stroke-width="2"/>
    
    <!-- Efficiency curve line -->
    <polyline points="90,213 130,116 170,131 200,104 620,100" 
              fill="none" stroke="#2E86AB" stroke-width="3"/>
    
    <!-- Error bars (simplified for key points) -->
    <!-- 20% point error bar -->
    <line x1="200" y1="103" x2="200" y2="105" stroke="black" stroke-width="1"/>
    <line x1="197" y1="103" x2="203" y2="103" stroke="black" stroke-width="1"/>
    <line x1="197" y1="105" x2="203" y2="105" stroke="black" stroke-width="1"/>
    
    <!-- Value labels -->
    <text x="90" y="200" class="value" text-anchor="middle" fill="#2E86AB">45.5%</text>
    <text x="130" y="105" class="value" text-anchor="middle" fill="#2E86AB">78.0%</text>
    <text x="170" y="120" class="value" text-anchor="middle" fill="#2E86AB">73.0%</text>
    <text x="200" y="90" class="value" text-anchor="middle" fill="#FFD700" font-weight="bold">82.1%</text>
    <text x="620" y="90" class="value" text-anchor="middle" fill="#2E86AB">83.3%</text>
    
    <!-- Key achievement annotation -->
    <rect x="280" y="60" width="140" height="35" fill="#FFFACD" stroke="#FF6B6B" stroke-width="1.5" rx="5"/>
    <text x="350" y="75" class="annotation" text-anchor="middle" fill="#FF6B6B">Key Achievement:</text>
    <text x="350" y="88" class="annotation" text-anchor="middle" fill="#FF6B6B">82.1% F1 @ 20% Labels</text>
    <line x1="320" y1="95" x2="205" y2="110" stroke="#FF6B6B" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- Axis titles -->
    <text x="365" y="410" class="axis" text-anchor="middle">Label Ratio (%)</text>
    <text x="25" y="200" class="axis" text-anchor="middle" transform="rotate(-90, 25, 200)">Macro F1 Score</text>
    
    <!-- Cost reduction highlight -->
    <text x="350" y="320" class="value" text-anchor="middle" fill="#2E8B57" font-weight="bold">
        80% Cost Reduction: 20% labels → 82.1% F1
    </text>
    
</svg>'''
    
    return svg

def create_figure4_svg():
    """Generate Figure 4: D4 Label Efficiency (Alternative simple version)."""
    
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="350" xmlns="http://www.w3.org/2000/svg">
    <style>
        .title { font-family: Times, serif; font-size: 14px; font-weight: bold; }
        .label { font-family: Times, serif; font-size: 10px; }
        .value { font-family: Times, serif; font-size: 9px; }
    </style>
    
    <!-- Background -->
    <rect width="600" height="350" fill="white"/>
    
    <!-- Title -->
    <text x="300" y="25" class="title" text-anchor="middle">D4 Label Efficiency Results Summary</text>
    
    <!-- Table-style layout for efficiency data -->
    <rect x="50" y="50" width="500" height="250" fill="none" stroke="black" stroke-width="1"/>
    
    <!-- Header -->
    <rect x="50" y="50" width="500" height="30" fill="#E8E8E8" stroke="black" stroke-width="0.5"/>
    <text x="100" y="70" class="label" text-anchor="middle" font-weight="bold">Label %</text>
    <text x="200" y="70" class="label" text-anchor="middle" font-weight="bold">F1 Score</text>
    <text x="300" y="70" class="label" text-anchor="middle" font-weight="bold">Performance</text>
    <text x="450" y="70" class="label" text-anchor="middle" font-weight="bold">Status</text>
    
    <!-- Data rows -->
    <line x1="50" y1="80" x2="550" y2="80" stroke="black" stroke-width="0.5"/>
    <text x="100" y="100" class="value" text-anchor="middle">1%</text>
    <text x="200" y="100" class="value" text-anchor="middle">45.5%</text>
    <text x="300" y="100" class="value" text-anchor="middle">Bootstrap</text>
    <text x="450" y="100" class="value" text-anchor="middle">📊 Progress</text>
    
    <line x1="50" y1="110" x2="550" y2="110" stroke="black" stroke-width="0.5"/>
    <text x="100" y="130" class="value" text-anchor="middle">5%</text>
    <text x="200" y="130" class="value" text-anchor="middle">78.0%</text>
    <text x="300" y="130" class="value" text-anchor="middle">Good</text>
    <text x="450" y="130" class="value" text-anchor="middle">📈 Approaching</text>
    
    <line x1="50" y1="140" x2="550" y2="140" stroke="black" stroke-width="0.5"/>
    <text x="100" y="160" class="value" text-anchor="middle">10%</text>
    <text x="200" y="160" class="value" text-anchor="middle">73.0%</text>
    <text x="300" y="160" class="value" text-anchor="middle">Moderate</text>
    <text x="450" y="160" class="value" text-anchor="middle">📊 Variable</text>
    
    <!-- KEY ROW - 20% labels -->
    <rect x="50" y="170" width="500" height="30" fill="#FFFACD" stroke="#FF6B6B" stroke-width="2"/>
    <text x="100" y="190" class="value" text-anchor="middle" font-weight="bold">20%</text>
    <text x="200" y="190" class="value" text-anchor="middle" font-weight="bold" fill="#FF6B6B">82.1%</text>
    <text x="300" y="190" class="value" text-anchor="middle" font-weight="bold">Excellent</text>
    <text x="450" y="190" class="value" text-anchor="middle" font-weight="bold">🎯 TARGET</text>
    
    <line x1="50" y1="200" x2="550" y2="200" stroke="black" stroke-width="0.5"/>
    <text x="100" y="220" class="value" text-anchor="middle">100%</text>
    <text x="200" y="220" class="value" text-anchor="middle">83.3%</text>
    <text x="300" y="220" class="value" text-anchor="middle">Ceiling</text>
    <text x="450" y="220" class="value" text-anchor="middle">📊 Reference</text>
    
    <!-- Key achievement callout -->
    <text x="300" y="330" class="annotation" text-anchor="middle" fill="#FF6B6B">
        🏆 BREAKTHROUGH: 82.1% F1 @ 20% Labels (80% Cost Reduction)
    </text>
    
</svg>'''
    
    return svg

def main():
    print("📊 Direct SVG Generation (IEEE IoTJ Compatible)")
    print("=" * 55)
    
    # Generate Figure 3
    fig3_svg = create_figure3_svg()
    with open('figure3_d3_cross_domain_direct.svg', 'w') as f:
        f.write(fig3_svg)
    
    # Generate Figure 4
    fig4_svg = create_figure4_svg()
    with open('figure4_d4_efficiency_table.svg', 'w') as f:
        f.write(fig4_svg)
    
    print("✅ Generated direct SVG figures:")
    print("  📊 figure3_d3_cross_domain_direct.svg")
    print("  📈 figure4_d4_efficiency_table.svg")
    
    print(f"\n🔄 SVG → PDF Conversion Options:")
    print(f"  1. Inkscape: inkscape *.svg --export-pdf=*.pdf")
    print(f"  2. Chrome: Open SVG → Print → Save as PDF")
    print(f"  3. Online: CloudConvert.com SVG to PDF")
    print(f"  4. LibreOffice Draw: Import SVG → Export PDF")
    
    print(f"\n📊 Figure Content:")
    print(f"  • Figure 3: Enhanced 83.0±0.1% consistency highlight")
    print(f"  • Figure 4: 82.1% @ 20% labels breakthrough table")
    print(f"  • IEEE IoTJ compliance: Proper sizing and fonts")
    print(f"  • Key achievements prominently displayed")

if __name__ == "__main__":
    main()