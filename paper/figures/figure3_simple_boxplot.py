#!/usr/bin/env python3
"""
Figure 3: Enhanced Cross-Domain Performance Boxplot
Simple, reliable, and visually appealing replacement for violin plot
IEEE IoTJ Paper - WiFi CSI HAR
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure for IEEE IoTJ standards
plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_enhanced_boxplot():
    """Create enhanced cross-domain performance analysis"""
    
    # Data from paper results
    data = {
        'Model': ['Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite'],
        'LOSO_F1': [0.830, 0.842, 0.803, 0.403],
        'LOSO_Std': [0.001, 0.025, 0.022, 0.386],
        'LORO_F1': [0.830, 0.796, 0.789, 0.841],
        'LORO_Std': [0.001, 0.097, 0.044, 0.040]
    }
    df = pd.DataFrame(data)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # === Main Performance Comparison ===
    x = np.arange(len(df['Model']))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['LOSO_F1'], width, yerr=df['LOSO_Std'], 
                   label='LOSO', color='#2ECC71', alpha=0.8, capsize=5, capthick=2)
    bars2 = ax1.bar(x + width/2, df['LORO_F1'], width, yerr=df['LORO_Std'],
                   label='LORO', color='#3498DB', alpha=0.8, capsize=5, capthick=2)
    
    # Enhanced model golden highlighting
    bars1[0].set_edgecolor('gold')
    bars1[0].set_linewidth(4)
    bars2[0].set_edgecolor('gold') 
    bars2[0].set_linewidth(4)
    
    # Add performance values on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + df['LOSO_Std'].iloc[i] + 0.01,
                f'{height1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + df['LORO_Std'].iloc[i] + 0.01,
                f'{height2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_title('Cross-Domain Performance Comparison\n(Enhanced Model in Gold)', 
                 fontweight='bold', fontsize=12)
    ax1.set_ylabel('Macro F1 Score', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], fontweight='bold')
    ax1.legend(fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add consistency annotation for Enhanced
    ax1.annotate('Perfect Consistency\n83.0±0.1% (LOSO = LORO)', 
                xy=(0, 0.83), xytext=(1, 0.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                fontsize=10, ha='center', fontweight='bold')
    
    # === Cross-Domain Gap Analysis ===
    gaps = abs(df['LOSO_F1'] - df['LORO_F1'])
    bars_gap = ax2.bar(range(len(df)), gaps, 
                      color=['gold', 'blue', 'orange', 'red'], alpha=0.8, edgecolor='black')
    
    # Enhanced model special highlighting
    bars_gap[0].set_edgecolor('darkgoldenrod')
    bars_gap[0].set_linewidth(4)
    
    ax2.set_title('Cross-Domain Gap Analysis\n|LOSO - LORO| (Lower = Better)', fontweight='bold')
    ax2.set_ylabel('Performance Gap', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['Model'], fontweight='bold', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add gap values on bars
    for bar, gap in zip(bars_gap, gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # === Consistency Score Analysis ===
    # Calculate consistency based on CV (lower CV = higher consistency)
    cv_scores = [0.12, 2.97, 2.74, 95.79]  # From paper
    consistency_scores = [1/(1 + cv/100) for cv in cv_scores]  # Convert to 0-1 scale
    
    bars_cons = ax3.bar(range(len(df)), consistency_scores, 
                       color=['gold', 'blue', 'orange', 'red'], alpha=0.8, edgecolor='black')
    
    # Enhanced model highlighting
    bars_cons[0].set_edgecolor('darkgoldenrod')
    bars_cons[0].set_linewidth(4)
    
    ax3.set_title('Model Consistency Score\n(Based on CV, Higher = Better)', fontweight='bold')
    ax3.set_ylabel('Consistency Score', fontweight='bold')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Model'], fontweight='bold', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add consistency values
    for bar, score, cv in zip(bars_cons, consistency_scores, cv_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{score:.3f}\n(CV={cv:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=8)
    
    # === Statistical Significance Analysis ===
    comparisons = ['Enhanced vs CNN', 'Enhanced vs BiLSTM', 'Enhanced vs Conformer']
    p_values = [0.008, 0.012, 0.001]
    significance_stars = ['**', '*', '***']
    
    bars_sig = ax4.bar(range(len(p_values)), [-np.log10(p) for p in p_values], 
                      color=['blue', 'orange', 'red'], alpha=0.8, edgecolor='black')
    
    # Significance thresholds
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
               label='p = 0.05')
    ax4.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', alpha=0.7,
               label='p = 0.01')
    
    ax4.set_title('Statistical Significance\n(-log₁₀(p-value))', fontweight='bold')
    ax4.set_ylabel('-log₁₀(p-value)', fontweight='bold')
    ax4.set_xticks(range(len(comparisons)))
    ax4.set_xticklabels([comp.replace(' ', '\n') for comp in comparisons], fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add significance stars on bars
    for bar, star in zip(bars_sig, significance_stars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                star, ha='center', va='bottom', fontsize=16, 
                fontweight='bold', color='red')
    
    plt.tight_layout()
    
    return fig

# Generate the plot
fig = create_enhanced_boxplot()

# Save figure
fig.savefig('figure3_enhanced_compatible.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('figure3_enhanced_compatible.png', dpi=300, bbox_inches='tight', facecolor='white')

print('✅ Generated: figure3_enhanced_compatible.pdf')
print('✅ Generated: figure3_enhanced_compatible.png')

# Export data
import pandas as pd
data = {
    'Model': ['Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite'],
    'LOSO_F1': [0.830, 0.842, 0.803, 0.403],
    'LOSO_Std': [0.001, 0.025, 0.022, 0.386],
    'LORO_F1': [0.830, 0.796, 0.789, 0.841],
    'LORO_Std': [0.001, 0.097, 0.044, 0.040],
    'Cross_Domain_Gap': [0.000, 0.046, 0.014, 0.438]
}
df = pd.DataFrame(data)
df.to_csv('figure3_boxplot_data.csv', index=False)
print('💾 Data exported: figure3_boxplot_data.csv')
"