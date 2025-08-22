#!/usr/bin/env python3
"""
Simple Meta-Analysis for Fruit-Picking Robot Literature
Works with basic Python packages and curated dataset

Author: Research Team
Date: 2024
Purpose: Reliable meta-analysis with minimal dependencies
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class SimpleMetaAnalysis:
    """Simple meta-analysis with basic statistics"""
    
    def __init__(self, data_file: str = 'curated_literature_data.csv'):
        self.data_file = data_file
        self.df = None
        
        # Set plotting style
        plt.rcParams.update({
            'font.size': 11,
            'axes.linewidth': 1.2,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        # Color palette
        self.colors = {
            'R-CNN': '#E74C3C',
            'YOLO': '#3498DB',
            'SSD': '#2ECC71', 
            'Hybrid': '#F39C12',
            'Traditional': '#9B59B6',
            'Other': '#95A5A6'
        }
    
    def load_data(self):
        """Load curated dataset"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(self.df)} curated studies")
            
            # Data summary
            print(f"   📊 Year range: {self.df['year'].min()}-{self.df['year'].max()}")
            print(f"   🤖 Algorithm families: {self.df['algorithm_family'].nunique()}")
            print(f"   🍎 Fruit types: {self.df['fruit_type'].nunique()}")
            print(f"   🏭 Environments: {self.df['environment'].nunique()}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def basic_statistics(self):
        """Calculate basic statistics"""
        print("\n=== BASIC STATISTICS ===")
        
        # Overall performance statistics
        accuracy_mean = self.df['accuracy'].mean()
        accuracy_std = self.df['accuracy'].std()
        f1_mean = self.df['f1_score'].mean()
        f1_std = self.df['f1_score'].std()
        speed_mean = self.df['speed_ms'].mean()
        speed_std = self.df['speed_ms'].std()
        
        print(f"Overall Performance:")
        print(f"  • Accuracy: {accuracy_mean:.2f}% ± {accuracy_std:.2f}%")
        print(f"  • F1-Score: {f1_mean:.3f} ± {f1_std:.3f}")
        print(f"  • Speed: {speed_mean:.1f} ± {speed_std:.1f} ms/image")
        
        # Performance by algorithm family
        print(f"\nPerformance by Algorithm Family:")
        algo_stats = self.df.groupby('algorithm_family').agg({
            'accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std'],
            'speed_ms': ['mean', 'std']
        }).round(3)
        
        for algo in self.df['algorithm_family'].unique():
            if pd.notna(algo):
                subset = self.df[self.df['algorithm_family'] == algo]
                acc_mean = subset['accuracy'].mean()
                acc_count = subset['accuracy'].count()
                print(f"  • {algo}: {acc_mean:.2f}% (n={acc_count})")
        
        return algo_stats
    
    def temporal_analysis(self):
        """Analyze temporal trends"""
        print("\n=== TEMPORAL TREND ANALYSIS ===")
        
        # Yearly performance
        yearly_performance = self.df.groupby('year')['accuracy'].agg(['mean', 'std', 'count'])
        
        # Simple linear trend
        years = yearly_performance.index.values
        accuracy_means = yearly_performance['mean'].values
        
        # Remove NaN values
        valid_idx = ~np.isnan(accuracy_means)
        if np.sum(valid_idx) > 3:
            years_clean = years[valid_idx]
            accuracy_clean = accuracy_means[valid_idx]
            
            # Linear fit
            slope = np.polyfit(years_clean, accuracy_clean, 1)[0]
            
            # Calculate R-squared manually
            y_pred = np.polyval([slope, np.polyfit(years_clean, accuracy_clean, 1)[1]], years_clean)
            y_mean = np.mean(accuracy_clean)
            ss_tot = np.sum((accuracy_clean - y_mean) ** 2)
            ss_res = np.sum((accuracy_clean - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"Annual improvement rate: {slope:.3f}% per year")
            print(f"R-squared: {r_squared:.3f}")
            
            # Future projection
            future_accuracy_2025 = slope * 2025 + np.polyfit(years_clean, accuracy_clean, 1)[1]
            print(f"Projected 2025 accuracy: {future_accuracy_2025:.2f}%")
            
            return slope, r_squared
        else:
            print("Insufficient data for trend analysis")
            return None, None
    
    def create_comprehensive_figure(self):
        """Create comprehensive analysis figure"""
        print("\n=== CREATING COMPREHENSIVE FIGURE ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        # Removed main title to prevent overlap with subplot titles
        # Individual subplot titles provide sufficient identification
        
        # Panel A: Performance Evolution
        ax1 = axes[0, 0]
        yearly_data = self.df.groupby('year')['accuracy'].mean().dropna()
        if len(yearly_data) > 0:
            ax1.plot(yearly_data.index, yearly_data.values, 'o-', linewidth=2, markersize=8, color='#2C3E50')
            
            # Add trend line
            if len(yearly_data) > 3:
                z = np.polyfit(yearly_data.index, yearly_data.values, 1)
                p = np.poly1d(z)
                ax1.plot(yearly_data.index, p(yearly_data.index), '--', color='#E74C3C', linewidth=2)
                
                # Future projection
                future_years = [2025, 2026, 2027]
                future_acc = p(future_years)
                ax1.plot(future_years, future_acc, 's-', color='#E74C3C', linewidth=2, markersize=8)
            
            ax1.set_xlabel('Publication Year')
            ax1.set_ylabel('Average Accuracy (%)')
            ax1.set_title('(A) Performance Evolution & Projections')
            ax1.grid(True, alpha=0.3)
        
        # Panel B: Algorithm Performance Comparison
        ax2 = axes[0, 1]
        algo_data = []
        algo_labels = []
        algo_colors = []
        
        for algo in self.df['algorithm_family'].unique():
            if pd.notna(algo):
                acc_values = self.df[self.df['algorithm_family'] == algo]['accuracy'].dropna()
                if len(acc_values) > 0:
                    algo_data.append(acc_values)
                    algo_labels.append(f"{algo}\n(n={len(acc_values)})")
                    algo_colors.append(self.colors.get(algo, '#95A5A6'))
        
        if algo_data:
            bp = ax2.boxplot(algo_data, labels=algo_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], algo_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_ylabel('Detection Accuracy (%)')
            ax2.set_title('(B) Algorithm Family Performance')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Panel C: Speed vs Accuracy Trade-off
        ax3 = axes[0, 2]
        speed_acc_data = self.df.dropna(subset=['speed_ms', 'accuracy'])
        
        if len(speed_acc_data) > 5:
            for algo in speed_acc_data['algorithm_family'].unique():
                if pd.notna(algo):
                    algo_subset = speed_acc_data[speed_acc_data['algorithm_family'] == algo]
                    if len(algo_subset) > 0:
                        ax3.scatter(algo_subset['speed_ms'], algo_subset['accuracy'], 
                                  c=self.colors.get(algo, '#95A5A6'), label=algo, alpha=0.7, s=60)
            
            ax3.set_xlabel('Processing Speed (ms/image)')
            ax3.set_ylabel('Detection Accuracy (%)')
            ax3.set_title('(C) Speed-Accuracy Trade-off')
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(True, alpha=0.3)
            # Fix x-axis label spacing
            ax3.tick_params(axis='x', rotation=0, labelsize=9)
            plt.setp(ax3.get_xticklabels(), ha='center')
        
        # Panel D: Environmental Performance
        ax4 = axes[1, 0]
        env_performance = self.df.groupby('environment')['accuracy'].agg(['mean', 'count']).sort_values('mean', ascending=True)
        
        if len(env_performance) > 0:
            bars = ax4.barh(range(len(env_performance)), env_performance['mean'], 
                           color=['#E74C3C', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6'][:len(env_performance)])
            
            # Add count labels
            for i, (env, row) in enumerate(env_performance.iterrows()):
                ax4.text(row['mean'] + 1, i, f"n={int(row['count'])}", va='center', fontweight='bold')
            
            ax4.set_yticks(range(len(env_performance)))
            ax4.set_yticklabels(env_performance.index)
            ax4.set_xlabel('Average Accuracy (%)')
            ax4.set_title('(D) Environmental Performance Impact')
            ax4.grid(True, alpha=0.3)
        
        # Panel E: Algorithm Adoption Timeline  
        ax5 = axes[1, 1]
        adoption_data = pd.crosstab(self.df['year'], self.df['algorithm_family'])
        
        for algo in adoption_data.columns:
            if algo in self.colors:
                ax5.plot(adoption_data.index, adoption_data[algo], 'o-', 
                        color=self.colors[algo], label=algo, linewidth=2, markersize=6)
        
        ax5.set_xlabel('Publication Year')
        ax5.set_ylabel('Number of Papers')
        ax5.set_title('(E) Algorithm Adoption Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel F: Performance Summary by Fruit Type
        ax6 = axes[1, 2]
        fruit_performance = self.df.groupby('fruit_type')['accuracy'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        if len(fruit_performance) > 0:
            # Take top 8 fruit types to avoid overcrowding
            top_fruits = fruit_performance.head(8)
            
            bars = ax6.bar(range(len(top_fruits)), top_fruits['mean'], 
                          color='#3498DB', alpha=0.7, edgecolor='black')
            
            ax6.set_xticks(range(len(top_fruits)))
            ax6.set_xticklabels(top_fruits.index, rotation=45, ha='right')
            ax6.set_ylabel('Average Accuracy (%)')
            ax6.set_title('(F) Performance by Fruit Type')
            ax6.grid(True, alpha=0.3)
        
        # Improved spacing to prevent overlaps
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        
        # Save figure
        plt.savefig('fig_comprehensive_meta_analysis.png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.2)
        plt.savefig('fig_comprehensive_meta_analysis.pdf', bbox_inches='tight', 
                   pad_inches=0.2)
        print("✅ Comprehensive figure saved: fig_comprehensive_meta_analysis.png/.pdf")
        plt.close()
    
    def generate_summary_table(self):
        """Generate statistical summary table for LaTeX"""
        print("\n=== GENERATING SUMMARY TABLE ===")
        
        # Algorithm family summary
        summary_stats = self.df.groupby('algorithm_family').agg({
            'accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'speed_ms': ['mean', 'std'],
            'year': ['min', 'max']
        }).round(2)
        
        # Create LaTeX table content
        latex_table = """
\\begin{table}[htbp]
\\centering
\\small
\\caption{Meta-Analysis Summary: Algorithm Family Performance Statistics (49 Studies, 2015-2024)}
\\label{tab:meta_analysis_summary}
\\renewcommand{\\arraystretch}{1.3}
\\begin{tabular}{p{2cm}p{1.5cm}p{2cm}p{2cm}p{1.5cm}p{2cm}}
\\toprule
\\textbf{Algorithm} & \\textbf{Studies} & \\textbf{Accuracy (\\%)} & \\textbf{Speed (ms)} & \\textbf{Period} & \\textbf{Trend} \\\\
\\midrule
"""
        
        for algo, row in summary_stats.iterrows():
            if row[('accuracy', 'count')] > 0:
                acc_mean = row[('accuracy', 'mean')]
                acc_std = row[('accuracy', 'std')]
                speed_mean = row[('speed_ms', 'mean')] if not pd.isna(row[('speed_ms', 'mean')]) else 'N/A'
                year_range = f"{int(row[('year', 'min')])}-{int(row[('year', 'max')])}"
                count = int(row[('accuracy', 'count')])
                
                # Determine trend
                algo_subset = self.df[self.df['algorithm_family'] == algo]
                recent_acc = algo_subset[algo_subset['year'] >= 2020]['accuracy'].mean()
                early_acc = algo_subset[algo_subset['year'] <= 2019]['accuracy'].mean()
                
                if pd.notna(recent_acc) and pd.notna(early_acc):
                    trend = "↗" if recent_acc > early_acc else "↘"
                else:
                    trend = "→"
                
                speed_str = f"{speed_mean:.0f}" if isinstance(speed_mean, (int, float)) else speed_mean
                
                latex_table += f"{algo} & {count} & {acc_mean:.1f}±{acc_std:.1f} & {speed_str} & {year_range} & {trend} \\\\\n"
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        # Save LaTeX table
        with open('meta_analysis_summary_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ LaTeX summary table saved: meta_analysis_summary_table.tex")
        return summary_stats
    
    def run_analysis(self):
        """Run complete simple meta-analysis"""
        print("🔬 STARTING SIMPLE META-ANALYSIS")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        try:
            # Basic statistics
            stats = self.basic_statistics()
            
            # Temporal analysis
            slope, r_squared = self.temporal_analysis()
            
            # Create comprehensive figure
            self.create_comprehensive_figure()
            
            # Generate summary table
            self.generate_summary_table()
            
            print("\n" + "=" * 60)
            print("🎉 SIMPLE META-ANALYSIS COMPLETED SUCCESSFULLY!")
            print("📁 Generated files:")
            print("  • fig_comprehensive_meta_analysis.png/.pdf")
            print("  • meta_analysis_summary_table.tex")
            print("✅ Ready for journal integration!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Meta-analysis failed: {e}")
            return False

def main():
    """Main function"""
    analyzer = SimpleMetaAnalysis()
    success = analyzer.run_analysis()
    
    if success:
        print("\n🚀 READY FOR JOURNAL INTEGRATION!")
        print("Next steps:")
        print("1. Copy fig_comprehensive_meta_analysis.png to journal directories")
        print("2. Replace old tables with meta_analysis_summary_table.tex content")
        print("3. Update paper text to reference meta-analysis results")
    
    return success

if __name__ == "__main__":
    main()