#!/usr/bin/env python3
"""
Simple Figure Generator for Literature Data (Fallback)
Creates basic publication-quality figures with minimal dependencies

Author: Research Team  
Date: 2024
Purpose: Fallback figure generation when full pipeline fails
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class SimpleFigureGenerator:
    """Generate basic figures with minimal dependencies"""
    
    def __init__(self, data_file: str = 'meta_analysis_output/fruit_picking_literature_data.csv'):
        self.data_file = data_file
        self.df = None
        
        # Set basic plotting style
        plt.rcParams.update({
            'font.size': 10,
            'axes.linewidth': 1,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def load_data(self):
        """Load extracted data"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} studies")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_basic_performance_figure(self):
        """Create basic performance analysis figure"""
        print("Creating basic performance figure...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Literature Performance Analysis (Basic)', fontsize=14, fontweight='bold')
        
        # Panel A: Performance by Year
        ax1 = axes[0, 0]
        if 'year' in self.df.columns and 'accuracy' in self.df.columns:
            yearly_data = self.df.groupby('year')['accuracy'].mean().dropna()
            if len(yearly_data) > 0:
                ax1.plot(yearly_data.index, yearly_data.values * 100, 'o-', linewidth=2)
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Average Accuracy (%)')
                ax1.set_title('(A) Performance Over Time')
                ax1.grid(True, alpha=0.3)
        
        # Panel B: Algorithm Family Distribution
        ax2 = axes[0, 1]
        if 'algorithm_family' in self.df.columns:
            algo_counts = self.df['algorithm_family'].value_counts()
            if len(algo_counts) > 0:
                ax2.pie(algo_counts.values, labels=algo_counts.index, autopct='%1.1f%%')
                ax2.set_title('(B) Algorithm Family Distribution')
        
        # Panel C: Performance Distribution
        ax3 = axes[1, 0]
        if 'accuracy' in self.df.columns:
            accuracy_data = self.df['accuracy'].dropna() * 100
            if len(accuracy_data) > 0:
                ax3.hist(accuracy_data, bins=15, alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Accuracy (%)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('(C) Accuracy Distribution')
                ax3.grid(True, alpha=0.3)
        
        # Panel D: Speed vs Accuracy
        ax4 = axes[1, 1]
        speed_acc_data = self.df.dropna(subset=['speed_ms', 'accuracy'])
        if len(speed_acc_data) > 3:
            ax4.scatter(speed_acc_data['speed_ms'], speed_acc_data['accuracy'] * 100, alpha=0.6)
            ax4.set_xlabel('Processing Speed (ms)')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('(D) Speed vs Accuracy')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fig_basic_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_basic_performance_analysis.pdf', bbox_inches='tight')
        print("Basic figure saved: fig_basic_performance_analysis.png/.pdf")
        plt.close()
    
    def create_basic_timeline_figure(self):
        """Create basic technology timeline figure"""
        print("Creating basic timeline figure...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        if 'year' in self.df.columns and 'algorithm_family' in self.df.columns:
            # Algorithm adoption over time
            adoption_data = pd.crosstab(self.df['year'], self.df['algorithm_family'])
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(adoption_data.columns)))
            
            for i, algo in enumerate(adoption_data.columns):
                ax.plot(adoption_data.index, adoption_data[algo], 'o-', 
                       color=colors[i], label=algo, linewidth=2, markersize=6)
            
            ax.set_xlabel('Publication Year')
            ax.set_ylabel('Number of Papers')
            ax.set_title('Algorithm Family Adoption Over Time')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fig_basic_timeline.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_basic_timeline.pdf', bbox_inches='tight')
        print("Timeline figure saved: fig_basic_timeline.png/.pdf")
        plt.close()
    
    def generate_basic_figures(self):
        """Generate all basic figures"""
        if not self.load_data():
            return False
        
        print("Generating basic figures with minimal dependencies...")
        
        try:
            self.create_basic_performance_figure()
            self.create_basic_timeline_figure()
            
            print("\n✅ BASIC FIGURES GENERATED SUCCESSFULLY!")
            print("Files created:")
            print("  • fig_basic_performance_analysis.png/.pdf")
            print("  • fig_basic_timeline.png/.pdf")
            print("\nThese can be used as fallback if full pipeline fails.")
            
            return True
        except Exception as e:
            print(f"❌ Basic figure generation failed: {e}")
            return False

def main():
    """Main function"""
    print("Starting Simple Figure Generation...")
    
    generator = SimpleFigureGenerator()
    success = generator.generate_basic_figures()
    
    if success:
        print("\n🎉 Basic figures ready for integration!")
    else:
        print("\n❌ Figure generation failed.")
    
    return success

if __name__ == "__main__":
    main()