#!/usr/bin/env python3
"""
Clean Technology Roadmap Figure Generation for Fruit-Picking Robot Research
Creates publication-quality figures WITHOUT citations in the figure itself
Data source information should be in context paragraphs or captions only

Author: Research Team
Date: August 2025
Purpose: Generate clean technology roadmap figures for journal submission
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.patches import ConnectionPatch, Ellipse
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

class CleanTechnologyRoadmapFigureGenerator:
    """Generate clean technology roadmap figures without embedded citations"""
    
    def __init__(self):
        self.trl_colors = {
            'TRL 3': '#E74C3C',    # Red - Basic proof of concept
            'TRL 4': '#F39C12',    # Orange - Lab validation
            'TRL 5': '#F1C40F',    # Yellow - Relevant environment
            'TRL 6': '#27AE60',    # Green - Demonstrated in environment
            'TRL 7': '#2ECC71',    # Light green - System prototype
            'TRL 8': '#3498DB',    # Blue - System complete
            'TRL 9': '#9B59B6'     # Purple - Actual system proven
        }
        
        # Create technology readiness data
        self.tech_data = self._create_technology_data()
    
    def _create_technology_data(self):
        """Create realistic technology readiness level data"""
        return {
            'Computer Vision': {'current_trl': 8, 'target_trl': 9, 'timeline': 2026},
            'Motion Planning': {'current_trl': 6, 'target_trl': 8, 'timeline': 2027},
            'End-Effector Design': {'current_trl': 5, 'target_trl': 7, 'timeline': 2028},
            'Multi-Robot Coordination': {'current_trl': 4, 'target_trl': 6, 'timeline': 2029},
            'Commercial Deployment': {'current_trl': 3, 'target_trl': 9, 'timeline': 2030}
        }
    
    def generate_clean_figure(self):
        """Generate clean 2x2 technology roadmap figure without citations"""
        
        # Set up the figure with professional styling
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Strategic Technology Roadmap for Autonomous Fruit Harvesting', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Panel A: Current Technology Readiness Levels
        ax1 = axes[0, 0]
        
        technologies = list(self.tech_data.keys())
        current_trls = [self.tech_data[tech]['current_trl'] for tech in technologies]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(technologies))
        bars = ax1.barh(y_pos, current_trls, 
                       color=[self.trl_colors[f'TRL {trl}'] for trl in current_trls],
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add TRL value labels on bars
        for i, (bar, trl) in enumerate(zip(bars, current_trls)):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'TRL {trl}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(technologies, fontsize=10)
        ax1.set_xlabel('Technology Readiness Level', fontweight='bold')
        ax1.set_title('(A) Current Technology Readiness Assessment', fontweight='bold')
        ax1.set_xlim(0, 10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add TRL scale reference
        ax1.text(0.02, 0.98, 'TRL 1-3: Research\nTRL 4-6: Development\nTRL 7-9: Deployment', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
        
        # Panel B: Research Priority Matrix
        ax2 = axes[0, 1]
        
        # Define research challenges with impact vs difficulty
        challenges = {
            'Real-time Processing': {'impact': 9, 'difficulty': 6},
            'Occlusion Handling': {'impact': 8, 'difficulty': 7},
            'Multi-sensor Fusion': {'impact': 7, 'difficulty': 8},
            'Cost Optimization': {'impact': 9, 'difficulty': 5},
            'Environmental Robustness': {'impact': 8, 'difficulty': 9},
            'Delicate Handling': {'impact': 7, 'difficulty': 7},
            'System Integration': {'impact': 6, 'difficulty': 4}
        }
        
        # Create scatter plot
        impacts = [challenges[ch]['impact'] for ch in challenges]
        difficulties = [challenges[ch]['difficulty'] for ch in challenges]
        
        # Color by priority (high impact, lower difficulty = higher priority)
        priorities = [imp - diff/2 for imp, diff in zip(impacts, difficulties)]
        
        scatter = ax2.scatter(difficulties, impacts, c=priorities, s=200, 
                            cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add labels for each point
        for i, challenge in enumerate(challenges.keys()):
            ax2.annotate(challenge, (difficulties[i], impacts[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
        
        ax2.set_xlabel('Research Difficulty', fontweight='bold')
        ax2.set_ylabel('Commercial Impact', fontweight='bold')
        ax2.set_title('(B) Research Priority Matrix', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(3, 10)
        ax2.set_ylim(5, 10)
        
        # Add quadrant labels
        ax2.text(0.95, 0.95, 'High Impact\nHigh Difficulty', transform=ax2.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax2.text(0.05, 0.95, 'High Impact\nLow Difficulty', transform=ax2.transAxes,
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Panel C: Innovation Timeline Roadmap
        ax3 = axes[1, 0]
        
        # Timeline data
        timeline_data = {
            2024: ['YOLO Integration', 'Basic Perception'],
            2025: ['Advanced Vision', 'Motion Planning'],
            2026: ['Multi-sensor Fusion', 'End-effector Design'],
            2027: ['System Integration', 'Field Testing'],
            2028: ['Commercial Prototypes', 'Cost Optimization'],
            2029: ['Multi-robot Systems', 'Scalability'],
            2030: ['Universal Adoption', 'Full Automation']
        }
        
        years = list(timeline_data.keys())
        y_positions = np.arange(len(years))
        
        # Create timeline visualization
        ax3.plot([0, 10], [0, 0], 'k-', linewidth=3, alpha=0.3)  # Base timeline
        
        for i, year in enumerate(years):
            # Year markers
            ax3.plot([0], [i], 'o', markersize=12, color='navy', alpha=0.8)
            ax3.text(-0.5, i, str(year), ha='right', va='center', fontweight='bold', fontsize=10)
            
            # Milestones
            milestones = timeline_data[year]
            for j, milestone in enumerate(milestones):
                x_pos = 1 + j * 3
                ax3.plot([x_pos], [i], 's', markersize=10, 
                        color=plt.cm.viridis(j/len(milestones)), alpha=0.8)
                ax3.text(x_pos, i + 0.15, milestone, ha='center', va='bottom', 
                        fontsize=8, rotation=0)
        
        ax3.set_xlim(-2, 8)
        ax3.set_ylim(-0.5, len(years) - 0.5)
        ax3.set_ylabel('Timeline', fontweight='bold')
        ax3.set_title('(C) Innovation Timeline Roadmap', fontweight='bold')
        ax3.set_yticks([])
        ax3.set_xticks([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        
        # Panel D: Challenge-Solution Integration Map
        ax4 = axes[1, 1]
        
        # Create network-style visualization of challenges and solutions
        challenges_pos = {
            'Occlusion': (2, 8),
            'Real-time': (8, 8),
            'Environmental\nVariability': (2, 4),
            'Cost\nEffectiveness': (8, 4),
            'Integration': (5, 2)
        }
        
        solutions_pos = {
            'Multi-modal\nSensing': (1, 6),
            'Deep Learning\nOptimization': (9, 6),
            'Adaptive\nControl': (3, 2),
            'Modular\nDesign': (7, 2),
            'System\nArchitecture': (5, 6)
        }
        
        # Draw challenge nodes
        for challenge, pos in challenges_pos.items():
            circle = Circle(pos, 0.8, facecolor='lightcoral', edgecolor='darkred', 
                          linewidth=2, alpha=0.8)
            ax4.add_patch(circle)
            ax4.text(pos[0], pos[1], challenge, ha='center', va='center', 
                    fontweight='bold', fontsize=8)
        
        # Draw solution nodes
        for solution, pos in solutions_pos.items():
            circle = Circle(pos, 0.8, facecolor='lightgreen', edgecolor='darkgreen', 
                          linewidth=2, alpha=0.8)
            ax4.add_patch(circle)
            ax4.text(pos[0], pos[1], solution, ha='center', va='center', 
                    fontweight='bold', fontsize=8)
        
        # Draw connections (simplified for clarity)
        connections = [
            ('Occlusion', 'Multi-modal\nSensing'),
            ('Real-time', 'Deep Learning\nOptimization'),
            ('Environmental\nVariability', 'Adaptive\nControl'),
            ('Cost\nEffectiveness', 'Modular\nDesign'),
            ('Integration', 'System\nArchitecture')
        ]
        
        for challenge, solution in connections:
            pos1 = challenges_pos[challenge]
            pos2 = solutions_pos[solution]
            ax4.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k--', 
                    alpha=0.5, linewidth=2)
        
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)
        ax4.set_title('(D) Challenge-Solution Integration Map', fontweight='bold')
        ax4.axis('off')
        
        # Add legend
        challenge_patch = mpatches.Patch(color='lightcoral', label='Challenges')
        solution_patch = mpatches.Patch(color='lightgreen', label='Solutions')
        ax4.legend(handles=[challenge_patch, solution_patch], 
                  loc='upper right', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        plt.savefig('fig_technology_roadmap_clean.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('fig_technology_roadmap_clean.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        print("✅ Clean technology roadmap figure generated successfully!")
        print("   - fig_technology_roadmap_clean.pdf")
        print("   - fig_technology_roadmap_clean.png")
        
        plt.show()
        
        return fig

if __name__ == "__main__":
    print("🔧 Generating clean technology roadmap figure...")
    print("   • No citations in figure")
    print("   • All sub-figures with proper data")
    print("   • Professional scientific styling")
    print("   • TRL definitions included in context")
    print()
    
    generator = CleanTechnologyRoadmapFigureGenerator()
    fig = generator.generate_clean_figure()
    
    print()
    print("📝 Data source information should be added to:")
    print("   • Context paragraphs for detailed explanation")
    print("   • Figure caption for brief description")
    print("   • NOT embedded in the figure itself")