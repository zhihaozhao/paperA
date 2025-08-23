#!/usr/bin/env python3
"""
Create Section V Figure: Motion Planning System Architecture and Algorithm Comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

def create_motion_planning_figure():
    """Create comprehensive motion planning visualization"""
    
    # Set style for professional publication
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Main title removed to prevent overlap (as per Bug 2 fix)
    
    # Panel A: Motion Planning System Architecture
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    
    # Draw system components
    # Perception Layer
    perception = patches.Rectangle((0.5, 6), 2, 1.5, linewidth=2, edgecolor='#2E86AB', facecolor='#A23B72', alpha=0.7)
    ax1.add_patch(perception)
    ax1.text(1.5, 6.75, 'Visual\nPerception', ha='center', va='center', fontweight='bold', color='white')
    
    # Planning Layer  
    planning = patches.Rectangle((4, 6), 2, 1.5, linewidth=2, edgecolor='#F18F01', facecolor='#C73E1D', alpha=0.7)
    ax1.add_patch(planning)
    ax1.text(5, 6.75, 'Motion\nPlanning', ha='center', va='center', fontweight='bold', color='white')
    
    # Control Layer
    control = patches.Rectangle((7.5, 6), 2, 1.5, linewidth=2, edgecolor='#6A994E', facecolor='#386641', alpha=0.7)
    ax1.add_patch(control)
    ax1.text(8.5, 6.75, 'Control\nExecution', ha='center', va='center', fontweight='bold', color='white')
    
    # Environment
    environment = patches.Rectangle((2, 3.5), 6, 1.5, linewidth=2, edgecolor='#7209B7', facecolor='#560BAD', alpha=0.7)
    ax1.add_patch(environment)
    ax1.text(5, 4.25, 'Agricultural Environment\n(Dynamic Obstacles, Variable Conditions)', ha='center', va='center', fontweight='bold', color='white')
    
    # Robot System
    robot = patches.Rectangle((4, 1), 2, 1.5, linewidth=2, edgecolor='#BC6C25', facecolor='#DDB892', alpha=0.7)
    ax1.add_patch(robot)
    ax1.text(5, 1.75, 'Robotic\nHarvester', ha='center', va='center', fontweight='bold')
    
    # Add arrows showing information flow
    ax1.arrow(2.5, 6.5, 1.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax1.arrow(6, 6.5, 1.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax1.arrow(5, 5.8, 0, -0.8, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax1.arrow(5, 3.3, 0, -0.6, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    ax1.set_title('(A) Motion Planning System Architecture', fontweight='bold', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Panel B: Algorithm Performance Comparison
    ax2 = plt.subplot(2, 2, 2)
    
    algorithms = ['A*', 'RRT*', 'DDPG', 'Bi-RRT', 'DWA']
    success_rates = [75, 68, 82, 71, 79]
    computation_times = [45, 85, 120, 65, 25]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    scatter = ax2.scatter(computation_times, success_rates, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    for i, alg in enumerate(algorithms):
        ax2.annotate(alg, (computation_times[i], success_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Computation Time (ms)', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('(B) Algorithm Performance Trade-offs', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(15, 130)
    ax2.set_ylim(60, 90)
    
    # Panel C: Motion Planning Approaches Timeline
    ax3 = plt.subplot(2, 2, 3)
    
    years = [2015, 2017, 2019, 2021, 2023, 2024]
    traditional = [12, 15, 10, 8, 5, 3]
    rl_based = [2, 5, 8, 12, 18, 22]
    hybrid = [3, 6, 9, 15, 20, 25]
    
    ax3.plot(years, traditional, 'o-', linewidth=3, markersize=8, label='Traditional (A*, RRT)', color='#E74C3C')
    ax3.plot(years, rl_based, 's-', linewidth=3, markersize=8, label='RL-based (DDPG)', color='#3498DB')
    ax3.plot(years, hybrid, '^-', linewidth=3, markersize=8, label='Hybrid Approaches', color='#2ECC71')
    
    ax3.set_xlabel('Publication Year', fontweight='bold')
    ax3.set_ylabel('Number of Papers', fontweight='bold')
    ax3.set_title('(C) Motion Planning Algorithm Adoption', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Motion Control Performance by Environment
    ax4 = plt.subplot(2, 2, 4)
    
    environments = ['Greenhouse', 'Orchard\n(Structured)', 'Orchard\n(Unstructured)', 'Field\n(Open)']
    success_rates_env = [85, 78, 65, 58]
    colors_env = ['#27AE60', '#F39C12', '#E67E22', '#C0392B']
    
    bars = ax4.bar(environments, success_rates_env, color=colors_env, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (env, rate) in enumerate(zip(environments, success_rates_env)):
        ax4.text(i, rate + 2, f'{rate}%', ha='center', fontweight='bold', fontsize=11)
    
    ax4.set_ylabel('Average Success Rate (%)', fontweight='bold')
    ax4.set_title('(D) Performance by Environment Type', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Improve spacing
    plt.tight_layout(pad=3.0, h_pad=3.5, w_pad=3.0)
    
    # Save figure
    plt.savefig('fig_motion_planning_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig('fig_motion_planning_analysis.pdf', bbox_inches='tight', pad_inches=0.2)
    print("✅ Motion planning figure created: fig_motion_planning_analysis.png/.pdf")
    plt.close()

def create_future_directions_figure():
    """Create future directions and technology roadmap visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Technology Readiness Levels
    technologies = ['Computer\nVision', 'Motion\nPlanning', 'Multi-Robot\nCoordination', 'Commercial\nDeployment']
    trl_levels = [8, 6, 4, 3]
    colors_trl = ['#27AE60', '#F39C12', '#E67E22', '#C0392B']
    
    bars1 = ax1.barh(technologies, trl_levels, color=colors_trl, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add TRL value labels
    for i, trl in enumerate(trl_levels):
        ax1.text(trl + 0.1, i, f'TRL {trl}', va='center', fontweight='bold', fontsize=11)
    
    ax1.set_xlabel('Technology Readiness Level', fontweight='bold')
    ax1.set_title('(A) Current Technology Readiness', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Panel B: Research Priority Matrix  
    challenges = ['Multi-Sensor\nFusion', 'Cost\nReduction', 'Environmental\nRobustness', 'Real-time\nPerformance', 'Commercial\nScalability']
    impact = [9, 8, 7, 8.5, 9.5]
    difficulty = [7, 9, 8, 6, 9]
    
    scatter2 = ax2.scatter(difficulty, impact, s=[200, 250, 180, 220, 280], 
                          c=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'], 
                          alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, challenge in enumerate(challenges):
        ax2.annotate(challenge, (difficulty[i], impact[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=9, ha='center')
    
    ax2.set_xlabel('Research Difficulty', fontweight='bold')
    ax2.set_ylabel('Commercial Impact', fontweight='bold')
    ax2.set_title('(B) Research Priority Matrix', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 10)
    ax2.set_ylim(6, 10)
    
    # Panel C: Innovation Timeline Roadmap
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    
    # Timeline components
    milestones = [
        (1, 5, '2024-2025\nYOLO Integration'),
        (3, 4, '2025-2026\nRL Motion Control'),
        (5, 5, '2026-2027\nMulti-Robot Fleets'),
        (7, 3, '2027-2028\nCommercial\nDeployment'),
        (9, 4, '2028-2030\nUniversal\nAdoption')
    ]
    
    # Draw timeline
    ax3.plot([0.5, 9.5], [2, 2], 'k-', linewidth=3)
    
    for x, y, text in milestones:
        circle = patches.Circle((x, 2), 0.2, facecolor='#3498DB', edgecolor='black', linewidth=2)
        ax3.add_patch(circle)
        ax3.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax3.plot([x, x], [1.8, y-0.3], 'k--', alpha=0.5)
    
    ax3.set_title('(C) Innovation Roadmap Timeline', fontweight='bold', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # Panel D: Challenge-Solution Mapping
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # Challenge boxes (left side)
    challenges_map = [
        (1.5, 8.5, 'Occlusion\nHandling'),
        (1.5, 6.5, 'Real-time\nConstraints'),
        (1.5, 4.5, 'Environmental\nVariability'),
        (1.5, 2.5, 'Cost\nEffectiveness')
    ]
    
    # Solution boxes (right side)
    solutions_map = [
        (8.5, 8.5, 'Multi-Modal\nSensing'),
        (8.5, 6.5, 'Edge\nComputing'),
        (8.5, 4.5, 'Adaptive\nAlgorithms'),
        (8.5, 2.5, 'Modular\nDesign')
    ]
    
    # Draw challenge boxes
    for x, y, text in challenges_map:
        challenge_box = patches.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                        facecolor='#E74C3C', alpha=0.7, edgecolor='black', linewidth=2)
        ax4.add_patch(challenge_box)
        ax4.text(x, y, text, ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    
    # Draw solution boxes
    for x, y, text in solutions_map:
        solution_box = patches.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                                       facecolor='#27AE60', alpha=0.7, edgecolor='black', linewidth=2)
        ax4.add_patch(solution_box)
        ax4.text(x, y, text, ha='center', va='center', fontweight='bold', color='white', fontsize=9)
    
    # Draw connecting arrows
    for i, ((cx, cy, _), (sx, sy, _)) in enumerate(zip(challenges_map, solutions_map)):
        ax4.arrow(cx+0.7, cy, sx-cx-1.4, sy-cy, head_width=0.2, head_length=0.3, 
                 fc='#34495E', ec='#34495E', linewidth=2)
    
    ax4.set_title('(D) Challenge-Solution Integration', fontweight='bold', fontsize=12)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    
    plt.tight_layout(pad=3.0, h_pad=3.5, w_pad=3.0)
    
    # Save figure
    plt.savefig('fig_future_directions_roadmap.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig('fig_future_directions_roadmap.pdf', bbox_inches='tight', pad_inches=0.2)
    print("✅ Future directions figure created: fig_future_directions_roadmap.png/.pdf")
    plt.close()

if __name__ == "__main__":
    print("=== CREATING SECTION V & VI FIGURES ===")
    
    try:
        create_motion_planning_figure()
        create_future_directions_figure()
        print("\n🎉 BOTH FIGURES CREATED SUCCESSFULLY!")
        print("📁 Generated files:")
        print("  • fig_motion_planning_analysis.png/.pdf (Section V)")
        print("  • fig_future_directions_roadmap.png/.pdf (Section VI)")
        print("✅ Ready for integration into all journal versions!")
    
    except Exception as e:
        print(f"❌ Error creating figures: {e}")
        print("Please check matplotlib installation and dependencies")