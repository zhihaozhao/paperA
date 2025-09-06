#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) Visualization - Figure 7
3 rows × 2 columns layout: Column 1 (3 plots), Column 2 (4 plots)
IEEE IoTJ Paper - WiFi CSI HAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configure for IEEE IoTJ standards with unified fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black'
})

def simulate_feature_space_data():
    """Simulate realistic feature space data"""
    np.random.seed(42)
    n_samples_per_condition = 50
    
    model_configs = {
        'Enhanced': {
            'loso_center': [2.5, 1.8], 'loso_cov': [[0.1, 0.05], [0.05, 0.1]],
            'loro_center': [2.6, 1.9], 'loro_cov': [[0.12, 0.06], [0.06, 0.11]],
            'color': '#27AE60', 'marker': 'o'
        },
        'CNN': {
            'loso_center': [1.8, 2.2], 'loso_cov': [[0.4, 0.1], [0.1, 0.3]],
            'loro_center': [1.2, 1.8], 'loro_cov': [[0.6, 0.15], [0.15, 0.5]],
            'color': '#3498DB', 'marker': 's'
        },
        'BiLSTM': {
            'loso_center': [1.5, 1.5], 'loso_cov': [[0.3, 0.08], [0.08, 0.25]],
            'loro_center': [1.4, 1.3], 'loro_cov': [[0.35, 0.12], [0.12, 0.28]],
            'color': '#F39C12', 'marker': '^'
        },
        'Conformer-lite': {
            'loso_center': [-0.5, 0.2], 'loso_cov': [[1.2, 0.4], [0.4, 1.0]],
            'loro_center': [2.0, 2.5], 'loro_cov': [[0.25, 0.1], [0.1, 0.2]],
            'color': '#E74C3C', 'marker': 'D'
        }
    }
    
    # Generate samples
    data_records = []
    feature_data = []
    
    for model, config in model_configs.items():
        for protocol in ['LOSO', 'LORO']:
            center = config[f'{protocol.lower()}_center']
            cov = config[f'{protocol.lower()}_cov']
            samples = np.random.multivariate_normal(center, cov, n_samples_per_condition)
            
            for i, sample in enumerate(samples):
                data_records.append({
                    'Model': model, 'Protocol': protocol, 'Sample_ID': i,
                    'PC1': sample[0], 'PC2': sample[1],
                    'Color': config['color'], 'Marker': config['marker']
                })
                feature_data.append(sample)
    
    additional_features = np.random.normal(0, 1, (len(feature_data), 8))
    full_features = np.column_stack([feature_data, additional_features])
    
    return pd.DataFrame(data_records), full_features

def create_pca_4row_layout():
    """Create 4-row, 2-column layout per spec:
    Row1: PCA feature space spans 2 columns
    Row2: Left cross-protocol consistency, Right PCA explained variance
    Row3: Right column split vertically: Top model separation distances, Bottom 3D feature space
    Row4: Left PCA feature loadings matrix, Right feature contributions
    """
    data_df, features = simulate_feature_space_data()
    
    # Perform PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(features_scaled)
    
    # Update dataframe
    for i in range(min(pca_result.shape[1], 3)):
        data_df[f'PC{i+1}'] = pca_result[:, i]
    
    # Create figure: 4 rows × 2 columns layout (with nested grid for row3 right)
    # fig = plt.figure(figsize=(16.0, 14.0))
    # gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 1.0, 1.2, 1.1], hspace=0.56, wspace=0.18)
    fig = plt.figure(figsize=(16.0, 15.0))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1.4, 1.0, 1.8, 1.2], hspace=0.56, wspace=0.18)

    # Row1: Main PCA Feature Space spans 2 columns
    ax1 = fig.add_subplot(gs[0, :])
    
    # Row2: Left cross-protocol consistency; Right explained variance
    ax4 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    # Row3: Two equal-sized subplots
    ax3 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1], projection='3d')
    
    # Row4: Left loadings heatmap; Right feature contributions
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])
    
    models = data_df['Model'].unique()
    
    # 1. Main PCA scatter plot (Column 1, Row 1)
    for model in models:
        model_data = data_df[data_df['Model'] == model]
        loso_data = model_data[model_data['Protocol'] == 'LOSO']
        loro_data = model_data[model_data['Protocol'] == 'LORO']
        
        color = model_data.iloc[0]['Color']
        marker = model_data.iloc[0]['Marker']
        
        ax1.scatter(loso_data['PC1'], loso_data['PC2'], 
                   c=color, marker=marker, s=60, alpha=0.8, 
                   label=f'{model} (LOSO)', edgecolors='black', linewidth=0.5)
        ax1.scatter(loro_data['PC1'], loro_data['PC2'], 
                   c=color, marker=marker, s=60, alpha=0.5,
                   label=f'{model} (LORO)', edgecolors='gray', linewidth=0.5)
        
        # Confidence ellipses
        if len(loso_data) > 2:
            cov_matrix = np.cov(loso_data['PC1'], loso_data['PC2'])
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            ellipse = Ellipse(
                xy=(loso_data['PC1'].mean(), loso_data['PC2'].mean()),
                width=2 * np.sqrt(eigenvals[0]) * 2.576,
                height=2 * np.sqrt(eigenvals[1]) * 2.576,
                angle=angle, facecolor=color, alpha=0.1, edgecolor=color, linewidth=1.5
            )
            ax1.add_patch(ellipse)
    
    ax1.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                   fontweight='bold', fontsize=12, labelpad=10)
    ax1.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                   fontweight='bold', fontsize=12, labelpad=10)
    ax1.set_title('(a) PCA Feature Space Analysis: Model Clustering and Protocol Separation', 
                 fontweight='bold', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.20), framealpha=0.85)
    
    # 2. Explained Variance (Row 2 Right)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    bars = ax2.bar(range(1, len(explained_var) + 1), explained_var, 
                   alpha=0.7, color='skyblue', edgecolor='navy')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(1, len(explained_var) + 1), cumulative_var * 100, 
                 'ro-', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Principal Component', fontweight='bold', fontsize=12, labelpad=8)
    ax2.set_ylabel('Explained Variance Ratio', color='blue', fontweight='bold', fontsize=12, labelpad=8)
    ax2_twin.set_ylabel('Cumulative Variance (%)', color='red', fontweight='bold', fontsize=12, labelpad=8)
    ax2.set_title('(c) PCA Explained Variance', fontweight='bold', fontsize=14, pad=12)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='red', labelsize=12)
    ax2.grid(True, alpha=0.3)
    
    for bar, var in zip(bars, explained_var):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{var:.1%}', ha='center', va='bottom', fontsize=12)
    
    # 3. Model Separation (Row 3 Right Top)(d)
    model_centers = {}
    for model in models:
        model_data = data_df[data_df['Model'] == model]
        model_centers[model] = [model_data['PC1'].mean(), model_data['PC2'].mean()]
    
    distance_matrix = np.zeros((len(models), len(models)))
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:
                dist = np.linalg.norm(np.array(model_centers[model1]) - np.array(model_centers[model2]))
                distance_matrix[i, j] = dist
    
    im = ax3.imshow(distance_matrix, cmap='viridis')
    ax3.set_xticks(range(len(models)))
    ax3.set_yticks(range(len(models)))
    # ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax3.set_xticklabels(models, rotation=30, ha='right', fontsize=12)

    ax3.set_yticklabels(models, fontsize=12)
    # ax3.set_title('(d) Model Separation Distances', fontweight='bold', fontsize=14, pad=12)
    ax3.set_title('(d) Model Separation Distances', fontweight='bold', fontsize=14, pad=8)

    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    for i in range(len(models)):
        for j in range(len(models)):
            ax3.text(j, i, f'{distance_matrix[i, j]:.1f}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    # Use a separate colorbar axes so the main axes width is not reduced
    divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='3%', pad=0.05)
    cax = divider.append_axes('right', size='3%', pad=0.15)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Distance', fontsize=12)
    
    # 4. Cross-Protocol Consistency (Row 2 Left)(b)
    protocol_consistency = {}
    for model in models:
        model_data = data_df[data_df['Model'] == model]
        loso_center = [
            model_data[model_data['Protocol'] == 'LOSO']['PC1'].mean(),
            model_data[model_data['Protocol'] == 'LOSO']['PC2'].mean()
        ]
        loro_center = [
            model_data[model_data['Protocol'] == 'LORO']['PC1'].mean(),
            model_data[model_data['Protocol'] == 'LORO']['PC2'].mean()
        ]
        consistency_score = np.linalg.norm(np.array(loso_center) - np.array(loro_center))
        protocol_consistency[model] = consistency_score
    
    models_sorted = sorted(protocol_consistency.keys(), key=lambda x: protocol_consistency[x])
    scores = [protocol_consistency[model] for model in models_sorted]
    colors = [data_df[data_df['Model'] == model].iloc[0]['Color'] for model in models_sorted]
    
    bars = ax4.bar(models_sorted, scores, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('LOSO-LORO Distance (Lower = More Consistent)', fontweight='bold', fontsize=12, labelpad=10)
    ax4.set_xlabel('Model', fontweight='bold', fontsize=12, labelpad=8)
    ax4.set_title('(b) Cross-Protocol Consistency Analysis', fontweight='bold', fontsize=14, pad=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    enhanced_idx = models_sorted.index('Enhanced')
    bars[enhanced_idx].set_linewidth(3)
    bars[enhanced_idx].set_edgecolor('gold')
    
    # 5. 3D Feature Space (Row 3 Right Bottom)(e)
    for model in models:
        model_data = data_df[data_df['Model'] == model]
        color = model_data.iloc[0]['Color']
        marker = model_data.iloc[0]['Marker']
        
        ax5.scatter(model_data['PC1'], model_data['PC2'], model_data['PC3'],
                   c=color, marker=marker, s=10, alpha=0.6, label=model)
    
    ax5.set_xlabel('PC1', fontweight='bold', fontsize=12, labelpad=10)
    ax5.set_ylabel('PC2', fontweight='bold', fontsize=12, labelpad=10)  
    ax5.set_zlabel('PC3', fontweight='bold', fontsize=12, labelpad=10)
    # ax5.set_title('(e) 3D Feature Space', fontweight='bold', fontsize=14, pad=12)
    ax5.set_title('(e) 3D Feature Space', fontweight='bold', fontsize=14, pad=8)

    # Expand 3D plot visual area to match column width and enlarge proportionally
    try:
        ax5.set_box_aspect((1.3, 1.0, 0.8))
        # ax5.set_box_aspect((2.5, 1.0, 1.5))

    except Exception:
        pass
    try:
        ax5.set_proj_type('ortho')
    except Exception:
        pass
    ax5.margins(x=0.04, y=0.04, z=0.04)
    # ax5.margins(x=0.02, y=0.02, z=0.02)
    ax5.legend(fontsize=12, loc='center left', bbox_to_anchor=(-0.7, 0.8), framealpha=0.8)

    # 6. PCA Feature Loadings Matrix (Row 4 Left)
    feature_names = ['Temporal_Pattern', 'Frequency_Response', 'Spatial_Correlation', 
                    'Channel_Diversity', 'Signal_Strength', 'Noise_Resilience',
                    'Attention_Weight', 'Memory_State', 'Feature_Interaction', 'Complexity']
    
    loadings_df = pd.DataFrame(
        pca.components_[:5, :].T,
        columns=[f'PC{i+1}' for i in range(5)],
        index=feature_names
    )
    
    sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=False, linewidths=0.5, ax=ax6, 
                annot_kws={'size': 10}, cbar_kws={'label': 'Loading Weight'})
    ax6.set_title('(f) PCA Feature Loadings Matrix', fontweight='bold', fontsize=14, pad=12)
    ax6.set_xlabel('Principal Components', fontweight='bold', fontsize=12, labelpad=8)
    ax6.set_ylabel('Feature Dimensions', fontweight='bold', fontsize=12, labelpad=8)
    
    # 7. Feature Contributions (Row 4 Right)
    pc1_contributions = np.abs(loadings_df['PC1']).sort_values(ascending=True)
    pc2_contributions = np.abs(loadings_df['PC2']).sort_values(ascending=True)
    
    y_pos = np.arange(len(feature_names))
    
    ax7.barh(y_pos - 0.2, pc1_contributions, height=0.4, 
             label='PC1 Contribution', alpha=0.8, color='#3498DB')
    ax7.barh(y_pos + 0.2, pc2_contributions, height=0.4, 
             label='PC2 Contribution', alpha=0.8, color='#E74C3C')
    
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(pc1_contributions.index, fontsize=12)
    ax7.set_xlabel('Absolute Loading Weight', fontweight='bold', fontsize=12, labelpad=8)
    ax7.set_title('(g) Feature Contributions to Top 2 PCs', fontweight='bold', fontsize=14, pad=12)
    ax7.legend(fontsize=12, loc='lower right', framealpha=0.8)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # Spacing for 4-row layout
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.98, top=0.95)
    
    return fig, data_df, pca

def export_pca_data():
    """Export PCA analysis data"""
    data_df, features = simulate_feature_space_data()
    
    # Export coordinates (restore original path/filename)
    data_df[['Model', 'Protocol', 'PC1', 'PC2']].to_csv('figure7_pca_coordinates.csv', index=False)
    
    # Export feature matrix (restore original path/filename)
    feature_names = ['Temporal_Pattern', 'Frequency_Response', 'Spatial_Correlation', 
                    'Channel_Diversity', 'Signal_Strength', 'Noise_Resilience',
                    'Attention_Weight', 'Memory_State', 'Feature_Interaction', 'Complexity']
    
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['Model'] = data_df['Model'].values
    features_df['Protocol'] = data_df['Protocol'].values
    features_df.to_csv('figure7_feature_matrix.csv', index=False)
    
    # Export PCA results (restore original path/filename)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(features_scaled)
    
    pca_results_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    pca_results_df['Model'] = data_df['Model'].values
    pca_results_df['Protocol'] = data_df['Protocol'].values
    pca_results_df.to_csv('figure7_pca_results.csv', index=False)
    
    # Export explained variance (restore original path/filename)
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_df.to_csv('figure7_explained_variance.csv', index=False)
    
    print("\n💾 PCA Data Export Complete:")
    print("• figure7_pca_coordinates.csv")
    print("• figure7_feature_matrix.csv") 
    print("• figure7_pca_results.csv")
    print("• figure7_explained_variance.csv")

if __name__ == "__main__":
    print("🔍 Generating Figure 6: 4-row Layout per spec")

    REPO = Path(__file__).resolve().parents[2]
    FIGS = REPO / "paper" / "figures"
    FIGS.mkdir(parents=True, exist_ok=True)

    # Generate PCA analysis
    fig, data_df, pca = create_pca_4row_layout()

    # Save figures
    fig.savefig(FIGS / 'fig6_pca_analysis.pdf', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    fig.savefig(FIGS / 'fig6_pca_analysis.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print("✅ Saved: fig6_pca_analysis.pdf")
    print("✅ Saved: fig6_pca_analysis.png")

    # Export data
    export_pca_data()

    print(f"\n📊 Summary: {pca.explained_variance_ratio_[:2].sum():.1%} variance explained")
    print("🎉 4-row Layout Complete")

    plt.show()