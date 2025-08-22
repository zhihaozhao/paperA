#!/usr/bin/env python3
"""
Meta-Analysis Statistical Experiments for Fruit-Picking Robot Literature
Performs comprehensive statistical analysis on extracted literature data

Author: Research Team
Date: 2024
Purpose: Generate statistical insights and high-order figures from literature meta-analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MetaAnalysisExperiments:
    """Comprehensive statistical analysis of literature data"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.results = {}
        
        # Set publication-quality plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load extracted literature data"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded data: {len(self.df)} studies from {self.df['year'].min()}-{self.df['year'].max()}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def experiment_1_performance_distribution(self):
        """Experiment 1: Performance Distribution Analysis"""
        print("\n=== Experiment 1: Performance Distribution Analysis ===")
        
        # Descriptive statistics by algorithm family
        performance_stats = self.df.groupby('algorithm_family').agg({
            'accuracy': ['mean', 'std', 'median', 'count'],
            'f1_score': ['mean', 'std', 'median', 'count'],
            'speed_ms': ['mean', 'std', 'median', 'count']
        }).round(3)
        
        print("Performance Statistics by Algorithm Family:")
        print(performance_stats)
        
        # Statistical distribution tests
        algorithms = self.df['algorithm_family'].unique()
        distribution_tests = {}
        
        for algo in algorithms:
            algo_data = self.df[self.df['algorithm_family'] == algo]['accuracy'].dropna()
            if len(algo_data) > 3:
                # Shapiro-Wilk test for normality
                shapiro_stat, shapiro_p = stats.shapiro(algo_data)
                distribution_tests[algo] = {
                    'normality_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
        
        self.results['performance_distribution'] = {
            'stats': performance_stats,
            'distribution_tests': distribution_tests
        }
        
        return performance_stats
    
    def experiment_2_correlation_analysis(self):
        """Experiment 2: Correlation Analysis"""
        print("\n=== Experiment 2: Correlation Analysis ===")
        
        # Select numerical columns for correlation
        numerical_cols = ['accuracy', 'f1_score', 'map_score', 'speed_ms', 
                         'cycle_time', 'success_rate', 'model_size', 'year']
        
        # Calculate correlation matrix
        corr_data = self.df[numerical_cols].corr()
        
        # Calculate p-values for correlations
        n = len(self.df)
        corr_p_values = pd.DataFrame(index=corr_data.index, columns=corr_data.columns)
        
        for i, col1 in enumerate(corr_data.columns):
            for j, col2 in enumerate(corr_data.columns):
                if i != j:
                    data1 = self.df[col1].dropna()
                    data2 = self.df[col2].dropna()
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 3:
                        corr, p_val = stats.pearsonr(data1[common_idx], data2[common_idx])
                        corr_p_values.loc[col1, col2] = p_val
                    else:
                        corr_p_values.loc[col1, col2] = np.nan
                else:
                    corr_p_values.loc[col1, col2] = 0.0
        
        # Principal Component Analysis
        pca_data = self.df[numerical_cols].dropna()
        if len(pca_data) > 5:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data)
            
            pca = PCA()
            pca_transformed = pca.fit_transform(scaled_data)
            
            pca_results = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_
            }
        else:
            pca_results = None
        
        self.results['correlation_analysis'] = {
            'correlation_matrix': corr_data,
            'p_values': corr_p_values,
            'pca_results': pca_results
        }
        
        print("Significant correlations (p < 0.05):")
        significant_corrs = []
        for i, col1 in enumerate(corr_data.columns):
            for j, col2 in enumerate(corr_data.columns):
                if i < j and not pd.isna(corr_p_values.loc[col1, col2]):
                    if corr_p_values.loc[col1, col2] < 0.05:
                        corr_val = corr_data.loc[col1, col2]
                        significant_corrs.append((col1, col2, corr_val, corr_p_values.loc[col1, col2]))
        
        for col1, col2, corr, p_val in significant_corrs:
            print(f"  {col1} ↔ {col2}: r={corr:.3f}, p={p_val:.3f}")
        
        return corr_data
    
    def experiment_3_temporal_trends(self):
        """Experiment 3: Temporal Trend Analysis"""
        print("\n=== Experiment 3: Temporal Trend Analysis ===")
        
        # Yearly performance aggregation
        yearly_stats = self.df.groupby('year').agg({
            'accuracy': ['mean', 'std', 'count'],
            'speed_ms': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std', 'count']
        })
        
        # Trend analysis for accuracy
        years = yearly_stats.index.values
        accuracy_means = yearly_stats[('accuracy', 'mean')].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, accuracy_means)
        
        # Polynomial fitting
        poly_coeffs = np.polyfit(years, accuracy_means, 2)
        
        # Change point detection (simple method)
        change_points = []
        for i in range(1, len(accuracy_means)-1):
            if abs(accuracy_means[i] - accuracy_means[i-1]) > 2*np.std(accuracy_means):
                change_points.append(years[i])
        
        # Future projections
        future_years = np.array([2025, 2026, 2027])
        linear_projection = slope * future_years + intercept
        poly_projection = np.polyval(poly_coeffs, future_years)
        
        self.results['temporal_trends'] = {
            'yearly_stats': yearly_stats,
            'linear_trend': {'slope': slope, 'intercept': intercept, 'r_squared': r_value**2, 'p_value': p_value},
            'poly_coeffs': poly_coeffs,
            'change_points': change_points,
            'projections': {
                'linear': linear_projection,
                'polynomial': poly_projection
            }
        }
        
        print(f"Accuracy improvement rate: {slope:.4f} per year (R²={r_value**2:.3f}, p={p_value:.3f})")
        print(f"Projected 2025 accuracy: {linear_projection[0]:.2f}% (linear)")
        
        return yearly_stats
    
    def experiment_4_environmental_impact(self):
        """Experiment 4: Environmental Impact Assessment"""
        print("\n=== Experiment 4: Environmental Impact Assessment ===")
        
        # ANOVA for environmental differences
        environments = self.df['environment'].unique()
        env_groups = [self.df[self.df['environment'] == env]['accuracy'].dropna() 
                     for env in environments]
        
        # Remove empty groups
        env_groups = [group for group in env_groups if len(group) > 0]
        environments = [env for i, env in enumerate(environments) 
                       if len(self.df[self.df['environment'] == env]['accuracy'].dropna()) > 0]
        
        if len(env_groups) > 1:
            f_stat, p_value = stats.f_oneway(*env_groups)
            
            # Post-hoc pairwise comparisons
            pairwise_results = {}
            for i, env1 in enumerate(environments):
                for j, env2 in enumerate(environments):
                    if i < j:
                        group1 = self.df[self.df['environment'] == env1]['accuracy'].dropna()
                        group2 = self.df[self.df['environment'] == env2]['accuracy'].dropna()
                        if len(group1) > 0 and len(group2) > 0:
                            t_stat, t_p = stats.ttest_ind(group1, group2)
                            effect_size = (group1.mean() - group2.mean()) / np.sqrt(
                                ((len(group1)-1)*group1.var() + (len(group2)-1)*group2.var()) / 
                                (len(group1) + len(group2) - 2)
                            )
                            pairwise_results[f"{env1}_vs_{env2}"] = {
                                't_stat': t_stat, 'p_value': t_p, 'effect_size': effect_size
                            }
        else:
            f_stat, p_value = None, None
            pairwise_results = {}
        
        # Environmental performance matrix
        env_performance = self.df.groupby(['algorithm_family', 'environment'])['accuracy'].agg(['mean', 'std', 'count'])
        
        self.results['environmental_impact'] = {
            'anova': {'f_stat': f_stat, 'p_value': p_value},
            'pairwise_comparisons': pairwise_results,
            'performance_matrix': env_performance
        }
        
        if f_stat is not None:
            print(f"Environmental ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
        print("Environmental impact on accuracy:")
        env_means = self.df.groupby('environment')['accuracy'].mean().sort_values(ascending=False)
        for env, mean_acc in env_means.items():
            print(f"  {env}: {mean_acc:.2f}% average accuracy")
        
        return env_performance
    
    def experiment_5_adoption_patterns(self):
        """Experiment 5: Technology Adoption Patterns"""
        print("\n=== Experiment 5: Technology Adoption Patterns ===")
        
        # Algorithm adoption over time
        adoption_matrix = pd.crosstab(self.df['year'], self.df['algorithm_family'])
        adoption_percentages = adoption_matrix.div(adoption_matrix.sum(axis=1), axis=0) * 100
        
        # Calculate adoption growth rates
        growth_rates = {}
        for algo in adoption_matrix.columns:
            yearly_counts = adoption_matrix[algo].values
            years = adoption_matrix.index.values
            
            # Simple growth rate calculation
            if len(yearly_counts) > 1:
                growth_rate = (yearly_counts[-1] - yearly_counts[0]) / len(yearly_counts)
                growth_rates[algo] = growth_rate
        
        # Algorithm lifecycle analysis
        lifecycle_stages = {}
        for algo in adoption_matrix.columns:
            counts = adoption_matrix[algo].values
            total_papers = counts.sum()
            peak_year = adoption_matrix.index[np.argmax(counts)]
            
            if total_papers > 0:
                lifecycle_stages[algo] = {
                    'total_papers': total_papers,
                    'peak_year': peak_year,
                    'current_trend': 'growing' if counts[-1] > counts[-2] else 'declining'
                }
        
        self.results['adoption_patterns'] = {
            'adoption_matrix': adoption_matrix,
            'adoption_percentages': adoption_percentages,
            'growth_rates': growth_rates,
            'lifecycle_stages': lifecycle_stages
        }
        
        print("Algorithm adoption trends:")
        for algo, rate in growth_rates.items():
            print(f"  {algo}: {rate:+.1f} papers/year growth rate")
        
        return adoption_matrix
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n=== META-ANALYSIS SUMMARY REPORT ===")
        
        # Overall statistics
        total_studies = len(self.df)
        year_span = self.df['year'].max() - self.df['year'].min() + 1
        algorithm_count = self.df['algorithm_family'].nunique()
        
        print(f"Dataset Overview:")
        print(f"  Total studies: {total_studies}")
        print(f"  Year span: {year_span} years ({self.df['year'].min()}-{self.df['year'].max()})")
        print(f"  Algorithm families: {algorithm_count}")
        print(f"  Fruit types: {self.df['fruit_type'].nunique()}")
        print(f"  Environments: {self.df['environment'].nunique()}")
        
        # Performance overview
        acc_mean = self.df['accuracy'].mean()
        acc_std = self.df['accuracy'].std()
        speed_mean = self.df['speed_ms'].mean()
        speed_std = self.df['speed_ms'].std()
        
        print(f"\nPerformance Overview:")
        print(f"  Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
        print(f"  Processing speed: {speed_mean:.1f} ± {speed_std:.1f} ms/image")
        
        # Key findings
        print(f"\nKey Statistical Findings:")
        if 'temporal_trends' in self.results:
            trend = self.results['temporal_trends']['linear_trend']
            print(f"  • Annual accuracy improvement: {trend['slope']:.3f}% per year (p={trend['p_value']:.3f})")
        
        if 'environmental_impact' in self.results:
            env_anova = self.results['environmental_impact']['anova']
            if env_anova['p_value'] is not None:
                print(f"  • Environmental impact significant: p={env_anova['p_value']:.3f}")
        
        # Save results
        with open('meta_analysis_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    json_results[key] = self._convert_to_json_serializable(value)
                else:
                    json_results[key] = str(value)
            
            import json
            json.dump(json_results, f, indent=2)
        
        return self.results
    
    def _convert_to_json_serializable(self, obj):
        """Convert pandas/numpy objects to JSON serializable format"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def run_all_experiments(self):
        """Run all statistical experiments"""
        if not self.load_data():
            return None
        
        # Run all experiments
        self.experiment_1_performance_distribution()
        self.experiment_2_correlation_analysis()
        self.experiment_3_temporal_trends()
        self.experiment_4_environmental_impact()
        self.experiment_5_adoption_patterns()
        
        # Generate summary
        return self.generate_summary_report()

def main():
    """Main function to run meta-analysis experiments"""
    print("Starting Meta-Analysis Statistical Experiments...")
    
    # Initialize analyzer
    analyzer = MetaAnalysisExperiments('fruit_picking_literature_data.csv')
    
    # Run all experiments
    results = analyzer.run_all_experiments()
    
    print("\n=== EXPERIMENTS COMPLETED ===")
    print("Results saved to: meta_analysis_results.json")
    print("Ready for high-order figure generation!")
    
    return results

if __name__ == "__main__":
    results = main()