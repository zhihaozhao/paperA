#!/usr/bin/env python3
"""
Master Script for Complete Literature Meta-Analysis Pipeline
Runs data extraction, statistical experiments, and figure generation

Author: Research Team
Date: 2024
Purpose: Complete automated pipeline for literature meta-analysis
"""

import sys
import os
from pathlib import Path
import subprocess
import json

# Add script directories to path
benchmarks_dir = Path(__file__).parent
sys.path.append(str(benchmarks_dir / 'data_extraction'))
sys.path.append(str(benchmarks_dir / 'statistical_analysis'))
sys.path.append(str(benchmarks_dir / 'figure_generation'))

class CompletePipeline:
    """Master pipeline for complete meta-analysis"""
    
    def __init__(self, tex_file_path: str, output_dir: str = 'output'):
        self.tex_file_path = tex_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Intermediate files
        self.data_file = self.output_dir / 'fruit_picking_literature_data.csv'
        self.results_file = self.output_dir / 'meta_analysis_results.json'
        
    def step_1_extract_data(self):
        """Step 1: Extract quantitative data from LaTeX tables"""
        print("=" * 60)
        print("STEP 1: DATA EXTRACTION")
        print("=" * 60)
        
        try:
            from literature_data_extractor import LiteratureDataExtractor
            
            # Initialize extractor
            extractor = LiteratureDataExtractor(self.tex_file_path)
            
            # Extract data
            data = extractor.extract_all_data()
            
            # Save to output directory
            output_path = str(self.data_file)
            extractor.save_data(output_path)
            
            # Print summary
            summary = extractor.get_data_summary()
            print(f"\n✅ DATA EXTRACTION COMPLETED")
            print(f"   📊 Extracted: {summary.get('studies_count', 0)} studies")
            print(f"   📈 Time span: {len(summary.get('year_distribution', {}))} years")
            print(f"   🤖 Algorithms: {len(summary.get('algorithm_distribution', {}))} families")
            print(f"   💾 Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ DATA EXTRACTION FAILED: {e}")
            return False
    
    def step_2_statistical_analysis(self):
        """Step 2: Perform statistical experiments"""
        print("\n" + "=" * 60)
        print("STEP 2: STATISTICAL ANALYSIS")
        print("=" * 60)
        
        try:
            from meta_analysis_experiments import MetaAnalysisExperiments
            
            # Initialize analyzer
            analyzer = MetaAnalysisExperiments(str(self.data_file))
            
            # Run all experiments
            results = analyzer.run_all_experiments()
            
            # Move results to output directory
            if os.path.exists('meta_analysis_results.json'):
                os.rename('meta_analysis_results.json', str(self.results_file))
            
            print(f"\n✅ STATISTICAL ANALYSIS COMPLETED")
            print(f"   📊 5 experiments performed")
            print(f"   📈 Statistical tests completed")
            print(f"   💾 Results saved to: {self.results_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ STATISTICAL ANALYSIS FAILED: {e}")
            return False
    
    def step_3_generate_figures(self):
        """Step 3: Generate high-order figures"""
        print("\n" + "=" * 60)
        print("STEP 3: FIGURE GENERATION")
        print("=" * 60)
        
        try:
            from create_meta_analysis_figures import MetaAnalysisFigureGenerator
            
            # Initialize generator
            generator = MetaAnalysisFigureGenerator(str(self.data_file), str(self.results_file))
            
            # Generate all figures
            success = generator.generate_all_figures()
            
            if success:
                # Move figures to output directory
                figure_files = [
                    'fig_meta_analysis_visual_detection.png',
                    'fig_meta_analysis_visual_detection.pdf',
                    'fig_motion_control_analysis.png', 
                    'fig_motion_control_analysis.pdf',
                    'fig_technology_roadmap.png',
                    'fig_technology_roadmap.pdf',
                    'fig_comprehensive_dashboard.png',
                    'fig_comprehensive_dashboard.pdf'
                ]
                
                moved_count = 0
                for fig_file in figure_files:
                    if os.path.exists(fig_file):
                        os.rename(fig_file, str(self.output_dir / fig_file))
                        moved_count += 1
                
                print(f"\n✅ FIGURE GENERATION COMPLETED")
                print(f"   🎨 Generated: {moved_count} high-order figures")
                print(f"   📁 Saved to: {self.output_dir}/")
                print(f"   📊 Ready for LaTeX integration")
                
                return True
            else:
                print("❌ FIGURE GENERATION FAILED")
                return False
                
        except Exception as e:
            print(f"❌ FIGURE GENERATION FAILED: {e}")
            return False
    
    def step_4_matlab_analysis(self):
        """Step 4: Run MATLAB analysis (optional)"""
        print("\n" + "=" * 60)
        print("STEP 4: MATLAB ANALYSIS (OPTIONAL)")
        print("=" * 60)
        
        matlab_script = benchmarks_dir / 'figure_generation' / 'motion_control_analysis.m'
        
        if matlab_script.exists():
            print("MATLAB script available for additional analysis:")
            print(f"  📄 Script: {matlab_script}")
            print(f"  🚀 Run: matlab -batch \"cd('{benchmarks_dir}'); motion_control_analysis\"")
            print("  ⚠️  Requires MATLAB with Statistics Toolbox")
        
        return True
    
    def step_5_integration_report(self):
        """Step 5: Generate integration report"""
        print("\n" + "=" * 60)
        print("STEP 5: INTEGRATION REPORT")
        print("=" * 60)
        
        report_content = f"""# Meta-Analysis Pipeline Execution Report

## 📊 Data Processing Summary
- **Source file**: {self.tex_file_path}
- **Output directory**: {self.output_dir}
- **Execution time**: {pd.Timestamp.now()}

## 📈 Generated Files

### Data Files:
- `fruit_picking_literature_data.csv` - Extracted quantitative data
- `meta_analysis_results.json` - Statistical analysis results

### High-Order Figures:
- `fig_meta_analysis_visual_detection.png/.pdf` - Visual detection meta-analysis
- `fig_motion_control_analysis.png/.pdf` - Motion control statistical analysis  
- `fig_technology_roadmap.png/.pdf` - Technology evolution & projections
- `fig_comprehensive_dashboard.png/.pdf` - Complete meta-analysis dashboard

### Scripts:
- `literature_data_extractor.py` - Data extraction from LaTeX tables
- `meta_analysis_experiments.py` - Statistical experiments
- `create_meta_analysis_figures.py` - High-order figure generation
- `motion_control_analysis.m` - MATLAB supplementary analysis

## 🎯 LaTeX Integration

### For SN Applied Sciences (Single-column):
```latex
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{fig_meta_analysis_visual_detection.png}}
\\caption{{Visual detection performance meta-analysis across 137 studies (2015-2024). 
(A) Performance evolution showing significant improvement trend. 
(B) Algorithm family comparison with statistical significance. 
(C) Speed-accuracy trade-off revealing Pareto frontier. 
(D) Environmental performance matrix.}}
\\label{{fig:meta_analysis_visual}}
\\end{{figure}}
```

### For IEEE Access/RAS (Double-column):
```latex
\\begin{{figure*}}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{{fig_meta_analysis_visual_detection.png}}
\\caption{{Comprehensive meta-analysis results...}}
\\label{{fig:meta_analysis_visual}}
\\end{{figure*}}
```

## 🚀 Next Steps:
1. Integrate figures into journal versions
2. Update table content with statistical summaries
3. Revise text to reference meta-analysis results
4. Test compilation with new figures
"""
        
        report_file = self.output_dir / 'integration_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"✅ INTEGRATION REPORT CREATED")
        print(f"   📝 Report: {report_file}")
        print(f"   🔧 Contains LaTeX integration instructions")
        
        return True
    
    def run_complete_pipeline(self):
        """Run the complete meta-analysis pipeline"""
        print("🚀 STARTING COMPLETE META-ANALYSIS PIPELINE")
        print("=" * 80)
        
        # Run all steps
        success = True
        success &= self.step_1_extract_data()
        success &= self.step_2_statistical_analysis()
        success &= self.step_3_generate_figures()
        success &= self.step_4_matlab_analysis()
        success &= self.step_5_integration_report()
        
        print("\n" + "=" * 80)
        if success:
            print("🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
            print(f"📁 All outputs saved to: {self.output_dir}")
            print("🔧 Ready for journal integration!")
        else:
            print("❌ PIPELINE EXECUTION FAILED")
            print("🔍 Check error messages above")
        
        return success

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete literature meta-analysis pipeline')
    parser.add_argument('--tex-file', default='FP_2025_SN-APPLIED-SCIENCES/FP_2025_SN-APPLIED-SCIENCES_v1.tex',
                       help='Path to LaTeX file for analysis')
    parser.add_argument('--output-dir', default='meta_analysis_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompletePipeline(args.tex_file, args.output_dir)
    
    # Run complete analysis
    success = pipeline.run_complete_pipeline()
    
    return success

if __name__ == "__main__":
    main()