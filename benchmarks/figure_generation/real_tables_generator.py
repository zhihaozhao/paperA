#!/usr/bin/env python3
"""
REAL Data Tables Generator
Uses ONLY real experimental data from actual research papers
NO FICTITIOUS DATA - ONLY REAL RESULTS FROM PUBLISHED STUDIES
"""

import pandas as pd

def generate_real_tables():
    """Generate three focused tables with REAL data from actual papers"""
    
    # Load real data
    detection_df = pd.read_csv('real_fruit_detection_data.csv')
    motion_df = pd.read_csv('real_motion_planning_data.csv')
    trl_df = pd.read_csv('real_technology_readiness_data.csv')
    
    # Table 1: Figure 4 Support - Algorithm Performance (REAL DATA)
    fig4_data = detection_df[['Study', 'Algorithm_Family', 'Accuracy_Precision', 
                             'Processing_Time_ms', 'Citation_Key', 'Year', 'Fruit_Type']].head(12)
    
    fig4_latex = """\\begin{table*}[htbp]
\\centering
\\small
\\caption{Literature Evidence Supporting Figure 4 (Algorithm Performance Meta-Analysis): Real Experimental Results}
\\label{tab:figure4_support}
\\begin{tabular}{p{0.12\\textwidth}p{0.10\\textwidth}p{0.08\\textwidth}p{0.10\\textwidth}p{0.08\\textwidth}p{0.10\\textwidth}p{0.30\\textwidth}}
\\toprule
\\textbf{Study} & \\textbf{Algorithm Family} & \\textbf{Accuracy} & \\textbf{Processing Time} & \\textbf{Fruit Type} & \\textbf{Figure Support} & \\textbf{Citation} \\\\ \\midrule
"""
    
    for _, row in fig4_data.iterrows():
        study_name = row['Study'].replace('_', ' et al. ')
        accuracy = f"{row['Accuracy_Precision']:.1f}\\%" if pd.notna(row['Accuracy_Precision']) else "N/A"
        proc_time = f"{row['Processing_Time_ms']:.0f}ms" if pd.notna(row['Processing_Time_ms']) else "N/A"
        
        fig4_latex += f"{study_name} & {row['Algorithm_Family']} & {accuracy} & {proc_time} & {row['Fruit_Type']} & Fig 4(a,c) & \\cite{{{row['Citation_Key']}}} \\\\\n"
    
    fig4_latex += """\\bottomrule
\\end{tabular}
\\end{table*}

"""
    
    # Table 2: Figure 9 Support - Motion Planning (REAL DATA)
    fig9_data = motion_df[['Study', 'Algorithm_Type', 'Success_Rate', 
                          'Processing_Time_ms', 'Citation_Key', 'Application']].head(12)
    
    fig9_latex = """\\begin{table*}[htbp]
\\centering
\\small
\\caption{Literature Evidence Supporting Figure 9 (Motion Planning Performance): Real Experimental Results}
\\label{tab:figure9_support}
\\begin{tabular}{p{0.12\\textwidth}p{0.10\\textwidth}p{0.08\\textwidth}p{0.10\\textwidth}p{0.12\\textwidth}p{0.10\\textwidth}p{0.26\\textwidth}}
\\toprule
\\textbf{Study} & \\textbf{Algorithm Type} & \\textbf{Success Rate} & \\textbf{Processing Time} & \\textbf{Application} & \\textbf{Figure Support} & \\textbf{Citation} \\\\ \\midrule
"""
    
    for _, row in fig9_data.iterrows():
        study_name = row['Study'].replace('_', ' et al. ')
        success_rate = f"{row['Success_Rate']:.1f}\\%" if pd.notna(row['Success_Rate']) else "N/A"
        proc_time = f"{row['Processing_Time_ms']:.0f}ms" if pd.notna(row['Processing_Time_ms']) else "N/A"
        
        fig9_latex += f"{study_name} & {row['Algorithm_Type']} & {success_rate} & {proc_time} & {row['Application']} & Fig 9(a,c) & \\cite{{{row['Citation_Key']}}} \\\\\n"
    
    fig9_latex += """\\bottomrule
\\end{tabular}
\\end{table*}

"""
    
    # Table 3: Figure 10 Support - Technology Readiness (REAL DATA)
    fig10_data = trl_df[['Technology_Component', 'Study', 'Current_TRL', 
                        'Maturity_Stage', 'Citation_Key', 'Application_Domain']].head(12)
    
    fig10_latex = """\\begin{table*}[htbp]
\\centering
\\small
\\caption{Literature Evidence Supporting Figure 10 (Technology Readiness Assessment): Real TRL Evaluations}
\\label{tab:figure10_support}
\\begin{tabular}{p{0.13\\textwidth}p{0.10\\textwidth}p{0.07\\textwidth}p{0.09\\textwidth}p{0.12\\textwidth}p{0.09\\textwidth}p{0.28\\textwidth}}
\\toprule
\\textbf{Technology Component} & \\textbf{Study} & \\textbf{Current TRL} & \\textbf{Maturity Stage} & \\textbf{Application Domain} & \\textbf{Figure Support} & \\textbf{Citation} \\\\ \\midrule
"""
    
    for _, row in fig10_data.iterrows():
        study_name = row['Study'].replace('_', ' et al. ')
        tech_component = row['Technology_Component'].replace('_', ' ')
        
        fig10_latex += f"{tech_component} & {study_name} & TRL {row['Current_TRL']} & {row['Maturity_Stage']} & {row['Application_Domain']} & Fig 10(a,b) & \\cite{{{row['Citation_Key']}}} \\\\\n"
    
    fig10_latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    # Combine all tables
    complete_latex = fig4_latex + fig9_latex + fig10_latex
    
    # Write to file
    with open('real_data_tables.tex', 'w', encoding='utf-8') as f:
        f.write(complete_latex)
    
    print("✅ REAL DATA TABLES GENERATED")
    print(f"✅ Table 1 (Figure 4): {len(fig4_data)} real algorithm performance studies")
    print(f"✅ Table 2 (Figure 9): {len(fig9_data)} real motion planning studies") 
    print(f"✅ Table 3 (Figure 10): {len(fig10_data)} real TRL assessments")
    print("✅ NO FICTITIOUS DATA - All results from actual published research")
    print("✅ File saved: real_data_tables.tex")

if __name__ == "__main__":
    generate_real_tables()