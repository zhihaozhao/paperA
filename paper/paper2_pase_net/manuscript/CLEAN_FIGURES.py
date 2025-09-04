#!/usr/bin/env python3
"""
Clean figure folders for paper submission
Remove draft figures and keep only those actually used in the paper
"""

import os
import shutil
from pathlib import Path

def clean_paper2_figures():
    """Clean Paper 2 (PASE-Net) figure folder"""
    
    plots_dir = Path("plots")
    if not plots_dir.exists():
        print("❌ plots/ directory not found")
        return
    
    # Figures actually used in the paper (from enhanced_claude_v1.tex)
    used_figures = {
        "fig1_system_architecture.pdf",
        "fig2_physics_modeling_new.pdf",  
        "fig3_cross_domain.pdf",
        "fig4_calibration.pdf",
        "fig5_label_efficiency.pdf",
        "fig6_interpretability.pdf",
        # Commented out but keep for supplementary materials
        "d5_progressive_enhanced.pdf",
        "ablation_noise_env_heatmap.pdf", 
        "ablation_components.pdf"
    }
    
    # Essential scripts to keep
    essential_scripts = {
        "scr1_system_architecture.py",
        "scr2_physics_modeling.py",
        "scr3_cross_domain.py",
        "scr4_calibration.py", 
        "scr5_label_efficiency.py",
        "scr6_interpretability.py"
    }
    
    # Create backup directory
    backup_dir = Path("plots_backup")
    if not backup_dir.exists():
        backup_dir.mkdir()
        print(f"✅ Created backup directory: {backup_dir}")
    
    # Statistics
    total_files = 0
    kept_files = 0
    moved_files = 0
    
    # Process all files in plots directory
    for file in plots_dir.iterdir():
        if file.is_file():
            total_files += 1
            filename = file.name
            
            # Keep used figures and essential scripts
            if filename in used_figures or filename in essential_scripts:
                print(f"✅ KEEP: {filename}")
                kept_files += 1
            else:
                # Move to backup
                backup_path = backup_dir / filename
                shutil.move(str(file), str(backup_path))
                print(f"📦 BACKUP: {filename} -> plots_backup/")
                moved_files += 1
    
    # Summary
    print("\n" + "="*50)
    print("📊 CLEANING SUMMARY FOR PAPER 2 (PASE-Net)")
    print("="*50)
    print(f"Total files processed: {total_files}")
    print(f"Files kept (used in paper): {kept_files}")
    print(f"Files moved to backup: {moved_files}")
    print(f"\n✅ Plots folder now contains only submission-ready figures!")
    
    # Create manifest
    manifest_path = Path("FIGURE_MANIFEST.md")
    with open(manifest_path, "w") as f:
        f.write("# Figure Manifest for Paper 2 (PASE-Net)\n\n")
        f.write("## Figures Used in Main Paper\n\n")
        f.write("| Figure | File | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| Fig. 1 | `fig1_system_architecture.pdf` | PASE-Net architecture |\n")
        f.write("| Fig. 2 | `fig2_physics_modeling_new.pdf` | Physics-informed synthesis |\n")
        f.write("| Fig. 3 | `fig3_cross_domain.pdf` | Cross-domain performance |\n")
        f.write("| Fig. 4 | `fig4_calibration.pdf` | Calibration analysis |\n")
        f.write("| Fig. 5 | `fig5_label_efficiency.pdf` | Label efficiency |\n")
        f.write("| Fig. 6 | `fig6_interpretability.pdf` | Interpretability analysis |\n")
        f.write("\n## Figures for Supplementary Materials\n\n")
        f.write("| Figure | File | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| Supp. 1 | `d5_progressive_enhanced.pdf` | Progressive temporal |\n")
        f.write("| Supp. 2 | `ablation_noise_env_claude4.pdf` | Nuisance factors |\n")
        f.write("| Supp. 3 | `ablation_components.pdf` | Component analysis |\n")
        f.write("\n## Generation Scripts\n\n")
        for script in essential_scripts:
            f.write(f"- `{script}`\n")
        f.write("\n## Backup Location\n\n")
        f.write("Draft and unused figures: `plots_backup/`\n")
    
    print(f"\n📄 Created figure manifest: {manifest_path}")

def verify_latex_references():
    """Verify all figure references in LaTeX are available"""
    
    tex_file = Path("enhanced_claude_v1.tex")
    if not tex_file.exists():
        print("❌ LaTeX file not found")
        return
        
    plots_dir = Path("plots")
    available_figures = set()
    
    if plots_dir.exists():
        available_figures = {f.name for f in plots_dir.glob("*.pdf")}
    
    print("\n" + "="*50)
    print("🔍 VERIFYING LATEX FIGURE REFERENCES")
    print("="*50)
    
    # Extract figure references from LaTeX
    import re
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Find all includegraphics commands
    pattern = r'\\includegraphics.*?\{plots/(.*?)\}'
    references = re.findall(pattern, content)
    
    # Check commented out figures too
    pattern_commented = r'%\\includegraphics.*?\{plots/(.*?)\}'
    commented_refs = re.findall(pattern_commented, content)
    
    print("\n📊 Active Figure References:")
    all_good = True
    for ref in references:
        if ref in available_figures:
            print(f"  ✅ {ref}")
        else:
            print(f"  ❌ {ref} - NOT FOUND!")
            all_good = False
    
    if commented_refs:
        print("\n📝 Commented Figure References (for supplementary):")
        for ref in commented_refs:
            if ref in available_figures:
                print(f"  ✅ {ref} (available)")
            else:
                print(f"  ⚠️  {ref} (not in plots/)")
    
    if all_good:
        print("\n✅ All active figure references are valid!")
    else:
        print("\n⚠️  Some figure references need attention!")
    
    return all_good

if __name__ == "__main__":
    print("🧹 CLEANING FIGURE FOLDERS FOR PAPER SUBMISSION")
    print("="*60)
    
    # Change to manuscript directory
    manuscript_dir = Path("/workspace/paper/paper2_pase_net/manuscript")
    if manuscript_dir.exists():
        os.chdir(manuscript_dir)
        print(f"📁 Working directory: {manuscript_dir}")
        
        # Clean figures
        clean_paper2_figures()
        
        # Verify references
        verify_latex_references()
    else:
        print("❌ Manuscript directory not found!")
    
    print("\n✅ CLEANING COMPLETE!")
    print("Ready for journal submission with clean, organized figures.")