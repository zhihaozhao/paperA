#!/usr/bin/env python3
"""
Clean up the paper/enhanced directory by moving auxiliary files to paper_supplementary
while keeping only essential paper files.
"""

import os
import shutil
from pathlib import Path

def cleanup_enhanced_directory():
    """Move auxiliary files from paper/enhanced to paper_supplementary"""
    
    enhanced_dir = Path("/workspace/paper/enhanced")
    supplementary_dir = Path("/workspace/paper_supplementary")
    
    # Files to keep in enhanced directory (essential for paper)
    keep_files = {
        'enhanced_claude_v1.tex',
        'enhanced_refs.bib',
        'SUPPLEMENTARY_MATERIALS.tex',
        'IEEEtran.cls',
        'IEEEtranS.bst',
    }
    
    # Move patterns
    move_patterns = {
        '*_REPORT.md': 'docs/analysis_reports',
        '*_ANALYSIS*.md': 'docs/analysis_reports',
        '*_SUMMARY.md': 'docs/analysis_reports',
        '*_PLAN.md': 'docs/analysis_reports',
        '*_NOTE.md': 'docs/analysis_reports',
        '*_cover_letter*.md': 'docs/submission',
        '*.py': 'scripts/archive',
        'extract_*.py': 'scripts/archive',
        'analyze_*.py': 'scripts/archive',
        'create_*.py': 'scripts/archive',
        'verify_*.py': 'scripts/archive',
        'SOLUTION*.md': 'docs/analysis_reports',
        'FIGURE*.md': 'docs/analysis_reports',
        'DATA*.md': 'docs/analysis_reports',
        'MINIMAL*.md': 'docs/analysis_reports',
        'REAL*.md': 'docs/analysis_reports',
        'COMPREHENSIVE*.md': 'docs/analysis_reports',
    }
    
    moved_files = []
    kept_files = []
    
    print("="*60)
    print("CLEANING UP paper/enhanced DIRECTORY")
    print("="*60)
    
    # Process each file in enhanced directory
    for file_path in enhanced_dir.glob('*'):
        if file_path.is_file():
            file_name = file_path.name
            
            # Check if file should be kept
            if file_name in keep_files:
                kept_files.append(file_name)
                print(f"✅ Keeping: {file_name}")
                continue
            
            # Check if file matches move patterns
            moved = False
            for pattern, target_dir in move_patterns.items():
                if file_path.match(pattern):
                    target = supplementary_dir / target_dir
                    target.mkdir(parents=True, exist_ok=True)
                    
                    new_path = target / file_name
                    if new_path.exists():
                        print(f"⚠️  Already exists: {target_dir}/{file_name}")
                    else:
                        shutil.move(str(file_path), str(new_path))
                        moved_files.append(f"{target_dir}/{file_name}")
                        print(f"📦 Moved: {file_name} → {target_dir}/")
                    moved = True
                    break
            
            if not moved and file_name.endswith(('.md', '.py', '.txt')):
                # Move other documentation/script files to archive
                target = supplementary_dir / 'docs' / 'archive'
                target.mkdir(parents=True, exist_ok=True)
                new_path = target / file_name
                
                if new_path.exists():
                    print(f"⚠️  Already exists: docs/archive/{file_name}")
                else:
                    shutil.move(str(file_path), str(new_path))
                    moved_files.append(f"docs/archive/{file_name}")
                    print(f"📦 Moved: {file_name} → docs/archive/")
    
    # Clean up plots directory (keep only essential scripts)
    plots_dir = enhanced_dir / 'plots'
    if plots_dir.exists():
        plots_keep = {
            'scr1_system_architecture.py',
            'scr2_physics_modeling.py',
            'scr3_cross_domain.py',
            'scr4_calibration.py',
            'scr5_label_efficiency.py',
            'scr6_interpretability.py',
        }
        
        for file_path in plots_dir.glob('*'):
            if file_path.is_file():
                if file_path.name not in plots_keep and not file_path.name.endswith('.pdf'):
                    # Move auxiliary plot scripts
                    target = supplementary_dir / 'scripts' / 'figure_generation' / 'archive'
                    target.mkdir(parents=True, exist_ok=True)
                    new_path = target / file_path.name
                    
                    if not new_path.exists():
                        shutil.move(str(file_path), str(new_path))
                        print(f"📦 Moved: plots/{file_path.name} → scripts/figure_generation/archive/")
    
    # Summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"Files kept in enhanced/: {len(kept_files)}")
    print(f"Files moved to supplementary/: {len(moved_files)}")
    
    print("\nEssential files remaining in paper/enhanced/:")
    for f in kept_files:
        print(f"  ✅ {f}")
    
    print("\nThe enhanced directory is now clean and ready for submission!")
    
    return kept_files, moved_files

if __name__ == "__main__":
    cleanup_enhanced_directory()