#!/usr/bin/env python3
"""
Master script to generate all figures for the paper using real experimental data.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Run a figure generation script"""
    print(f"\n{'='*60}")
    print(f"Generating: {description}")
    print('='*60)
    
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_name}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent)
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully generated {description}")
            # Print key output lines
            for line in result.stdout.split('\n'):
                if 'saved' in line.lower() or 'figure' in line.lower():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ Failed to generate {description}")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception while running {script_name}: {e}")
        return False

def main():
    """Generate all figures for the paper"""
    
    print("="*60)
    print("GENERATING ALL FIGURES WITH REAL EXPERIMENTAL DATA")
    print("="*60)
    
    # Define all figure generation scripts
    figures = [
        ("scr2_physics_modeling.py", "Figure 2: Physics Modeling and SRV Performance"),
        ("scr3_cross_domain_FINAL.py", "Figure 3: Cross-Domain Performance (LOSO/LORO)"),
        ("scr4_calibration_REAL.py", "Figure 4: Calibration Performance"),
        ("scr5_label_efficiency_FINAL.py", "Figure 5: Label Efficiency (Sim2Real)"),
        ("scr6_fall_detection_FINAL.py", "Figure 6: Fall Detection Performance"),
    ]
    
    # Track success
    success_count = 0
    total_count = len(figures)
    
    # Generate each figure
    for script, description in figures:
        if run_script(script, description):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successfully generated: {success_count}/{total_count} figures")
    
    if success_count == total_count:
        print("🎉 All figures generated successfully!")
        
        # Move generated PDFs to figures directory
        output_dir = Path(__file__).parent.parent.parent / "figures" / "main"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nMoving PDFs to {output_dir}")
        for pdf in Path(__file__).parent.glob("*.pdf"):
            target = output_dir / pdf.name
            pdf.rename(target)
            print(f"   Moved: {pdf.name}")
        
        return 0
    else:
        print(f"⚠️  Some figures failed to generate. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())