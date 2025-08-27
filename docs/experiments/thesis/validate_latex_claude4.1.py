#!/usr/bin/env python3
"""
LaTeX Document Validator
Checks for common LaTeX syntax errors without compilation
"""

import re
import sys
from pathlib import Path

def validate_latex(filepath):
    """Validate LaTeX document for common syntax errors"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    errors = []
    warnings = []
    
    # Check document structure
    if '\\documentclass' not in content:
        errors.append("Missing \\documentclass declaration")
    if '\\begin{document}' not in content:
        errors.append("Missing \\begin{document}")
    if '\\end{document}' not in content:
        errors.append("Missing \\end{document}")
    
    # Count begin/end pairs
    begins = re.findall(r'\\begin\{([^}]+)\}', content)
    ends = re.findall(r'\\end\{([^}]+)\}', content)
    
    begin_counts = {}
    end_counts = {}
    
    for env in begins:
        begin_counts[env] = begin_counts.get(env, 0) + 1
    for env in ends:
        end_counts[env] = end_counts.get(env, 0) + 1
    
    # Check for unmatched environments
    all_envs = set(begin_counts.keys()) | set(end_counts.keys())
    for env in all_envs:
        begin_count = begin_counts.get(env, 0)
        end_count = end_counts.get(env, 0)
        if begin_count != end_count:
            errors.append(f"Unmatched environment '{env}': {begin_count} \\begin vs {end_count} \\end")
    
    # Check bracket matching
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces != close_braces:
        warnings.append(f"Brace mismatch: {open_braces} {{ vs {close_braces} }}")
    
    open_brackets = content.count('[')
    close_brackets = content.count(']')
    if open_brackets != close_brackets:
        warnings.append(f"Bracket mismatch: {open_brackets} [ vs {close_brackets} ]")
    
    # Check for common LaTeX errors
    math_dollar_single = len(re.findall(r'(?<!\$)\$(?!\$)', content))
    if math_dollar_single % 2 != 0:
        errors.append("Unmatched single $ for inline math")
    
    math_dollar_double = len(re.findall(r'\$\$', content))
    if math_dollar_double % 2 != 0:
        errors.append("Unmatched $$ for display math")
    
    # Check citations and references
    citations = re.findall(r'\\cite\{([^}]+)\}', content)
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    refs = re.findall(r'\\ref\{([^}]+)\}', content)
    
    # Check if refs point to existing labels
    for ref in refs:
        if ref not in labels:
            warnings.append(f"Reference '\\ref{{{ref}}}' has no corresponding \\label")
    
    # Check table/figure environments
    tables = re.findall(r'\\begin\{table\}', content)
    figures = re.findall(r'\\begin\{figure\}', content)
    
    print(f"LaTeX Document Validation Report for: {filepath}")
    print("=" * 60)
    
    # Summary statistics
    print(f"\n📊 Document Statistics:")
    print(f"  - Lines: {len(lines)}")
    print(f"  - Characters: {len(content)}")
    print(f"  - Tables: {len(tables)}")
    print(f"  - Figures: {len(figures)}")
    print(f"  - Citations: {len(citations)}")
    print(f"  - Labels: {len(labels)}")
    print(f"  - References: {len(refs)}")
    print(f"  - Environments: {len(all_envs)} unique types")
    
    # Report errors
    if errors:
        print(f"\n❌ Errors found ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ No critical errors found!")
    
    # Report warnings
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ No warnings!")
    
    # Check specific LaTeX packages
    packages = re.findall(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}', content)
    print(f"\n📦 LaTeX Packages Used ({len(packages)}):")
    for pkg in packages:
        print(f"  - {pkg}")
    
    # Check for required elements for compilation
    print(f"\n🔍 Compilation Readiness Check:")
    checks = {
        "Document class": '\\documentclass' in content,
        "Begin document": '\\begin{document}' in content,
        "End document": '\\end{document}' in content,
        "Title defined": '\\title' in content,
        "Author defined": '\\author' in content,
        "Abstract present": '\\begin{abstract}' in content,
        "At least one section": '\\section' in content,
        "Bibliography style": '\\bibliographystyle' in content or '\\bibliography' in content
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    # Overall assessment
    print(f"\n📋 Overall Assessment:")
    if not errors and not warnings:
        print("  ✅ Document appears to be valid LaTeX!")
        print("  Ready for compilation with pdflatex/xelatex/lualatex")
    elif not errors:
        print("  ⚠️  Document has warnings but should compile")
        print("  Review warnings for potential issues")
    else:
        print("  ❌ Document has errors that need to be fixed")
        print("  Fix critical errors before attempting compilation")
    
    return len(errors) == 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "exp1_comprehensive_analysis_claude4.1.tex"
    
    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found!")
        sys.exit(1)
    
    is_valid = validate_latex(filepath)
    sys.exit(0 if is_valid else 1)