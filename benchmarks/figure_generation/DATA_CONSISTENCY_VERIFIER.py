#!/usr/bin/env python3
"""
DATA CONSISTENCY VERIFIER
Comprehensive verification of data consistency across all papers
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def load_analysis_data():
    """Load the comprehensive analysis data"""
    with open('/workspace/benchmarks/figure_generation/FINAL_COMPREHENSIVE_ANALYSIS.json', 'r') as f:
        return json.load(f)

def verify_citation_consistency(papers: dict) -> Dict[str, List[str]]:
    """Verify citation keys exist in refs.bib"""
    
    print("🔍 VERIFYING CITATION CONSISTENCY...")
    
    # Load refs.bib to get all valid citations
    with open('/workspace/benchmarks/FP_2025_IEEE-ACCESS/ref.bib', 'r', encoding='utf-8') as f:
        bib_content = f.read()
    
    # Extract all citation keys from refs.bib
    valid_citations = set()
    entries = re.findall(r'@\w+\{([^,]+),', bib_content)
    for entry in entries:
        valid_citations.add(entry.strip())
    
    print(f"📚 Valid citations in refs.bib: {len(valid_citations)}")
    
    # Check consistency
    issues = {
        'missing_citations': [],
        'extra_citations': [],
        'valid_citations': []
    }
    
    for paper_key in papers.keys():
        if paper_key in valid_citations:
            issues['valid_citations'].append(paper_key)
        else:
            issues['missing_citations'].append(paper_key)
    
    # Check for citations in refs.bib not in our analysis
    analyzed_citations = set(papers.keys())
    for citation in valid_citations:
        if citation not in analyzed_citations:
            issues['extra_citations'].append(citation)
    
    print(f"✅ Valid citations found: {len(issues['valid_citations'])}")
    print(f"❌ Missing citations: {len(issues['missing_citations'])}")
    print(f"📋 Extra citations in refs.bib: {len(issues['extra_citations'])}")
    
    return issues

def verify_performance_metrics_consistency(papers: dict) -> Dict[str, any]:
    """Verify performance metrics are realistic and consistent"""
    
    print("\n📊 VERIFYING PERFORMANCE METRICS CONSISTENCY...")
    
    metrics_stats = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'mAP': [],
        'fps': [],
        'processing_time_ms': []
    }
    
    issues = {
        'unrealistic_accuracy': [],
        'unrealistic_precision': [],
        'unrealistic_recall': [],
        'unrealistic_mAP': [],
        'unrealistic_fps': [],
        'unrealistic_processing_time': [],
        'missing_metrics': [],
        'inconsistent_metrics': []
    }
    
    for paper_key, paper_data in papers.items():
        metrics = paper_data.get('performance_metrics', {})
        
        if not metrics:
            issues['missing_metrics'].append(paper_key)
            continue
        
        # Collect statistics
        for metric_name in metrics_stats.keys():
            if metric_name in metrics:
                metrics_stats[metric_name].append(metrics[metric_name])
        
        # Check for unrealistic values
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            if acc < 50 or acc > 110:  # Allowing some margin for research claims
                issues['unrealistic_accuracy'].append(f"{paper_key}: {acc}%")
        
        if 'precision' in metrics:
            prec = metrics['precision']
            if prec < 50 or prec > 110:
                issues['unrealistic_precision'].append(f"{paper_key}: {prec}%")
        
        if 'recall' in metrics:
            rec = metrics['recall']
            if rec < 50 or rec > 110:
                issues['unrealistic_recall'].append(f"{paper_key}: {rec}%")
        
        if 'mAP' in metrics:
            map_val = metrics['mAP']
            if map_val < 50 or map_val > 110:
                issues['unrealistic_mAP'].append(f"{paper_key}: {map_val}%")
        
        if 'fps' in metrics:
            fps = metrics['fps']
            if fps < 1 or fps > 1000:  # Very broad range for different systems
                issues['unrealistic_fps'].append(f"{paper_key}: {fps} FPS")
        
        if 'processing_time_ms' in metrics:
            time_ms = metrics['processing_time_ms']
            if time_ms < 1 or time_ms > 10000:  # 1ms to 10 seconds
                issues['unrealistic_processing_time'].append(f"{paper_key}: {time_ms}ms")
        
        # Check for inconsistent metrics (precision/recall vs accuracy)
        if 'accuracy' in metrics and 'precision' in metrics and 'recall' in metrics:
            acc, prec, rec = metrics['accuracy'], metrics['precision'], metrics['recall']
            # Rough consistency check: accuracy should be roughly between precision and recall
            if not (min(prec, rec) - 15 <= acc <= max(prec, rec) + 15):
                issues['inconsistent_metrics'].append(f"{paper_key}: acc={acc}%, prec={prec}%, rec={rec}%")
    
    # Calculate statistics
    stats_summary = {}
    for metric_name, values in metrics_stats.items():
        if values:
            stats_summary[metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'median': sorted(values)[len(values)//2]
            }
    
    print(f"📈 Papers with metrics: {len(papers) - len(issues['missing_metrics'])}")
    print(f"❌ Papers missing metrics: {len(issues['missing_metrics'])}")
    print(f"⚠️  Unrealistic accuracies: {len(issues['unrealistic_accuracy'])}")
    print(f"⚠️  Inconsistent metrics: {len(issues['inconsistent_metrics'])}")
    
    return {'issues': issues, 'statistics': stats_summary}

def verify_year_consistency(papers: dict) -> Dict[str, List[str]]:
    """Verify year consistency between citation keys and paper data"""
    
    print("\n📅 VERIFYING YEAR CONSISTENCY...")
    
    issues = {
        'inconsistent_years': [],
        'missing_years': [],
        'future_years': [],
        'very_old_years': []
    }
    
    current_year = 2024
    
    for paper_key, paper_data in papers.items():
        paper_year = paper_data.get('year', '')
        
        if not paper_year:
            issues['missing_years'].append(paper_key)
            continue
        
        # Extract year from citation key if possible
        key_year_match = re.search(r'(\d{4})', paper_key)
        if key_year_match:
            key_year = key_year_match.group(1)
            if key_year != paper_year:
                issues['inconsistent_years'].append(f"{paper_key}: key={key_year} vs data={paper_year}")
        
        # Check for unrealistic years
        try:
            year_int = int(paper_year)
            if year_int > current_year:
                issues['future_years'].append(f"{paper_key}: {paper_year}")
            elif year_int < 1990:  # Very old for agricultural robotics
                issues['very_old_years'].append(f"{paper_key}: {paper_year}")
        except ValueError:
            issues['missing_years'].append(f"{paper_key}: invalid year '{paper_year}'")
    
    print(f"📅 Papers with valid years: {len(papers) - len(issues['missing_years'])}")
    print(f"⚠️  Year inconsistencies: {len(issues['inconsistent_years'])}")
    print(f"🔮 Future years: {len(issues['future_years'])}")
    
    return issues

def verify_title_abstract_consistency(papers: dict) -> Dict[str, any]:
    """Verify consistency between titles and abstracts"""
    
    print("\n📝 VERIFYING TITLE-ABSTRACT CONSISTENCY...")
    
    issues = {
        'missing_abstracts': [],
        'generic_abstracts': [],
        'title_abstract_mismatch': [],
        'duplicate_abstracts': []
    }
    
    abstract_counts = Counter()
    
    for paper_key, paper_data in papers.items():
        title = paper_data.get('title', '').lower()
        abstract = paper_data.get('abstract', '')
        
        if not abstract:
            issues['missing_abstracts'].append(paper_key)
            continue
        
        abstract_lower = abstract.lower()
        abstract_counts[abstract] += 1
        
        # Check for generic abstracts (same template used)
        if 'we propose a faster r-cnn based system' in abstract_lower:
            issues['generic_abstracts'].append(paper_key)
        
        # Check title-abstract consistency for key terms
        title_keywords = set(re.findall(r'\b\w+\b', title))
        abstract_keywords = set(re.findall(r'\b\w+\b', abstract_lower))
        
        # Check if main title concepts appear in abstract
        important_title_words = [word for word in title_keywords 
                               if len(word) > 3 and word not in {'with', 'using', 'based', 'for', 'and', 'the'}]
        
        if important_title_words:
            matching_words = sum(1 for word in important_title_words if word in abstract_keywords)
            match_ratio = matching_words / len(important_title_words)
            
            if match_ratio < 0.3:  # Less than 30% of important title words in abstract
                issues['title_abstract_mismatch'].append(f"{paper_key}: {match_ratio:.2f} match ratio")
    
    # Find duplicate abstracts
    for abstract, count in abstract_counts.items():
        if count > 1:
            duplicate_papers = [key for key, data in papers.items() if data.get('abstract') == abstract]
            issues['duplicate_abstracts'].extend(duplicate_papers)
    
    print(f"📝 Papers with abstracts: {len(papers) - len(issues['missing_abstracts'])}")
    print(f"🔄 Generic abstracts: {len(issues['generic_abstracts'])}")
    print(f"🔄 Duplicate abstracts: {len(set(issues['duplicate_abstracts']))}")
    print(f"⚠️  Title-abstract mismatches: {len(issues['title_abstract_mismatch'])}")
    
    return issues

def verify_dataset_consistency(papers: dict) -> Dict[str, any]:
    """Verify dataset information consistency"""
    
    print("\n📁 VERIFYING DATASET CONSISTENCY...")
    
    all_datasets = []
    dataset_counts = Counter()
    
    issues = {
        'missing_datasets': [],
        'generic_datasets': [],
        'unrealistic_sample_sizes': []
    }
    
    for paper_key, paper_data in papers.items():
        datasets = paper_data.get('datasets', [])
        exp_results = paper_data.get('experimental_results', {})
        
        if not datasets:
            issues['missing_datasets'].append(paper_key)
        else:
            for dataset in datasets:
                all_datasets.append(dataset)
                dataset_counts[dataset] += 1
                
                # Check for generic dataset names
                if dataset.lower() in ['agricultural dataset', 'computer vision dataset', 'custom dataset']:
                    issues['generic_datasets'].append(f"{paper_key}: {dataset}")
        
        # Check sample sizes
        sample_size = exp_results.get('sample_size')
        if sample_size:
            if sample_size < 100 or sample_size > 100000:
                issues['unrealistic_sample_sizes'].append(f"{paper_key}: {sample_size} samples")
    
    # Most common datasets
    common_datasets = dataset_counts.most_common(10)
    
    print(f"📁 Papers with datasets: {len(papers) - len(issues['missing_datasets'])}")
    print(f"📊 Unique datasets found: {len(set(all_datasets))}")
    print(f"🔄 Generic datasets: {len(issues['generic_datasets'])}")
    print(f"⚠️  Unrealistic sample sizes: {len(issues['unrealistic_sample_sizes'])}")
    
    return {
        'issues': issues,
        'common_datasets': common_datasets,
        'total_unique_datasets': len(set(all_datasets))
    }

def generate_consistency_report(citation_issues, metrics_analysis, year_issues, 
                              title_abstract_issues, dataset_analysis):
    """Generate comprehensive consistency report"""
    
    print("\n📋 COMPREHENSIVE CONSISTENCY REPORT")
    print("=" * 50)
    
    # Citation consistency
    print("\n🔗 CITATION CONSISTENCY:")
    print(f"  ✅ Valid citations: {len(citation_issues['valid_citations'])}")
    print(f"  ❌ Missing from refs.bib: {len(citation_issues['missing_citations'])}")
    print(f"  📋 Extra in refs.bib: {len(citation_issues['extra_citations'])}")
    
    if citation_issues['missing_citations'][:5]:
        print(f"  Examples of missing: {citation_issues['missing_citations'][:5]}")
    
    # Performance metrics consistency
    print("\n📊 PERFORMANCE METRICS CONSISTENCY:")
    metrics_issues = metrics_analysis['issues']
    print(f"  ✅ Papers with metrics: {312 - len(metrics_issues['missing_metrics'])}")
    print(f"  ❌ Missing metrics: {len(metrics_issues['missing_metrics'])}")
    print(f"  ⚠️  Unrealistic accuracies: {len(metrics_issues['unrealistic_accuracy'])}")
    print(f"  ⚠️  Inconsistent metrics: {len(metrics_issues['inconsistent_metrics'])}")
    
    if metrics_issues['unrealistic_accuracy'][:3]:
        print(f"  Examples of unrealistic: {metrics_issues['unrealistic_accuracy'][:3]}")
    
    # Statistics summary
    stats = metrics_analysis['statistics']
    if 'accuracy' in stats:
        acc_stats = stats['accuracy']
        print(f"  📈 Accuracy range: {acc_stats['min']:.1f}% - {acc_stats['max']:.1f}% (avg: {acc_stats['avg']:.1f}%)")
    
    # Year consistency
    print("\n📅 YEAR CONSISTENCY:")
    print(f"  ✅ Valid years: {312 - len(year_issues['missing_years'])}")
    print(f"  ⚠️  Year inconsistencies: {len(year_issues['inconsistent_years'])}")
    print(f"  🔮 Future years: {len(year_issues['future_years'])}")
    
    if year_issues['inconsistent_years'][:3]:
        print(f"  Examples: {year_issues['inconsistent_years'][:3]}")
    
    # Title-abstract consistency
    print("\n📝 TITLE-ABSTRACT CONSISTENCY:")
    print(f"  ✅ Papers with abstracts: {312 - len(title_abstract_issues['missing_abstracts'])}")
    print(f"  🔄 Generic abstracts: {len(title_abstract_issues['generic_abstracts'])}")
    print(f"  🔄 Duplicate abstracts: {len(set(title_abstract_issues['duplicate_abstracts']))}")
    print(f"  ⚠️  Title-abstract mismatches: {len(title_abstract_issues['title_abstract_mismatch'])}")
    
    # Dataset consistency
    print("\n📁 DATASET CONSISTENCY:")
    dataset_issues = dataset_analysis['issues']
    print(f"  ✅ Papers with datasets: {312 - len(dataset_issues['missing_datasets'])}")
    print(f"  📊 Unique datasets: {dataset_analysis['total_unique_datasets']}")
    print(f"  🔄 Generic datasets: {len(dataset_issues['generic_datasets'])}")
    
    if dataset_analysis['common_datasets'][:5]:
        print(f"  Top datasets: {[item[0] for item in dataset_analysis['common_datasets'][:5]]}")
    
    # Overall assessment
    print("\n🎯 OVERALL DATA QUALITY ASSESSMENT:")
    
    total_issues = (
        len(citation_issues['missing_citations']) +
        len(metrics_issues['unrealistic_accuracy']) +
        len(metrics_issues['inconsistent_metrics']) +
        len(year_issues['inconsistent_years']) +
        len(title_abstract_issues['title_abstract_mismatch'])
    )
    
    if total_issues < 10:
        quality_score = "🟢 EXCELLENT"
    elif total_issues < 50:
        quality_score = "🟡 GOOD"
    elif total_issues < 100:
        quality_score = "🟠 ACCEPTABLE"
    else:
        quality_score = "🔴 NEEDS IMPROVEMENT"
    
    print(f"  Data Quality Score: {quality_score}")
    print(f"  Total Issues Found: {total_issues}")
    print(f"  Citation Accuracy: {len(citation_issues['valid_citations'])/312*100:.1f}%")
    print(f"  Metrics Coverage: {(312-len(metrics_issues['missing_metrics']))/312*100:.1f}%")
    
    return {
        'quality_score': quality_score,
        'total_issues': total_issues,
        'citation_accuracy': len(citation_issues['valid_citations'])/312*100,
        'metrics_coverage': (312-len(metrics_issues['missing_metrics']))/312*100
    }

def main():
    """Main function to run comprehensive data consistency verification"""
    
    print("🔍 COMPREHENSIVE DATA CONSISTENCY VERIFICATION")
    print("=" * 55)
    
    # Load data
    data = load_analysis_data()
    papers = data['papers_with_metrics']
    
    print(f"📊 Total papers to verify: {len(papers)}")
    
    # Run all consistency checks
    citation_issues = verify_citation_consistency(papers)
    metrics_analysis = verify_performance_metrics_consistency(papers)
    year_issues = verify_year_consistency(papers)
    title_abstract_issues = verify_title_abstract_consistency(papers)
    dataset_analysis = verify_dataset_consistency(papers)
    
    # Generate comprehensive report
    overall_assessment = generate_consistency_report(
        citation_issues, metrics_analysis, year_issues, 
        title_abstract_issues, dataset_analysis
    )
    
    # Save detailed verification results
    verification_results = {
        'citation_consistency': citation_issues,
        'metrics_consistency': metrics_analysis,
        'year_consistency': year_issues,
        'title_abstract_consistency': title_abstract_issues,
        'dataset_consistency': dataset_analysis,
        'overall_assessment': overall_assessment
    }
    
    output_file = '/workspace/benchmarks/figure_generation/DATA_CONSISTENCY_VERIFICATION_REPORT.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ VERIFICATION COMPLETE!")
    print(f"📄 Detailed report saved: {output_file}")
    
    return verification_results

if __name__ == "__main__":
    main()