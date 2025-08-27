#!/usr/bin/env python3
"""
Extract bibliography from refs.bib and create JSON/CSV files with paper metadata and metrics.
"""

import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any

def parse_bibtex(bib_file: str) -> List[Dict[str, Any]]:
    """Parse BibTeX file and extract entries."""
    entries = []
    current_entry = None
    current_field = None
    field_value = []
    
    with open(bib_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Start of new entry
        if line.startswith('@'):
            if current_entry:
                if current_field:
                    current_entry[current_field] = ' '.join(field_value).strip().strip(',').strip('{}')
                entries.append(current_entry)
            
            # Extract entry type and key
            match = re.match(r'@(\w+)\{([^,]+),?', line)
            if match:
                entry_type, entry_key = match.groups()
                current_entry = {
                    'key': entry_key,
                    'type': entry_type.lower()
                }
                current_field = None
                field_value = []
        
        # Field within entry
        elif current_entry and '=' in line:
            if current_field:
                current_entry[current_field] = ' '.join(field_value).strip().strip(',').strip('{}')
            
            parts = line.split('=', 1)
            current_field = parts[0].strip()
            field_value = [parts[1].strip().strip(',')]
        
        # Continuation of field value
        elif current_entry and current_field:
            field_value.append(line.strip().strip(','))
        
        # End of entry
        elif line == '}' and current_entry:
            if current_field:
                current_entry[current_field] = ' '.join(field_value).strip().strip(',').strip('{}')
            entries.append(current_entry)
            current_entry = None
            current_field = None
            field_value = []
        
        i += 1
    
    # Handle last entry if file doesn't end with }
    if current_entry:
        if current_field:
            current_entry[current_field] = ' '.join(field_value).strip().strip(',').strip('{}')
        entries.append(current_entry)
    
    return entries

def extract_metrics_from_note(note: str) -> Dict[str, Any]:
    """Extract metrics from note field."""
    metrics = {
        'accuracy': None,
        'f1_score': None,
        'precision': None,
        'recall': None,
        'compression_ratio': None,
        'throughput': None,
        'params': None
    }
    
    if not note:
        return metrics
    
    # Common patterns for metrics
    patterns = {
        'accuracy': r'(\d+(?:\.\d+)?)\s*%?\s*(?:accuracy|acc)',
        'f1_score': r'(\d+(?:\.\d+)?)\s*%?\s*(?:F1|f1)',
        'precision': r'(\d+(?:\.\d+)?)\s*%?\s*precision',
        'recall': r'(\d+(?:\.\d+)?)\s*%?\s*recall',
        'compression_ratio': r'(\d+(?:\.\d+)?)[xX×]\s*compression',
        'throughput': r'(\d+(?:\.\d+)?)\s*(?:samples?/s|fps|FPS)',
        'params': r'(\d+(?:\.\d+)?)\s*[MmKk]?\s*(?:params|parameters)'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, note, re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                metrics[metric] = float(value)
            except ValueError:
                metrics[metric] = value
    
    return metrics

def categorize_paper(entry: Dict[str, Any]) -> str:
    """Categorize paper based on keywords and title."""
    title = entry.get('title', '').lower()
    note = entry.get('note', '').lower()
    key = entry.get('key', '').lower()
    
    # Categories based on content
    if any(word in title + note for word in ['sensefi', 'wifi sensing', 'csi', 'channel state']):
        if 'few' in title or 'few-shot' in note:
            return 'few-shot-learning'
        elif 'domain' in title or 'cross-domain' in note:
            return 'domain-adaptation'
        elif 'efficient' in title or 'lightweight' in note:
            return 'efficiency'
        else:
            return 'wifi-sensing'
    elif 'calibration' in title or 'uncertainty' in title:
        return 'trustworthy-ml'
    elif 'physics' in title or 'propagation' in title or 'fresnel' in title:
        return 'physics-modeling'
    elif 'attention' in title or 'transformer' in title:
        return 'attention-mechanisms'
    elif 'state' in title and 'space' in title:
        return 'state-space-models'
    else:
        return 'general'

def process_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process entries to extract structured information."""
    processed = []
    
    for entry in entries:
        # Extract metrics from note field
        metrics = extract_metrics_from_note(entry.get('note', ''))
        
        # Create processed entry
        processed_entry = {
            'key': entry.get('key', ''),
            'type': entry.get('type', ''),
            'title': entry.get('title', ''),
            'authors': entry.get('author', ''),
            'year': entry.get('year', ''),
            'venue': entry.get('journal', entry.get('booktitle', '')),
            'doi': entry.get('doi', ''),
            'url': entry.get('url', ''),
            'note': entry.get('note', ''),
            'category': categorize_paper(entry),
            'metrics': metrics,
            'reported_accuracy': metrics['accuracy'] if metrics['accuracy'] else 'NR',
            'reported_f1': metrics['f1_score'] if metrics['f1_score'] else 'NR',
            'reported_params': metrics['params'] if metrics['params'] else 'NR'
        }
        
        processed.append(processed_entry)
    
    return processed

def save_json(data: List[Dict[str, Any]], output_file: str):
    """Save data as JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {output_file}")

def save_csv(data: List[Dict[str, Any]], output_file: str):
    """Save data as CSV."""
    if not data:
        return
    
    # Flatten metrics for CSV
    flattened_data = []
    for entry in data:
        flat_entry = {k: v for k, v in entry.items() if k != 'metrics'}
        # Add metrics as separate columns
        if 'metrics' in entry and entry['metrics']:
            for metric_key, metric_value in entry['metrics'].items():
                flat_entry[f'metric_{metric_key}'] = metric_value
        flattened_data.append(flat_entry)
    
    # Write CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
        writer.writeheader()
        writer.writerows(flattened_data)
    
    print(f"Saved {len(data)} entries to {output_file}")

def create_summary_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics of the bibliography."""
    stats = {
        'total_papers': len(data),
        'papers_by_year': {},
        'papers_by_category': {},
        'papers_with_metrics': 0,
        'papers_with_doi': 0,
        'papers_with_url': 0
    }
    
    for entry in data:
        # Count by year
        year = entry.get('year', 'Unknown')
        stats['papers_by_year'][year] = stats['papers_by_year'].get(year, 0) + 1
        
        # Count by category
        category = entry.get('category', 'Unknown')
        stats['papers_by_category'][category] = stats['papers_by_category'].get(category, 0) + 1
        
        # Count papers with metrics
        if entry.get('metrics') and any(v is not None for v in entry['metrics'].values()):
            stats['papers_with_metrics'] += 1
        
        # Count papers with DOI
        if entry.get('doi'):
            stats['papers_with_doi'] += 1
        
        # Count papers with URL
        if entry.get('url'):
            stats['papers_with_url'] += 1
    
    return stats

def main():
    """Main function."""
    # Paths
    bib_file = '/workspace/paper/refs.bib'
    output_dir = Path('/workspace/docs/experiments/bibliography')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse BibTeX
    print(f"Parsing {bib_file}...")
    entries = parse_bibtex(bib_file)
    print(f"Found {len(entries)} bibliography entries")
    
    # Process entries
    print("Processing entries...")
    processed = process_entries(entries)
    
    # Save as JSON
    json_file = output_dir / 'refs_claude4.1.json'
    save_json(processed, str(json_file))
    
    # Save as CSV
    csv_file = output_dir / 'refs_claude4.1.csv'
    save_csv(processed, str(csv_file))
    
    # Create summary statistics
    stats = create_summary_stats(processed)
    stats_file = output_dir / 'bibliography_stats_claude4.1.json'
    save_json(stats, str(stats_file))
    
    # Print summary
    print("\n=== Bibliography Summary ===")
    print(f"Total papers: {stats['total_papers']}")
    print(f"Papers with metrics: {stats['papers_with_metrics']}")
    print(f"Papers with DOI: {stats['papers_with_doi']}")
    print(f"Papers with URL: {stats['papers_with_url']}")
    print("\nPapers by category:")
    for category, count in sorted(stats['papers_by_category'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")
    print("\nPapers by year (recent):")
    for year, count in sorted(stats['papers_by_year'].items(), reverse=True)[:5]:
        print(f"  {year}: {count}")

if __name__ == '__main__':
    main()