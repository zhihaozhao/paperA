#!/usr/bin/env python3
"""
REAL PDF DATA EXTRACTOR
Extracts genuine abstracts and performance metrics from uploaded PDF papers
"""

import os
import re
import json
import PyPDF2
import fitz  # PyMuPDF for better text extraction
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def find_pdf_papers() -> List[str]:
    """Find all PDF papers in the references directory"""
    
    references_dir = Path('/workspace/references')
    pdf_files = []
    
    if references_dir.exists():
        for pdf_file in references_dir.glob('*.pdf'):
            pdf_files.append(str(pdf_file))
    
    print(f"📚 Found {len(pdf_files)} PDF papers:")
    for pdf in pdf_files:
        print(f"  - {Path(pdf).name}")
    
    return pdf_files

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (better than PyPDF2)"""
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extract text from first few pages (where abstract usually is)
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
        
    except Exception as e:
        print(f"❌ Error extracting from {Path(pdf_path).name}: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(min(3, len(pdf_reader.pages))):
                    text += pdf_reader.pages[page_num].extract_text()
                return text
        except Exception as e2:
            print(f"❌ Fallback also failed for {Path(pdf_path).name}: {e2}")
            return ""

def extract_abstract_from_text(text: str, paper_name: str) -> Optional[str]:
    """Extract abstract from paper text"""
    
    if not text:
        return None
    
    # Clean text
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.replace('\n', ' ')
    
    # Common abstract patterns
    abstract_patterns = [
        r'ABSTRACT[:\s]+(.*?)(?:Keywords|KEYWORDS|Introduction|INTRODUCTION|1\.|I\.)',
        r'Abstract[:\s]+(.*?)(?:Keywords|Introduction|1\.|I\.)',
        r'—(.*?)(?:Index Terms|Keywords|Introduction)',
        r'Abstract—(.*?)(?:Index Terms|Keywords|Introduction)',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)  # Normalize spaces
            abstract = re.sub(r'[^\w\s\.,;:!?%-]', '', abstract)  # Remove weird chars
            
            # Validate abstract (should be reasonable length)
            if 50 <= len(abstract) <= 2000:
                print(f"✅ Extracted abstract from {paper_name}: {len(abstract)} chars")
                return abstract
    
    print(f"⚠️  No abstract found in {paper_name}")
    return None

def extract_performance_metrics_from_text(text: str, paper_name: str) -> Dict[str, float]:
    """Extract performance metrics from paper text"""
    
    metrics = {}
    
    if not text:
        return metrics
    
    # Accuracy patterns
    accuracy_patterns = [
        r'accuracy[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*accuracy',
        r'achieved\s*(\d+(?:\.\d+)?)\s*%',
        r'accuracy[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)',
    ]
    
    for pattern in accuracy_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                # Take the highest reasonable accuracy
                accuracies = [float(m) for m in matches if 50 <= float(m) <= 100]
                if accuracies:
                    metrics['accuracy'] = max(accuracies)
                    break
            except ValueError:
                continue
    
    # Precision patterns
    precision_patterns = [
        r'precision[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*precision',
    ]
    
    for pattern in precision_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                precisions = [float(m) for m in matches if 50 <= float(m) <= 100]
                if precisions:
                    metrics['precision'] = max(precisions)
                    break
            except ValueError:
                continue
    
    # Recall patterns
    recall_patterns = [
        r'recall[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*recall',
    ]
    
    for pattern in recall_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                recalls = [float(m) for m in matches if 50 <= float(m) <= 100]
                if recalls:
                    metrics['recall'] = max(recalls)
                    break
            except ValueError:
                continue
    
    # F1-Score patterns
    f1_patterns = [
        r'f1[:\s-]*score[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'f1[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
    ]
    
    for pattern in f1_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                f1_scores = [float(m) for m in matches if 50 <= float(m) <= 100]
                if f1_scores:
                    metrics['f1_score'] = max(f1_scores)
                    break
            except ValueError:
                continue
    
    # mAP patterns
    map_patterns = [
        r'mAP[:\s]*(?:of\s*)?(\d+(?:\.\d+)?)\s*%',
        r'mean\s*average\s*precision[:\s]*(\d+(?:\.\d+)?)\s*%',
    ]
    
    for pattern in map_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                maps = [float(m) for m in matches if 50 <= float(m) <= 100]
                if maps:
                    metrics['mAP'] = max(maps)
                    break
            except ValueError:
                continue
    
    # FPS patterns
    fps_patterns = [
        r'(\d+(?:\.\d+)?)\s*fps',
        r'(\d+(?:\.\d+)?)\s*frames?\s*per\s*second',
    ]
    
    for pattern in fps_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                fps_values = [float(m) for m in matches if 1 <= float(m) <= 1000]
                if fps_values:
                    metrics['fps'] = max(fps_values)
                    break
            except ValueError:
                continue
    
    if metrics:
        print(f"📊 Extracted metrics from {paper_name}: {list(metrics.keys())}")
    
    return metrics

def extract_datasets_from_text(text: str, paper_name: str) -> List[str]:
    """Extract dataset names from paper text"""
    
    datasets = []
    
    if not text:
        return datasets
    
    # Dataset patterns
    dataset_patterns = [
        r'(\w*COCO\w*)',
        r'(\w*ImageNet\w*)',
        r'(\w*PASCAL\w*)',
        r'(\w+)\s*dataset',
        r'dataset[s]?\s*(?:of\s*)?(\w+)',
    ]
    
    for pattern in dataset_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            if match and len(match) > 2:
                datasets.append(match.strip())
    
    # Remove duplicates and common words
    exclude_words = {'the', 'and', 'or', 'with', 'using', 'based', 'on', 'for', 'in', 'of', 'to', 'this', 'our'}
    datasets = list(set([d for d in datasets if d.lower() not in exclude_words]))
    
    if datasets:
        print(f"📁 Extracted datasets from {paper_name}: {datasets[:3]}")
    
    return datasets[:5]  # Limit to top 5

def match_pdf_to_citation(pdf_path: str, citations: List[str]) -> Optional[str]:
    """Try to match PDF file to citation key"""
    
    pdf_name = Path(pdf_path).stem.lower()
    
    # Try direct matching first
    for citation in citations:
        citation_lower = citation.lower()
        
        # Extract year and author from citation key
        year_match = re.search(r'(\d{4})', citation_lower)
        author_match = re.match(r'([a-zA-Z]+)', citation_lower)
        
        if year_match and author_match:
            year = year_match.group(1)
            author = author_match.group(1)
            
            # Check if PDF name contains author and year
            if author in pdf_name and year in pdf_name:
                return citation
        
        # Try partial matching
        if any(word in pdf_name for word in citation_lower.split('_') if len(word) > 3):
            return citation
    
    return None

def process_pdf_papers() -> Dict[str, Dict]:
    """Process all PDF papers to extract real data"""
    
    print("🔍 PROCESSING REAL PDF PAPERS...")
    print("=" * 50)
    
    # Find PDF papers
    pdf_files = find_pdf_papers()
    
    if not pdf_files:
        print("❌ No PDF papers found!")
        return {}
    
    # Load existing analysis to get citation keys
    try:
        with open('/workspace/benchmarks/figure_generation/FINAL_COMPREHENSIVE_ANALYSIS.json', 'r') as f:
            existing_data = json.load(f)
        citations = list(existing_data['papers_with_metrics'].keys())
        print(f"📋 Found {len(citations)} existing citations to match")
    except:
        print("⚠️  Could not load existing analysis, proceeding without citation matching")
        citations = []
    
    # Process each PDF
    real_paper_data = {}
    successful_extractions = 0
    
    for pdf_path in pdf_files:
        paper_name = Path(pdf_path).stem
        print(f"\n🔍 Processing: {paper_name}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"❌ Could not extract text from {paper_name}")
            continue
        
        # Extract data
        abstract = extract_abstract_from_text(text, paper_name)
        metrics = extract_performance_metrics_from_text(text, paper_name)
        datasets = extract_datasets_from_text(text, paper_name)
        
        # Try to match to citation
        matched_citation = match_pdf_to_citation(pdf_path, citations)
        
        # Store data
        paper_data = {
            'pdf_file': pdf_path,
            'paper_name': paper_name,
            'matched_citation': matched_citation,
            'abstract': abstract,
            'performance_metrics': metrics,
            'datasets': datasets,
            'text_length': len(text),
            'has_real_abstract': abstract is not None,
            'has_real_metrics': len(metrics) > 0
        }
        
        real_paper_data[paper_name] = paper_data
        
        if abstract or metrics:
            successful_extractions += 1
    
    print(f"\n✅ Successfully processed {successful_extractions}/{len(pdf_files)} papers")
    
    return real_paper_data

def update_analysis_with_real_data(real_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Update existing analysis with real PDF data"""
    
    print("\n🔄 UPDATING ANALYSIS WITH REAL PDF DATA...")
    
    # Load existing analysis
    try:
        with open('/workspace/benchmarks/figure_generation/FINAL_COMPREHENSIVE_ANALYSIS.json', 'r') as f:
            existing_data = json.load(f)
        papers = existing_data['papers_with_metrics']
    except:
        print("❌ Could not load existing analysis")
        return {}
    
    updates_made = 0
    
    # Update papers with real data
    for paper_name, real_paper in real_data.items():
        matched_citation = real_paper['matched_citation']
        
        if matched_citation and matched_citation in papers:
            # Update with real data
            if real_paper['has_real_abstract']:
                papers[matched_citation]['abstract'] = real_paper['abstract']
                papers[matched_citation]['has_abstract'] = True
                updates_made += 1
            
            if real_paper['has_real_metrics']:
                papers[matched_citation]['performance_metrics'].update(real_paper['performance_metrics'])
                updates_made += 1
            
            if real_paper['datasets']:
                papers[matched_citation]['datasets'] = real_paper['datasets']
            
            # Mark as real data
            papers[matched_citation]['data_source'] = 'real_pdf'
            papers[matched_citation]['pdf_file'] = real_paper['pdf_file']
    
    print(f"✅ Updated {updates_made} papers with real PDF data")
    
    return papers

def main():
    """Main function to extract real data from PDF papers"""
    
    print("📚 REAL PDF DATA EXTRACTION")
    print("=" * 40)
    
    # Check if PyMuPDF is available
    try:
        import fitz
        print("✅ PyMuPDF available for better text extraction")
    except ImportError:
        print("⚠️  PyMuPDF not available, using PyPDF2 fallback")
    
    # Process PDF papers
    real_data = process_pdf_papers()
    
    # Save real PDF data
    real_data_file = '/workspace/benchmarks/figure_generation/REAL_PDF_EXTRACTED_DATA.json'
    with open(real_data_file, 'w', encoding='utf-8') as f:
        json.dump(real_data, f, indent=2, ensure_ascii=False)
    
    # Update existing analysis
    updated_papers = update_analysis_with_real_data(real_data)
    
    if updated_papers:
        # Save updated analysis
        updated_analysis_file = '/workspace/benchmarks/figure_generation/UPDATED_WITH_REAL_PDF_DATA.json'
        updated_analysis = {
            'papers_with_metrics': updated_papers,
            'real_data_summary': {
                'total_pdfs_processed': len(real_data),
                'successful_extractions': sum(1 for p in real_data.values() if p['has_real_abstract'] or p['has_real_metrics']),
                'papers_with_real_abstracts': sum(1 for p in real_data.values() if p['has_real_abstract']),
                'papers_with_real_metrics': sum(1 for p in real_data.values() if p['has_real_metrics']),
            }
        }
        
        with open(updated_analysis_file, 'w', encoding='utf-8') as f:
            json.dump(updated_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ REAL DATA EXTRACTION COMPLETE!")
        print(f"📄 Real PDF data: {real_data_file}")
        print(f"📊 Updated analysis: {updated_analysis_file}")
        
        # Summary
        summary = updated_analysis['real_data_summary']
        print(f"\n📋 EXTRACTION SUMMARY:")
        print(f"  📚 PDFs processed: {summary['total_pdfs_processed']}")
        print(f"  ✅ Successful extractions: {summary['successful_extractions']}")
        print(f"  📝 Real abstracts: {summary['papers_with_real_abstracts']}")
        print(f"  📊 Real metrics: {summary['papers_with_real_metrics']}")
    
    return real_data

if __name__ == "__main__":
    main()