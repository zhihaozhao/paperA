#!/usr/bin/env python3
"""
Literature Data Extractor for Fruit-Picking Robot Survey
Extracts quantitative performance data from LaTeX tables for meta-analysis

Author: Research Team
Date: 2024
Purpose: Transform literature review into data-driven meta-analysis
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

class LiteratureDataExtractor:
    """Extract quantitative data from LaTeX survey tables"""
    
    def __init__(self, tex_file_path: str):
        self.tex_file_path = Path(tex_file_path)
        self.raw_data = []
        self.processed_data = None
        
        # Performance metric extraction patterns
        self.patterns = {
            'accuracy': r'(?:accuracy|precision)[:=\s]*(\d+\.?\d*)%',
            'f1_score': r'F1[:=\s]*(\d+\.?\d*)',
            'map_score': r'mAP[:=\s]*(\d+\.?\d*)',
            'speed_ms': r'(\d+\.?\d*)\s*ms/image',
            'speed_fps': r'(\d+\.?\d*)\s*FPS',
            'cycle_time': r'cycle\s*time[:=\s]*(\d+\.?\d*)\s*s',
            'success_rate': r'success\s*rate[:=\s]*(\d+\.?\d*)%',
            'processing_time': r'processing\s*time[:=\s]*(\d+\.?\d*)\s*(?:ms|s)',
            'model_size': r'model\s*size[:=\s]*(\d+\.?\d*)\s*MB',
            'epochs': r'(\d+)\s*epochs',
            'training_images': r'(\d+)[,\s]*(?:training\s*)?images'
        }
        
        # Algorithm family classification
        self.algorithm_families = {
            'R-CNN': ['r-cnn', 'faster r-cnn', 'mask r-cnn', 'deepfruits'],
            'YOLO': ['yolo', 'you only look once'],
            'SSD': ['ssd', 'single shot'],
            'Hybrid': ['fusion', 'ensemble', 'combined'],
            'Traditional': ['svm', 'hog', 'lbp', 'template']
        }
        
        # Environment classification
        self.environments = {
            'Greenhouse': ['greenhouse', 'glasshouse', 'indoor'],
            'Outdoor': ['outdoor', 'field', 'orchard', 'natural'],
            'Commercial': ['commercial', 'industrial'],
            'Laboratory': ['lab', 'laboratory', 'controlled']
        }

    def load_tex_file(self) -> str:
        """Load and return the LaTeX file content"""
        try:
            with open(self.tex_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading file {self.tex_file_path}: {e}")
            return ""

    def extract_table_content(self, content: str) -> List[str]:
        """Extract all table environments from LaTeX content"""
        # Find all table environments
        table_pattern = r'\\begin\{table\*?\}.*?\\end\{table\*?\}'
        tables = re.findall(table_pattern, content, re.DOTALL)
        return tables

    def parse_table_rows(self, table_content: str) -> List[Dict]:
        """Parse individual table rows and extract data"""
        rows = []
        
        # Extract citation and year
        cite_pattern = r'\\cite\{([^}]+)\}.*?(\d{4})'
        citations = re.findall(cite_pattern, table_content)
        
        # Extract performance metrics
        for cite, year in citations:
            row_data = {
                'reference': cite,
                'year': int(year),
                'algorithm_family': self.classify_algorithm(table_content, cite),
                'environment': self.classify_environment(table_content, cite),
                'fruit_type': self.extract_fruit_type(table_content, cite)
            }
            
            # Extract numerical metrics
            for metric, pattern in self.patterns.items():
                matches = re.findall(pattern, table_content, re.IGNORECASE)
                if matches:
                    try:
                        row_data[metric] = float(matches[0])
                    except ValueError:
                        row_data[metric] = None
                else:
                    row_data[metric] = None
            
            rows.append(row_data)
        
        return rows

    def classify_algorithm(self, content: str, citation: str) -> str:
        """Classify algorithm family based on content context"""
        # Extract text around citation
        cite_context = self.get_citation_context(content, citation)
        
        for family, keywords in self.algorithm_families.items():
            if any(keyword.lower() in cite_context.lower() for keyword in keywords):
                return family
        return 'Other'

    def classify_environment(self, content: str, citation: str) -> str:
        """Classify environment type based on content context"""
        cite_context = self.get_citation_context(content, citation)
        
        for env_type, keywords in self.environments.items():
            if any(keyword.lower() in cite_context.lower() for keyword in keywords):
                return env_type
        return 'Unknown'

    def extract_fruit_type(self, content: str, citation: str) -> str:
        """Extract fruit type from content context"""
        cite_context = self.get_citation_context(content, citation)
        
        fruits = ['apple', 'tomato', 'grape', 'citrus', 'cherry', 'strawberry', 
                 'pepper', 'melon', 'pear', 'kiwi', 'broccoli', 'cauliflower']
        
        for fruit in fruits:
            if fruit.lower() in cite_context.lower():
                return fruit.capitalize()
        return 'Multi-class'

    def get_citation_context(self, content: str, citation: str, context_length: int = 200) -> str:
        """Get text context around a citation"""
        cite_pos = content.find(f'\\cite{{{citation}}}')
        if cite_pos == -1:
            return ""
        
        start = max(0, cite_pos - context_length)
        end = min(len(content), cite_pos + context_length)
        return content[start:end]

    def extract_all_data(self) -> pd.DataFrame:
        """Main function to extract all data from the LaTeX file"""
        print(f"Extracting data from {self.tex_file_path}...")
        
        # Load file content
        content = self.load_tex_file()
        if not content:
            return pd.DataFrame()
        
        # Extract tables
        tables = self.extract_table_content(content)
        print(f"Found {len(tables)} tables")
        
        # Parse each table
        all_data = []
        for i, table in enumerate(tables):
            print(f"Processing table {i+1}/{len(tables)}...")
            table_data = self.parse_table_rows(table)
            all_data.extend(table_data)
        
        # Create DataFrame
        self.processed_data = pd.DataFrame(all_data)
        
        # Data cleaning and validation
        self.processed_data = self.clean_data(self.processed_data)
        
        print(f"Extracted data for {len(self.processed_data)} studies")
        return self.processed_data

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate extracted data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['reference', 'year'])
        
        # Validate year range
        df = df[(df['year'] >= 2015) & (df['year'] <= 2024)]
        
        # Convert percentage to decimal for some metrics
        percentage_cols = ['accuracy', 'success_rate']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col] / 100.0  # Convert to decimal
        
        # Fill missing values with NaN
        df = df.replace('', np.nan)
        
        return df

    def save_data(self, output_path: str = 'extracted_literature_data.csv'):
        """Save extracted data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
            
            # Save summary statistics
            summary_path = output_path.replace('.csv', '_summary.json')
            summary = {
                'total_studies': len(self.processed_data),
                'year_range': [int(self.processed_data['year'].min()), 
                             int(self.processed_data['year'].max())],
                'algorithm_families': self.processed_data['algorithm_family'].value_counts().to_dict(),
                'environments': self.processed_data['environment'].value_counts().to_dict(),
                'fruit_types': self.processed_data['fruit_type'].value_counts().to_dict()
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {summary_path}")
        else:
            print("No data to save. Run extract_all_data() first.")

    def get_data_summary(self) -> Dict:
        """Get summary statistics of extracted data"""
        if self.processed_data is None:
            return {}
        
        return {
            'studies_count': len(self.processed_data),
            'year_distribution': self.processed_data['year'].value_counts().sort_index().to_dict(),
            'algorithm_distribution': self.processed_data['algorithm_family'].value_counts().to_dict(),
            'performance_stats': {
                'accuracy_mean': self.processed_data['accuracy'].mean(),
                'accuracy_std': self.processed_data['accuracy'].std(),
                'speed_mean': self.processed_data['speed_ms'].mean(),
                'speed_std': self.processed_data['speed_ms'].std()
            }
        }

def main():
    """Main function to demonstrate usage"""
    # Initialize extractor
    extractor = LiteratureDataExtractor('FP_2025_SN-APPLIED-SCIENCES/FP_2025_SN-APPLIED-SCIENCES_v1.tex')
    
    # Extract data
    data = extractor.extract_all_data()
    
    # Save results
    extractor.save_data('fruit_picking_literature_data.csv')
    
    # Print summary
    summary = extractor.get_data_summary()
    print("\n=== DATA EXTRACTION SUMMARY ===")
    print(f"Total studies: {summary.get('studies_count', 0)}")
    print(f"Algorithm families: {list(summary.get('algorithm_distribution', {}).keys())}")
    print(f"Performance metrics extracted: {len([k for k in data.columns if not k.startswith('reference')])}")
    
    return data

if __name__ == "__main__":
    extracted_data = main()