#!/usr/bin/env python3
"""
Extract fencing knowledge from PDFs and prepare passages for the RAG system.
This script processes fencing PDFs and creates a JSON file with passages for knowledge enhancement.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from PyPDF2 import PdfReader
import re
import nltk
from nltk.tokenize import sent_tokenize
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with error handling."""
    logger.info(f"Extracting text from {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                else:
                    logger.warning(f"Page {i} in {pdf_path} yielded no text")
            except Exception as e:
                logger.error(f"Error extracting text from page {i} in {pdf_path}: {e}")
        return text
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Clean and normalize text with improved handling of fencing terminology."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Standardize fencing terms
    replacements = {
        'épée': 'epee',
        'Épée': 'Epee',
        'foil': 'foil',
        'sabre': 'saber',
        'riposte': 'riposte',
        'parries': 'parries',
        'parry': 'parry',
        'en garde': 'en garde',
        'feint': 'feint',
        'counter-attack': 'counter-attack',
        'counterattack': 'counter-attack',
        'right-of-way': 'right of way',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Additional cleaning
    text = re.sub(r'[\r\n]+', '\n', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_fencing_passages(text, filename, min_passage_length=150, max_passage_length=500):
    """Extract relevant passages about fencing techniques and concepts."""
    passages = []
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Define categories of fencing concepts and related keywords
    fencing_keywords = [
        # Footwork
        'advance', 'retreat', 'lunge', 'fleche', 'stance', 'footwork', 'balance',
        'distance', 'measure', 'mobility', 'step', 'position', 'posture',
        
        # Attacks
        'attack', 'offensive', 'thrust', 'cut', 'lunge', 'feint', 'beat', 
        'pressure', 'glide', 'disengage', 'extension', 'remise', 'redouble',
        
        # Defenses
        'parry', 'defense', 'defensive', 'riposte', 'counter-attack', 'stop hit',
        'opposition', 'block', 'counter', 'point-in-line', 'displacement',
        
        # Tactics
        'strategy', 'tactic', 'timing', 'second intention', 'counter-time',
        'preparation', 'priority', 'right of way', 'tempo', 'initiative',
        
        # Weapons
        'foil', 'epee', 'saber', 'blade', 'point', 'edge', 'guard', 'grip',
        'pommel', 'forte', 'foible', 'weapon',
        
        # Additional fencing-specific terms
        'bind', 'engagement', 'corps-à-corps', 'prise de fer', 'croisé',
        'double', 'reprise', 'passata sotto', 'appel', 'balestra',
        'invitation', 'line', 'fencer', 'bout', 'referee', 'director',
        'piste', 'strip', 'mask', 'glove', 'jacket', 'plastron'
    ]
    
    # Process each paragraph
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) < min_passage_length:  
            continue
            
        paragraph_lower = paragraph.lower()
        
        # Check if this paragraph contains fencing content
        if any(keyword in paragraph_lower for keyword in fencing_keywords):
            # Process paragraphs that are too long
            if len(paragraph) > max_passage_length:
                # Split into sentences
                sentences = sent_tokenize(paragraph)
                
                # Group sentences into smaller chunks (roughly max_passage_length)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_len = len(sentence)
                    
                    if current_length + sentence_len <= max_passage_length:
                        current_chunk.append(sentence)
                        current_length += sentence_len
                    else:
                        # Save current chunk if it's substantial
                        if current_length >= min_passage_length:
                            passage = ' '.join(current_chunk)
                            # Store in structured format with source information
                            passage_obj = {
                                "text": passage,
                                "source": os.path.basename(filename),
                                "keywords": [kw for kw in fencing_keywords if kw in passage.lower()]
                            }
                            passages.append(passage_obj)
                        
                        # Start a new chunk
                        current_chunk = [sentence]
                        current_length = sentence_len
                
                # Don't forget the last chunk
                if current_chunk and current_length >= min_passage_length:
                    passage = ' '.join(current_chunk)
                    passage_obj = {
                        "text": passage,
                        "source": os.path.basename(filename),
                        "keywords": [kw for kw in fencing_keywords if kw in passage.lower()]
                    }
                    passages.append(passage_obj)
            else:
                # Paragraph is already an appropriate size
                passage_obj = {
                    "text": paragraph,
                    "source": os.path.basename(filename),
                    "keywords": [kw for kw in fencing_keywords if kw in paragraph.lower()]
                }
                passages.append(passage_obj)
    
    logger.info(f"Extracted {len(passages)} fencing passages from {os.path.basename(filename)}")
    return passages

def process_fencing_books(books_dir, output_file):
    """Process all fencing PDFs and create a knowledge base."""
    books_path = Path(books_dir)
    if not books_path.exists():
        logger.error(f"Books directory not found: {books_dir}")
        return False
    
    all_passages = []
    pdf_files = list(books_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {books_dir}")
        return False
    
    logger.info(f"Processing {len(pdf_files)} fencing books...")
    
    for pdf_path in pdf_files:
        try:
            # Extract and clean text
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_path.name}, skipping")
                continue
                
            cleaned_text = clean_text(text)
            
            # Extract passages
            passages = extract_fencing_passages(cleaned_text, pdf_path)
            all_passages.extend(passages)
            
            logger.info(f"Processed {pdf_path.name}: extracted {len(passages)} passages")
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
    
    # Save all passages to JSON file
    if all_passages:
        output_path = Path(output_file)
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_passages, f, indent=2)
            
        logger.info(f"Saved {len(all_passages)} fencing passages to {output_file}")
        return True
    else:
        logger.error("No passages extracted from any PDF")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract fencing knowledge from PDFs for RAG')
    parser.add_argument('--books_dir', type=str, default='GardeRAG/books',
                        help='Directory containing fencing books (PDFs)')
    parser.add_argument('--output_file', type=str, default='GardeRAG/fencing_passages.json',
                        help='Output JSON file for fencing passages')
    parser.add_argument('--min_length', type=int, default=150,
                        help='Minimum length of passages to extract')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum length of passages (longer will be split)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    success = process_fencing_books(args.books_dir, args.output_file)
    if success:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 