import base64
import io
import logging
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import json
import torch
from PIL import Image
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from collections import Counter
import statistics  # Add this import for median calculation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .types import (
    PDFProcessingResult,
    ContentItem,
    ContentType,
    TableContent
)


class PDFProcessor:    
    def __init__(self):
        self.content_id_counter = 1
        self.summarizer = None
        self.summarizer_tokenizer = None
    
    def reset_counter(self):
        self.content_id_counter = 1
    
    def _get_next_id(self) -> int:
        current_id = self.content_id_counter
        self.content_id_counter += 1
        return current_id
    
    def _load_summarizer(self):
        """
        Load the Flan-T5-small model for summarization.
        """
        if self.summarizer is None:
            try:
                # Get the current file's directory and construct the model path
                current_dir = Path(__file__).parent.parent
                model_path = current_dir / "models" / "flan-t5-small"
                if not model_path.exists():
                    raise FileNotFoundError(f"Flan-T5-small model not found at {model_path}")
                
                # Convert to absolute path and ensure it's a string
                model_path_abs = model_path.absolute()
                model_path_str = str(model_path_abs)
                logger.info(f"Model path type: {type(model_path_str)}")
                logger.info(f"Model path value: {repr(model_path_str)}")
                logger.info(f"Loading Flan-T5-small model from {model_path_str}")
                
                # Check if all required files exist
                required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']
                for file_name in required_files:
                    file_path = model_path / file_name
                    if not file_path.exists():
                        raise FileNotFoundError(f"Required model file not found: {file_path}")
                    logger.info(f"Found model file: {file_path}")
                
                # Load tokenizer and model with explicit string path
                # Try without local_files_only first, then with it if needed
                try:
                    logger.info("Attempting to load tokenizer...")
                    self.summarizer_tokenizer = T5Tokenizer.from_pretrained(model_path_str)
                    logger.info("Tokenizer loaded successfully")
                    
                    logger.info("Attempting to load model...")
                    self.summarizer = T5ForConditionalGeneration.from_pretrained(model_path_str)
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load without local_files_only, trying with it: {e}")
                    logger.info("Attempting to load tokenizer with local_files_only=True...")
                    self.summarizer_tokenizer = T5Tokenizer.from_pretrained(model_path_str, local_files_only=True)
                    logger.info("Attempting to load model with local_files_only=True...")
                    self.summarizer = T5ForConditionalGeneration.from_pretrained(model_path_str, local_files_only=True)
                logger.info("Flan-T5-small model loaded successfully")
                
                # Test the model with a simple example
                test_text = "This is a test sentence to verify the summarization model is working correctly."
                test_input = f"Summarize the following text: {test_text}"
                test_inputs = self.summarizer_tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    test_summary_ids = self.summarizer.generate(
                        test_inputs["input_ids"],
                        max_length=50,
                        min_length=10,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True
                    )
                
                test_summary = self.summarizer_tokenizer.decode(test_summary_ids[0], skip_special_tokens=True)
                logger.info(f"Model test successful. Test summary: {test_summary}")
                
            except Exception as e:
                logger.error(f"Error loading summarizer: {e}")
                logger.error(f"Error type: {type(e)}")
                raise
    
    def _summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Summarize the given text using Flan-T5-small model.
        """
        if not text or len(text.strip()) < 30:  # Lowered threshold for shorter text
            logger.debug(f"Skipping summarization for short text (length: {len(text)})")
            return text
        
        try:
            self._load_summarizer()
            
            # Clean the text before summarization
            cleaned_text = self._clean_text_for_summarization(text)
            if not cleaned_text or len(cleaned_text.strip()) < 30:
                return text
            
            # Prepare the input with better prompt
            input_text = f"Summarize this academic text concisely: {cleaned_text}"
            logger.debug(f"Input text length: {len(input_text)}")
            
            inputs = self.summarizer_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            logger.debug(f"Tokenized input shape: {inputs['input_ids'].shape}")
            
            # Generate summary with better parameters
            with torch.no_grad():
                summary_ids = self.summarizer.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=20,  # Lowered minimum length
                    length_penalty=1.5,  # Reduced penalty for more natural summaries
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False  # Deterministic output
                )
            
            # Decode the summary
            summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary = summary.strip()
            
            # If summary is too short or same as input, return original
            if len(summary) < 20 or summary.lower() == cleaned_text.lower()[:len(summary)]:
                return text
            
            logger.debug(f"Generated summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Original text preview: {text[:100]}...")
            # Return original text if summarization fails
            return text
    
    def _clean_text_for_summarization(self, text: str) -> str:
        """
        Clean text specifically for summarization to improve quality.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common PDF artifacts that don't contribute to meaning
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\`\@\#\$\%\&\*\+\=\|\~\\\/]', '', text)
        
        # Remove repeated phrases (common in PDF extraction)
        words = text.split()
        if len(words) > 10:
            # Check for repeated phrases
            for i in range(len(words) - 5):
                phrase = ' '.join(words[i:i+5])
                if text.count(phrase) > 2:  # If phrase appears more than twice
                    # Keep only first occurrence
                    text = text.replace(phrase, '', text.count(phrase) - 1)
        
        return text.strip()

    def process_pdf_headings(self, pdf_content: bytes, filename: str) -> dict:
        try:
            pdf_document = fitz.Document(stream=pdf_content, filetype="pdf")
            logger.info(f"PDF opened with {len(pdf_document)} pages")
            
            # First pass: collect all font sizes and styles across the entire document
            all_font_data = []
            document_structure = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text_dict = page.get_text("dict")
                page_content = {
                    "page": page_num + 1,
                    "text": page.get_text(),
                    "y_positions": [],
                    "font_data": []
                }
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:  # Skip image blocks
                        continue
                        
                    for line in block.get("lines", []):
                        line_text = ""
                        line_fonts = []
                        line_bbox = None
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                                font_size = span.get("size", 0)
                                is_bold = (span.get("flags", 0) & 16) > 0
                                font_name = span.get("font", "")
                                
                                all_font_data.append({
                                    "size": font_size,
                                    "is_bold": is_bold,
                                    "font": font_name,
                                    "text": text
                                })
                                
                                line_fonts.append({
                                    "size": font_size,
                                    "is_bold": is_bold,
                                    "font": font_name
                                })
                                
                                if line_bbox is None:
                                    line_bbox = span["bbox"]
                                else:
                                    # Expand bbox to include this span
                                    line_bbox = [
                                        min(line_bbox[0], span["bbox"][0]),
                                        min(line_bbox[1], span["bbox"][1]),
                                        max(line_bbox[2], span["bbox"][2]),
                                        max(line_bbox[3], span["bbox"][3])
                                    ]
                        
                        line_text = line_text.strip()
                        if line_text and line_bbox:
                            page_content["y_positions"].append({
                                "text": line_text,
                                "y_position": line_bbox[1],
                                "bbox": line_bbox
                            })
                            
                            page_content["font_data"].append({
                                "text": line_text,
                                "bbox": line_bbox,
                                "fonts": line_fonts,
                                "y_position": line_bbox[1]
                            })
                
                document_structure.append(page_content)
            
            # Analyze font patterns to determine base size and heading thresholds
            font_analysis = self._analyze_fonts(all_font_data)
            logger.info(f"Font analysis: {font_analysis}")
            
            # Extract potential headings using rule-based approach
            potential_headings = []
            
            for page_content in document_structure:
                page_headings = self._extract_headings_rule_based(
                    page_content, font_analysis
                )
                potential_headings.extend(page_headings)
            
            # Sort headings by page and y_position
            potential_headings.sort(key=lambda x: (x["page"], x["y_position"]))
            
            # Add logging for debugging
            logger.info(f"Extracted {len(potential_headings)} potential headings before validation")
            
            # Filter and validate headings
            validated_headings = self._validate_headings(potential_headings, document_structure)
            
            # Add logging for debugging
            logger.info(f"Validated {len(validated_headings)} headings")
            
            # Extract content between headings and create final structure
            final_headings = []
            heading_id = 1
            
            for i, heading in enumerate(validated_headings):
                # Extract content from current heading to next heading
                raw_content = self._extract_content_simple(
                    heading, 
                    validated_headings[i + 1] if i + 1 < len(validated_headings) else None,
                    document_structure
                )
                
                # Skip headings with empty content (lowered threshold)
                if not raw_content or len(raw_content.strip()) < 5:
                    logger.debug(f"Skipping heading '{heading['text']}' due to empty or very short content")
                    continue
                
                # Summarize the content
                logger.info(f"Summarizing content for heading '{heading['text']}' (length: {len(raw_content)})")
                summarized_content = self._summarize_text(raw_content)
                logger.info(f"Summarized content length: {len(summarized_content)}")
                
                # Create final heading structure
                final_heading = {
                    "id": heading_id,
                    "text": heading["text"],
                    "page": heading["page"],
                    "bbox": heading["bbox"],
                    "y_position": heading["y_position"],
                    "x_position": heading["x_position"],
                    "confidence": heading["confidence"],
                    "content": summarized_content
                }
                
                final_headings.append(final_heading)
                heading_id += 1
            
            pdf_document.close()
            
            # Create output structure
            headings_data = {
                "filename": filename,
                "ordered_content": final_headings,
                "summary": {
                    "total_items": len(final_headings),
                    "headings": len(final_headings),
                    "original_headings": len(potential_headings),
                    "filtered_out": len(potential_headings) - len(final_headings)
                }
            }
            
            logger.info(f"Processing completed. Found {len(final_headings)} heading items (filtered out {len(potential_headings) - len(final_headings)} headings with empty content)")
            return headings_data
        
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            return {
                "filename": filename,
                "ordered_content": [],
                "summary": {
                    "total_items": 0, 
                    "headings": 0,
                    "original_headings": 0,
                    "filtered_out": 0
                }
            }

    def _analyze_fonts(self, font_data: List[Dict]) -> Dict:
        """
        Analyze font patterns in the document to determine heading characteristics.
        """
        if not font_data:
            return {
                "base_size": 12,
                "heading_threshold": 14,
                "common_fonts": [],
                "bold_threshold": 0.1
            }
        
        # Extract font sizes (use mode/most_common for base_size instead of median)
        sizes = [item["size"] for item in font_data if item["size"] > 0]
        if sizes:
            size_counts = Counter([round(size, 1) for size in sizes])
            # Use most common size as base_size instead of median
            base_size = size_counts.most_common(1)[0][0] if size_counts else statistics.median(sizes)
            # Use 75th percentile for heading_threshold instead of base_size * 1.2
            sorted_sizes = sorted(sizes)
            percentile_75_idx = int(len(sorted_sizes) * 0.75)
            heading_threshold = sorted_sizes[percentile_75_idx] if percentile_75_idx < len(sorted_sizes) else base_size * 1.2
            logger.info(f"Calculated base_size: {base_size}, heading_threshold: {heading_threshold}")
        else:
            base_size = 12
            heading_threshold = 14
        
        # Analyze bold usage
        total_items = len(font_data)
        bold_items = sum(1 for item in font_data if item["is_bold"])
        bold_ratio = bold_items / total_items if total_items > 0 else 0
        
        # Extract common fonts
        fonts = [item["font"] for item in font_data if item["font"]]
        font_counts = Counter(fonts)
        common_fonts = [font for font, count in font_counts.most_common(3)]
        
        return {
            "base_size": base_size,
            "heading_threshold": heading_threshold,
            "common_fonts": common_fonts,
            "bold_ratio": bold_ratio,
            "bold_threshold": min(0.3, bold_ratio * 2)
        }

    def _extract_headings_rule_based(self, page_content: Dict, font_analysis: Dict) -> List[Dict]:
        """
        Extract potential headings using rule-based approach.
        Much more strict filtering to avoid false positives.
        """
        headings = []
        base_size = font_analysis["base_size"]
        heading_threshold = font_analysis["heading_threshold"]
        
        # Sort font_data by y_position to help with merging adjacent lines
        sorted_font_data = sorted(page_content["font_data"], key=lambda x: x["y_position"])
        
        # Calculate page statistics for adaptive scoring
        page_height = max(line["bbox"][3] for line in sorted_font_data) if sorted_font_data else 800
        top_quarter = page_height * 0.25
        bottom_quarter = page_height * 0.75
        
        i = 0
        while i < len(sorted_font_data):
            line_data = sorted_font_data[i]
            text = line_data["text"].strip()
            if not text or len(text) < 2:
                i += 1
                continue
            
            # Skip obvious non-headings immediately
            if self._is_obviously_not_heading(text):
                i += 1
                continue
            
            # Merge with next line if close and similar font (for multi-line headings)
            merged = False
            if i + 1 < len(sorted_font_data):
                next_line = sorted_font_data[i + 1]
                space_between = next_line["y_position"] - line_data["bbox"][3]
                current_height = line_data["bbox"][3] - line_data["bbox"][1]
                if space_between < current_height * 1.2:  # Close vertically
                    # Check similar fonts
                    avg_size_curr = sum(f["size"] for f in line_data["fonts"]) / len(line_data["fonts"])
                    avg_size_next = sum(f["size"] for f in next_line["fonts"]) / len(next_line["fonts"])
                    if abs(avg_size_curr - avg_size_next) < 1 and any(f["is_bold"] for f in line_data["fonts"]) == any(f["is_bold"] for f in next_line["fonts"]):
                        text += " " + next_line["text"].strip()
                        line_data["fonts"].extend(next_line["fonts"])
                        line_data["bbox"] = [
                            min(line_data["bbox"][0], next_line["bbox"][0]),
                            line_data["bbox"][1],
                            max(line_data["bbox"][2], next_line["bbox"][2]),
                            next_line["bbox"][3]
                        ]
                        merged = True
                        i += 1  # Skip next line
            
            # Analyze fonts in this (possibly merged) line
            line_fonts = line_data["fonts"]
            if not line_fonts:
                i += 1
                continue
            
            avg_size = sum(f["size"] for f in line_fonts) / len(line_fonts)
            has_bold = any(f["is_bold"] for f in line_fonts)
            
            # Calculate heading score with much stricter criteria
            score = 0
            confidence_factors = []
            
            # Font size scoring (more strict)
            size_ratio = avg_size / base_size
            if size_ratio > 1.8:
                score += 4
                confidence_factors.append(f"large_font({avg_size:.1f})")
            elif size_ratio > 1.4:
                score += 3
                confidence_factors.append(f"medium_font({avg_size:.1f})")
            elif size_ratio > 1.2:
                score += 2
                confidence_factors.append(f"slightly_larger({avg_size:.1f})")
            
            if has_bold:
                score += 3
                confidence_factors.append("bold")
            
            # Text pattern analysis
            text_score, text_factors = self._analyze_heading_text_patterns(text)
            score += text_score
            confidence_factors.extend(text_factors)
            
            # Length scoring (more strict)
            word_count = len(text.split())
            if word_count <= 8:
                score += 3
                confidence_factors.append("short_length")
            elif word_count <= 15:
                score += 1
                confidence_factors.append("reasonable_length")
            elif word_count > 25:
                score -= 3
                confidence_factors.append("too_long")
            
            # Isolation scoring
            isolation_score = self._check_text_isolation(line_data, sorted_font_data)
            score += isolation_score
            if isolation_score > 0:
                confidence_factors.append("isolated")
            
            # Position-based scoring (more strict)
            y_pos = line_data["y_position"]
            if y_pos < top_quarter:
                score += 1
                confidence_factors.append("top_of_page")
            elif y_pos > bottom_quarter:
                score += 0.5
                confidence_factors.append("bottom_of_page")
            
            # Much stricter validation criteria
            strong_factors = {"large_font", "bold", "numbered_section", "heading_word", "roman_numeral", "short_length"}
            has_strong_factor = any(f.split('(')[0] in strong_factors for f in confidence_factors)
            
            # Require much higher score and at least one strong factor
            min_score = 6 if has_strong_factor else 8
            
            if score >= min_score:
                # Additional validation: check if text looks like a proper heading
                if self._is_valid_heading_content(text):
                    confidence = min(1.0, score / 10.0)
                    headings.append({
                        "text": text,
                        "page": page_content["page"],
                        "bbox": line_data["bbox"],
                        "y_position": line_data["y_position"],
                        "x_position": line_data["bbox"][0],
                        "confidence": confidence,
                        "score": score,
                        "factors": confidence_factors,
                        "font_size": avg_size,
                        "is_bold": has_bold
                    })
            
            if not merged:
                i += 1
            else:
                i += 1  # Already incremented for merged

        return headings

    def _is_obviously_not_heading(self, text: str) -> bool:
        """
        Quick check to filter out obviously non-heading text.
        """
        text_lower = text.lower().strip()
        
        # Author patterns (very strict)
        if re.match(r'^[A-ZŁ][a-zł]+\s+[A-ZŁ][a-zł]+(\s+[A-ZŁ][a-zł]+)*(\s*[∗\*†‡])*$', text):
            return True
        
        # Single letters or numbers
        if len(text) == 1:
            return True
        
        # Page numbers
        if re.match(r'^\d+$', text):
            return True
        
        # Email addresses
        if '@' in text:
            return True
        
        # URLs
        if text.startswith(('http://', 'https://', 'www.')):
            return True
        
        # Copyright notices
        if 'copyright' in text_lower or '©' in text:
            return True
        
        # Date patterns
        if re.match(r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$', text_lower):
            return True
        
        # Reference patterns
        if re.match(r'^\[\d+\]', text):
            return True
        
        # Table data patterns (S1, S2, etc.)
        if re.match(r'^S\d+', text):
            return True
        
        # Single common words
        if len(text.split()) == 1 and text_lower in ['of', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']:
            return True
        
        return False

    def _analyze_heading_text_patterns(self, text: str) -> tuple:
        """
        Analyze text patterns to determine if it looks like a heading.
        Much stricter analysis to avoid false positives.
        """
        score = 0
        factors = []
        text_lower = text.lower().strip()
        
        # Numbered section patterns (strong positive indicators)
        numbered_patterns = [
            r'^\d+\.',           # 1. 2. 3.
            r'^\d+\.\d+',        # 1.1 1.2 2.1
            r'^\d+\.\d+\.\d+',   # 1.1.1 1.1.2
            r'^chapter\s+\d+',   # Chapter 1, Chapter 2
            r'^section\s+\d+',   # Section 1, Section 2
            r'^part\s+\d+',      # Part 1, Part 2
            r'^appendix\s+[a-z]', # Appendix A, Appendix B
        ]
        
        for pattern in numbered_patterns:
            if re.match(pattern, text_lower):
                score += 5  # Strong positive
                factors.append("numbered_section")
                break
        
        # Roman numeral patterns
        if re.match(r'^[IVX]+\.', text_lower):
            score += 4
            factors.append("roman_numeral")
        
        # Letter patterns
        if re.match(r'^[a-z]\.|^[A-Z]\.', text):
            score += 3
            factors.append("letter_section")
        
        # All caps (but not too long and not just numbers/symbols)
        if text.isupper() and len(text) <= 40:
            # Check if it's not just numbers, symbols, or repetitive text
            alphanumeric_ratio = sum(1 for c in text if c.isalnum()) / len(text)
            if alphanumeric_ratio > 0.7:  # Higher threshold
                score += 3
                factors.append("all_caps")
        
        # Title case (more strict)
        if text.istitle() and len(text.split()) <= 10:
            score += 2
            factors.append("title_case")
        
        # Ends with colon
        if text.endswith(':'):
            score += 1
            factors.append("ends_with_colon")
        
        # Common heading words (strong positive indicators)
        heading_words = [
            # Academic/Research
            'introduction', 'conclusion', 'summary', 'abstract', 'overview',
            'background', 'methodology', 'results', 'discussion', 'references',
            'bibliography', 'acknowledgments', 'appendix', 'glossary', 'index',
            'findings', 'method', 'participants', 'data collection', 'literature review',
            'research question', 'hypothesis', 'analysis', 'evaluation',
            
            # Business/Technical
            'executive summary', 'objectives', 'scope', 'approach', 'strategy',
            'implementation', 'recommendations', 'conclusions', 'next steps',
            'challenges', 'opportunities', 'risks', 'benefits', 'costs',
            
            # General
            'overview', 'details', 'specifications', 'requirements', 'features',
            'limitations', 'assumptions', 'definitions', 'terminology'
        ]
        
        for word in heading_words:
            if word in text_lower:
                score += 4  # Strong positive
                factors.append(f"heading_word({word})")
                break
        
        # Strong negative patterns (filter out common false positives)
        negative_patterns = [
            # Email patterns
            (r'@\w+\.\w+', -10, "email_pattern"),
            # Author patterns (very strict)
            (r'^[A-ZŁ][a-zł]+\s+[A-ZŁ][a-zł]+(\s+[A-ZŁ][a-zł]+)*(\s*[∗\*†‡])*$', -8, "author_pattern"),
            # Date patterns
            (r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$', -6, "date_pattern"),
            # Page numbers
            (r'^\d+$', -8, "page_number"),
            # Copyright notices
            (r'copyright|©|all rights reserved', -8, "copyright_notice"),
            # URLs
            (r'https?://|www\.', -8, "url_pattern"),
            # File paths
            (r'[\\/][a-zA-Z0-9_\-\.]+[\\/]', -6, "file_path"),
            # Table data patterns (S1, S2, etc.)
            (r'^S\d+', -8, "table_data"),
            # Reference patterns
            (r'^\[\d+\]', -8, "reference_pattern"),
            # Single letters
            (r'^[A-Za-z]$', -6, "single_letter"),
        ]
        
        for pattern, penalty, factor in negative_patterns:
            if re.search(pattern, text_lower):
                score += penalty
                factors.append(factor)
                break
        
        # Body text patterns (negative scoring)
        if re.search(r'\.\s*[a-z]', text):  # Period followed by lowercase (body text)
            score -= 4
            factors.append("body_text_pattern")
        
        # Scientific notation or mathematical expressions
        if re.search(r'\d+\s*·\s*10\s*\d+|\d+\s*×\s*10\s*\d+', text):
            score -= 6
            factors.append("scientific_notation")
        
        # Incomplete sentences
        if text.endswith((',', ';', '-', '—', '–')):
            score -= 4
            factors.append("incomplete_sentence")
        
        # Single common words (strong negative)
        if len(text.split()) == 1:
            common_words = ['of', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
            if text.lower() in common_words:
                score -= 8
                factors.append("single_common_word")
        
        # Question format (only if it's a proper heading question)
        if text.endswith('?') and len(text.split()) <= 10:
            score += 1
            factors.append("question_format")
        
        # Check for repetitive patterns (strong negative)
        words = text.split()
        if len(words) > 3:
            word_freq = {}
            for word in words:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:  # If any word appears more than 30% of the time
                score -= 6
                factors.append("repetitive_text")
        
        # Check for excessive punctuation
        punctuation_count = sum(1 for c in text if c in '.,;:!?-_()[]{}"\'')
        if punctuation_count > len(text) * 0.3:  # If more than 30% is punctuation
            score -= 4
            factors.append("excessive_punctuation")
        
        return score, factors

    def _check_text_isolation(self, current_line: Dict, all_lines: List[Dict]) -> int:
        """
        Check if a line is spatially isolated (has space above/below).
        Uses adaptive spacing based on document characteristics.
        """
        current_y = current_line["y_position"]
        current_height = current_line["bbox"][3] - current_line["bbox"][1]
        
        # Find lines before and after
        lines_above = [line for line in all_lines if line["y_position"] < current_y]
        lines_below = [line for line in all_lines if line["y_position"] > current_y]
        
        isolation_score = 0
        
        # Calculate average line height for adaptive spacing
        all_heights = [line["bbox"][3] - line["bbox"][1] for line in all_lines]
        avg_line_height = sum(all_heights) / len(all_heights) if all_heights else current_height
        
        # Adaptive spacing multiplier based on document density
        if len(all_lines) > 20:  # Dense document
            spacing_multiplier = 1.2
        elif len(all_lines) > 10:  # Medium density
            spacing_multiplier = 1.5
        else:  # Sparse document
            spacing_multiplier = 2.0
        
        # Check space above
        if lines_above:
            closest_above = max(lines_above, key=lambda x: x["y_position"])
            space_above = current_y - (closest_above["bbox"][3])
            min_space_above = max(current_height, avg_line_height) * spacing_multiplier
            
            if space_above > min_space_above:
                isolation_score += 1
        else:
            isolation_score += 1  # At top of page
        
        # Check space below
        if lines_below:
            closest_below = min(lines_below, key=lambda x: x["y_position"])
            space_below = closest_below["y_position"] - current_line["bbox"][3]
            min_space_below = max(current_height, avg_line_height) * spacing_multiplier
            
            if space_below > min_space_below:
                isolation_score += 1
        else:
            isolation_score += 1  # At bottom of page
        
        return isolation_score

    def _validate_headings(self, headings: List[Dict], document_structure: List[Dict]) -> List[Dict]:
        """
        Validate and filter potential headings to remove false positives.
        """
        validated = []
        
        for heading in headings:
            # Skip if invalid content
            if not self._is_valid_heading_content(heading["text"]):
                logger.debug(f"Filtered invalid content: '{heading['text']}'")
                continue
            
            # Check for duplicates
            is_duplicate = False
            for existing in validated:
                if (existing["text"].lower().strip() == heading["text"].lower().strip() or
                    self._are_similar_headings(existing["text"], heading["text"])):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                logger.debug(f"Filtered duplicate: '{heading['text']}'")
                continue
            
            # Additional validation based on context
            if self._validate_heading_context(heading, document_structure):
                validated.append(heading)
            else:
                logger.debug(f"Filtered based on context: '{heading['text']}'")
        
        return validated

    def _are_similar_headings(self, text1: str, text2: str) -> bool:
        """Check if two headings are very similar."""
        t1_lower = text1.lower().strip()
        t2_lower = text2.lower().strip()
        
        if len(t1_lower) > 10 and len(t2_lower) > 10:
            return t1_lower in t2_lower or t2_lower in t1_lower
        return False

    def _validate_heading_context(self, heading: Dict, document_structure: List[Dict]) -> bool:
        """
        Additional contextual validation for headings.
        Much stricter filtering to eliminate false positives.
        """
        text = heading["text"]
        
        # Filter out very common text that's unlikely to be headings
        common_false_positives = [
            # Page elements
            "page", "figure", "table", "image", "photo", "chart", "graph",
            "see also", "note", "warning", "caution", "tip", "example",
            
            # Journal/Publication metadata
            "volume", "issue", "number", "doi", "issn", "isbn",
            "published", "received", "accepted", "revised",
            
            # Date patterns
            "january", "february", "march", "april", "may", "june", 
            "july", "august", "september", "october", "november", "december",
            
            # Common symbols/patterns
            "∗", "†", "‡", "©", "®", "™",
            
            # File/document metadata
            "filename", "created", "modified", "version", "revision",
            
            # Contact information
            "email", "phone", "fax", "address", "contact",
            
            # Technical metadata
            "url", "http", "www", "com", "org", "edu",
            
            # Table data patterns
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
            "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20",
            
            # Common words that shouldn't be headings alone
            "experience", "study", "research", "data", "analysis", "results",
            "fairly", "experienced", "female", "male", "years", "year",
        ]
        
        text_lower = text.lower()
        for false_pos in common_false_positives:
            if false_pos in text_lower:
                return False
        
        # Filter out text that's mostly numbers or special characters
        alphanumeric_count = sum(1 for c in text if c.isalnum())
        if len(text) > 0 and alphanumeric_count / len(text) < 0.5:  # Stricter threshold
            return False
        
        # Filter out very short text (unless it's a common heading word)
        if len(text.strip()) < 3:
            return False
        
        # Filter out text that's all numbers or mostly numbers
        numeric_count = sum(1 for c in text if c.isdigit())
        if numeric_count > len(text) * 0.6:  # If more than 60% are numbers
            return False
        
        # Filter out text that's all punctuation
        punctuation_chars = '.,;:!?-_()[]{}"\''
        punctuation_count = sum(1 for c in text if c in punctuation_chars)
        if punctuation_count > len(text) * 0.4:  # If more than 40% is punctuation
            return False
        
        # Filter out author name patterns more strictly
        if re.match(r'^[A-ZŁ][a-zł]+\s+[A-ZŁ][a-zł]+(\s+[A-ZŁ][a-zł]+)*(\s*[∗\*†‡])*$', text):
            return False
        
        # Filter out table data patterns
        if re.match(r'^S\d+', text):
            return False
        
        # Filter out reference patterns
        if re.match(r'^\[\d+\]', text):
            return False
        
        # Filter out single letters or numbers
        if len(text) == 1:
            return False
        
        # Allow some exceptions for valid short headings
        short_valid_headings = {
            'abstract', 'introduction', 'conclusion', 'summary', 'overview',
            'background', 'method', 'results', 'discussion', 'references',
            'appendix', 'index', 'glossary', 'acknowledgments', 'findings',
            'methodology', 'literature review', 'research question'
        }
        
        if len(text.split()) == 1 and text_lower in short_valid_headings:
            return True
        
        # Filter out text that looks like table headers or metadata
        if re.search(r'\d+\s*[-–]\s*\d+', text):  # Number ranges
            return False
        
        if re.search(r'^\d+\.\d+', text):  # Decimal numbers
            return False
        
        if re.search(r'^\d+%', text):  # Percentages
            return False
        
        # Filter out repetitive patterns
        words = text.split()
        if len(words) > 2:
            word_freq = {}
            for word in words:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.5:  # If any word appears more than 50% of the time
                return False
        
        return True
    
    def _extract_content_simple(self, current_heading: dict, next_heading: dict, all_text_content: list) -> str:
        """
        Improved content extraction between headings using y-position information.
        Better filtering and cleaning to avoid repetitive content.
        """
        content_lines = []
        current_page = current_heading["page"]
        
        # Get the page content with y-positions
        page_content = next((p for p in all_text_content if p["page"] == current_page), None)
        if not page_content or not page_content.get("y_positions"):
            return ""
        
        # Get all lines with their y-positions
        page_lines = sorted(page_content["y_positions"], key=lambda x: x["y_position"])
        current_heading_y = current_heading["y_position"]
        
        # Find lines that come after the current heading
        for line_data in page_lines:
            line_y = line_data["y_position"]
            line_text = line_data["text"].strip()
            
            if line_y > current_heading_y and line_text:
                # Enhanced filtering for headers/footers and non-content
                if (self._is_header_footer(line_text) or 
                    self._is_reference_citation(line_text) or
                    self._is_table_header(line_text) or
                    self._is_page_number(line_text) or
                    self._is_author_info(line_text) or
                    self._is_metadata_line(line_text) or
                    self._is_table_data(line_text)):
                    continue
                
                # Stop at next heading on same page
                if next_heading and next_heading["page"] == current_page:
                    next_heading_y = next_heading["y_position"]
                    if line_y >= next_heading_y:
                        break
                
                content_lines.append(line_text)
        
        # Handle cross-page content with better filtering
        if next_heading and next_heading["page"] > current_page:
            for page_num in range(current_page + 1, next_heading["page"]):
                page_data = next((p for p in all_text_content if p["page"] == page_num), None)
                if page_data:
                    page_lines = sorted(page_data["y_positions"], key=lambda x: x["y_position"])
                    for line_data in page_lines:
                        line_text = line_data["text"].strip()
                        if line_text and not self._is_header_footer(line_text) and not self._is_page_number(line_text) and not self._is_metadata_line(line_text) and not self._is_table_data(line_text):
                            content_lines.append(line_text)
            
            # Extract from next heading's page up to the heading
            next_page_data = next((p for p in all_text_content if p["page"] == next_heading["page"]), None)
            if next_page_data:
                page_lines = sorted(next_page_data["y_positions"], key=lambda x: x["y_position"])
                next_heading_y = next_heading["y_position"]
                for line_data in page_lines:
                    line_y = line_data["y_position"]
                    line_text = line_data["text"].strip()
                    if line_y >= next_heading_y:
                        break
                    elif line_text and not self._is_header_footer(line_text) and not self._is_page_number(line_text) and not self._is_metadata_line(line_text) and not self._is_table_data(line_text):
                        content_lines.append(line_text)
        
        # Last heading: extract remaining content
        elif not next_heading:
            for page_num in range(current_page + 1, len(all_text_content) + 1):
                page_data = next((p for p in all_text_content if p["page"] == page_num), None)
                if page_data:
                    page_lines = sorted(page_data["y_positions"], key=lambda x: x["y_position"])
                    for line_data in page_lines:
                        line_text = line_data["text"].strip()
                        if line_text and not self._is_header_footer(line_text) and not self._is_page_number(line_text) and not self._is_metadata_line(line_text) and not self._is_table_data(line_text):
                            content_lines.append(line_text)
        
        # Clean and join content with better filtering
        raw_content = " ".join(content_lines)
        
        # Remove excessive whitespace and normalize
        raw_content = re.sub(r'\s+', ' ', raw_content).strip()
        
        # Remove repetitive patterns (common in PDF extraction)
        raw_content = self._remove_repetitive_patterns(raw_content)
        
        # Additional cleaning: remove very short fragments and improve readability
        if len(raw_content) > 50:
            # Split into sentences and filter out very short ones
            sentences = re.split(r'[.!?]+', raw_content)
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15 and not self._is_metadata_line(sentence):
                    filtered_sentences.append(sentence)
            raw_content = '. '.join(filtered_sentences)
        
        # Limit content length to prevent overly long summaries
        if len(raw_content) > 1500:
            raw_content = raw_content[:1500] + "..."
        
        return raw_content if len(raw_content) > 20 else ""

    def _is_metadata_line(self, text: str) -> bool:
        """
        Check if a line contains metadata that should be filtered out.
        """
        text_lower = text.lower()
        
        # Common metadata patterns
        metadata_patterns = [
            r'^\d+\s*·\s*10\s*\d+',  # Scientific notation
            r'^\d+\s*×\s*10\s*\d+',  # Scientific notation with ×
            r'^[A-ZŁ][a-zł]+\s+[A-ZŁ][a-zł]+(\s*[∗\*†‡])*$',  # Author names
            r'^[∗\*†‡]\s*[A-ZŁ][a-zł]+',  # Author affiliations
            r'^©\s*\d{4}',  # Copyright notices
            r'^doi:',  # DOI references
            r'^issn:',  # ISSN references
            r'^isbn:',  # ISBN references
            r'^http[s]?://',  # URLs
            r'^www\.',  # URLs
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check for repetitive patterns
        words = text.split()
        if len(words) > 3:
            word_freq = {}
            for word in words:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.5:  # If any word appears more than 50% of the time
                return True
        
        return False

    def _remove_repetitive_patterns(self, text: str) -> str:
        """
        Remove repetitive patterns that are common in PDF extraction.
        """
        # Remove excessive repetition of words
        words = text.split()
        if len(words) < 10:
            return text
        
        # Find and remove repetitive sequences
        cleaned_words = []
        i = 0
        while i < len(words):
            # Look for repetitive sequences
            if i + 3 < len(words):
                # Check if next 3 words repeat
                sequence = words[i:i+3]
                repeat_count = 0
                j = i + 3
                while j + 2 < len(words) and words[j:j+3] == sequence:
                    repeat_count += 1
                    j += 3
                
                if repeat_count > 0:
                    # Keep only one instance of the sequence
                    cleaned_words.extend(sequence)
                    i = j
                    continue
            
            cleaned_words.append(words[i])
            i += 1
        
        return " ".join(cleaned_words)
    
    def _is_header_footer(self, text: str) -> bool:
        """Check if text is a header or footer."""
        text_lower = text.lower()
        header_footer_patterns = [
            'eurocall review', 'page', 'figure', 'table', 'appendix',
            'references', 'bibliography', 'acknowledgments'
        ]
        return any(pattern in text_lower for pattern in header_footer_patterns)
    
    def _is_reference_citation(self, text: str) -> bool:
        """Check if text is a reference citation."""
        return bool(re.match(r'^[A-Z][a-z]+,?\s+[A-Z]\.\s*\(\d{4}\)', text))
    
    def _is_table_header(self, text: str) -> bool:
        """Check if text is a table header."""
        return bool(re.match(r'^[A-Z][a-z]*\s*/\s*[A-Z][a-z]*$', text))
    
    def _is_page_number(self, text: str) -> bool:
        """Check if text is a page number."""
        return bool(re.match(r'^\d+$', text))
    
    def _is_author_info(self, text: str) -> bool:
        """Check if text is author information."""
        return bool(re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*[∗\*]?$', text) or '@' in text)
            
    def _is_valid_heading_content(self, text: str) -> bool:
        """
        Validate if text content is suitable for a heading.
        Filters out images, tables, and weird text patterns.
        """
        text = text.strip()
        
        if len(text) < 2:
            return False
        
        # Filter out single words that are common articles/prepositions
        if len(text.split()) == 1 and text.lower() in ['of', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']:
            return False
        
        # Filter out text that ends with incomplete sentences (likely body text fragments)
        if text.endswith((',', ';', '-', '—', '–')):
            return False
        
        # Filter out text that contains sentence fragments (period followed by lowercase)
        if re.search(r'\.\s*[a-z]', text):
            return False
        
        # Filter out email addresses
        if '@' in text and '.' in text:
            return False
        
        # Enhanced author pattern filter (including special characters and asterisks)
        if re.match(r'^[A-ZŁ][a-zł]+\s+[A-ZŁ][a-zł]+\s*[∗\*]?$', text):
            return False
        
        # Filter out reference citations (author names with years)
        if re.match(r'^[A-Z][a-z]+,?\s+[A-Z]\.\s*\(\d{4}\)', text):
            return False
        
        # Filter out full reference entries (bracketed numbers followed by text)
        if re.match(r'^\[\d+\]\s+[A-Z][a-z]+', text):
            return False
        
        # Filter out table headers that are just column names
        if re.match(r'^[A-Z][a-z]*\s*/\s*[A-Z][a-z]*$', text):
            return False
        
        # Filter out scientific notation and table data
        if re.match(r'^[\d\.,\s%$€£¥·×]+$', text):  # Numbers, punctuation, currency, scientific notation
            return False
        
        # Filter out text that looks like table data with scientific notation
        if re.search(r'\d+\s*·\s*10\s*\d+', text):  # Pattern like "1.0 · 10 20"
            return False
        
        # Filter out image/table indicators (expanded)
        image_table_indicators = [
            'figure', 'table', 'image', 'photo', 'graph', 'chart',
            'fig.', 'tab.', 'img.', 'pic.', 'eurocall review'  # PDF-specific footer
        ]
        text_lower = text.lower()
        for indicator in image_table_indicators:
            if indicator in text_lower:
                return False
        
        # Filter out weird text patterns
        weird_patterns = [
            r'^_+$',  # Only underscores
            r'^[_\-\=\.]+$',  # Only special characters
            r'^[0-9\s]+$',  # Only numbers and spaces
            r'^[^\w\s]+$',  # Only non-alphanumeric characters
            r'^[A-Z\s]+$',  # All caps with only spaces (likely table headers)
        ]
        
        for pattern in weird_patterns:
            if re.match(pattern, text):
                return False
        
        # Filter out very long text (likely paragraphs)
        if len(text) > 100:
            return False
            
        # Filter out text that's mostly special characters
        alphanumeric_ratio = len(re.findall(r'[a-zA-Z0-9]', text)) / len(text)
        if alphanumeric_ratio < 0.3:  # Less than 30% alphanumeric
            return False
        
        # Filter out text that looks like body text (contains multiple sentences or complex punctuation)
        if text.count('.') > 2 or text.count(',') > 3:
            return False
            
        return True

    def _is_table_data(self, text: str) -> bool:
        """
        Check if a line contains table data that should be filtered out.
        """
        text_lower = text.lower().strip()
        
        # Table data patterns
        table_patterns = [
            r'^S\d+',  # S1, S2, etc.
            r'^\d+\s+[A-Za-z]+',  # Number followed by text
            r'^[A-Za-z]+\s+\d+',  # Text followed by number
            r'^\d+\s*[-–]\s*\d+',  # Number ranges
            r'^[A-Za-z]+\s*[-–]\s*[A-Za-z]+',  # Text ranges
            r'^\d+\.\d+',  # Decimal numbers
            r'^\d+%',  # Percentages
            r'^\d+\s*[A-Za-z]+$',  # Number + unit
        ]
        
        for pattern in table_patterns:
            if re.match(pattern, text):
                return True
        
        # Check for repetitive patterns typical in tables
        words = text.split()
        if len(words) > 2:
            word_freq = {}
            for word in words:
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.4:  # If any word appears more than 40% of the time
                return True
        
        return False

    def extract_pdf_text_for_chatbot(self, filename: str) -> dict:
        """
        Extract all text from a PDF file and clean it for LLM consumption.
        If the text is more than 500,000 words, create a summary using Flan-T5-small.
        """
        try:
            # Construct the PDF file path
            current_dir = Path(__file__).parent.parent
            pdf_path = current_dir / "uploads" / filename
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file '{filename}' not found at {pdf_path}")
            
            logger.info(f"Extracting text from PDF: {filename}")
            
            # Open the PDF document
            pdf_document = fitz.Document(str(pdf_path))
            logger.info(f"PDF opened with {len(pdf_document)} pages")
            
            # Extract all text from the document
            all_text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                all_text += page_text + "\n"
            
            pdf_document.close()
            
            # Clean the extracted text
            cleaned_text = self._clean_text_for_llm(all_text)
            
            # Count words
            word_count = len(cleaned_text.split())
            logger.info(f"Extracted text has {word_count} words")
            
            # Check if summarization is needed (500,000 words threshold)
            if word_count > 500000:
                logger.info(f"Text exceeds 500,000 words ({word_count}), creating summary...")
                summarized_text = self._summarize_large_text(cleaned_text)
                summary_word_count = len(summarized_text.split())
                
                return {
                    "filename": filename,
                    "original_word_count": word_count,
                    "summary_word_count": summary_word_count,
                    "text": summarized_text,
                    "is_summarized": True,
                    "status": "success"
                }
            else:
                logger.info(f"Text is within limits ({word_count} words), returning as-is")
                return {
                    "filename": filename,
                    "original_word_count": word_count,
                    "summary_word_count": word_count,
                    "text": cleaned_text,
                    "is_summarized": False,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filename}: {str(e)}")
            return {
                "filename": filename,
                "original_word_count": 0,
                "summary_word_count": 0,
                "text": "",
                "is_summarized": False,
                "status": "error",
                "error_message": str(e)
            }
    
    def _clean_text_for_llm(self, text: str) -> str:
        """
        Clean extracted text for optimal LLM consumption.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove multiple consecutive line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = re.sub(r'\n\s+', '\n', text)  # Remove leading spaces after line breaks
        text = re.sub(r'\s+\n', '\n', text)  # Remove trailing spaces before line breaks
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)  # Page X of Y
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)  # - X - format
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\`\@\#\$\%\&\*\+\=\|\~\\\/]', '', text)  # Keep only readable characters
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes to regular quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes to regular apostrophes
        text = text.replace('–', '-').replace('—', '-')  # Em/en dashes to regular dash
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'\!{2,}', '!', text)  # Multiple exclamation marks
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([\.\,\;\:\!\?])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'([\.\,\;\:\!\?])\s*([\.\,\;\:\!\?])', r'\1\2', text)  # Fix consecutive punctuation
        
        # Remove empty lines at the beginning and end
        text = text.strip()
        
        # Ensure reasonable line length (break very long lines)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line) > 200:  # Break lines longer than 200 characters
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) > 200:
                        if current_line:
                            cleaned_lines.append(current_line.strip())
                            current_line = word
                        else:
                            cleaned_lines.append(word)
                    else:
                        current_line += (" " + word) if current_line else word
                if current_line:
                    cleaned_lines.append(current_line.strip())
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _summarize_large_text(self, text: str) -> str:
        """
        Summarize large text using Flan-T5-small model.
        """
        try:
            self._load_summarizer()
            
            # Split text into chunks if it's too long for the model
            max_chunk_length = 4000  # Conservative chunk size for T5
            chunks = self._split_text_into_chunks(text, max_chunk_length)
            
            if len(chunks) == 1:
                # Single chunk, summarize directly
                return self._summarize_text(text, max_length=1000)
            else:
                # Multiple chunks, summarize each and then combine
                logger.info(f"Text split into {len(chunks)} chunks for summarization")
                chunk_summaries = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
                    chunk_summary = self._summarize_text(chunk, max_length=500)
                    chunk_summaries.append(chunk_summary)
                
                # Combine chunk summaries
                combined_summary = " ".join(chunk_summaries)
                
                # If the combined summary is still too long, summarize it again
                if len(combined_summary.split()) > 10000:
                    logger.info("Combined summary is still too long, creating final summary")
                    final_summary = self._summarize_text(combined_summary, max_length=2000)
                    return final_summary
                else:
                    return combined_summary
                    
        except Exception as e:
            logger.error(f"Error summarizing large text: {e}")
            # Return a truncated version if summarization fails
            words = text.split()
            if len(words) > 10000:
                return " ".join(words[:10000]) + "... [Text truncated due to summarization error]"
            else:
                return text
    
    def _split_text_into_chunks(self, text: str, max_chunk_length: int) -> List[str]:
        """
        Split text into chunks of approximately max_chunk_length characters,
        trying to break at sentence boundaries.
        """
        if len(text) <= max_chunk_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) <= max_chunk_length:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk + " " + word) <= max_chunk_length:
                            current_chunk += (" " + word) if current_chunk else word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = word
                            else:
                                chunks.append(word)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Global processor instance
pdf_processor = PDFProcessor()