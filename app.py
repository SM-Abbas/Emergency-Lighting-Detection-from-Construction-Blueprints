from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import json
import uuid
import threading
import time
import io
import re
import sqlite3
from datetime import datetime
from sqlite3 import Connection, connect
from pathlib import Path
import cv2
import numpy as np
import pytesseract

# Check if Tesseract is installed, otherwise use a fallback approach
import os
import shutil
import logging

# First check if tesseract is in PATH
TESSERACT_AVAILABLE = shutil.which('tesseract') is not None
if not TESSERACT_AVAILABLE:
    # Then check the default installation path
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = os.path.exists(TESSERACT_PATH)
    if TESSERACT_AVAILABLE:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        # Try other common installation paths
        alt_paths = [
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'),
            os.path.join(os.environ.get('PROGRAMFILES', ''), 'Tesseract-OCR', 'tesseract.exe')
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                TESSERACT_AVAILABLE = True
                pytesseract.pytesseract.tesseract_cmd = path
                break

# Log Tesseract availability
if not TESSERACT_AVAILABLE:
    print("WARNING: Tesseract OCR is not available. Text extraction features will be limited.")
else:
    print("Tesseract OCR is available and configured.")
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import openai
from dotenv import load_dotenv
import logging

local_storage = threading.local()

def get_db_connection() -> Connection:
    if not hasattr(local_storage, 'connection'):
        local_storage.connection = connect('emergency_lighting.db')
    return local_storage.connection

# Load environment variables
load_dotenv()

# Validate environment variables on startup
if not os.getenv('UPLOAD_FOLDER'):
    raise EnvironmentError("Missing UPLOAD_FOLDER in environment")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ANNOTATIONS_FOLDER'] = 'annotations'

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER'], app.config['ANNOTATIONS_FOLDER'], 'static', 'templates']:
    os.makedirs(folder, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key (you'll need to set this in your environment)
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.png'):
        return jsonify({'error': 'Only PNG files are allowed'}), 400

    try:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        image = Image.open(filepath)
        detector = EmergencyLightingDetector()
        
        # Check if Tesseract is available and warn the user if it's not
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract OCR is not installed. Text extraction will be limited.")
        
        # Detect emergency lights
        detections = detector.detect_emergency_lights(image, filename)
        
        # Extract static content
        static_content = detector.extract_static_content([{'image': image, 'sheet_name': filename}])
        
        # Combine results
        results = {
            'detections': detections,
            'rulebook': static_content.get('rulebook', [])
        }

        # Save results to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO extracted_data (pdf_name, data_type, content, source_sheet) VALUES (?, ?, ?, ?)',
            (filename, 'detection_results', json.dumps(results), filename)
        )
        conn.commit()

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# Database setup
def init_db():
    conn = sqlite3.connect('emergency_lighting.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_jobs (
            id TEXT PRIMARY KEY,
            pdf_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            result TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT NOT NULL,
            data_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_sheet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

class EmergencyLightingDetector:
    def __init__(self):
        self.emergency_symbols = ['A1E', 'EM', 'EXIT', 'EMERGENCY', 'E', 'EX', 'A1-E', 'A1/E', 'EM-1', 'EM-2', 'EXIT-EM', 'EL']
        self.fixture_types = {
            'A1E': 'Exit/Emergency Combo Unit',
            'A1': '2x4 LED Emergency Fixture',
            'W': 'Wall-Mounted Emergency LED',
            'WP': 'Wallpack with Built-in Photocell',
            'E': 'Emergency Light',
            'EX': 'Exit Sign',
            '2X4': '2x4 RECESSED LED LUMINAIRE',
            'A1-E': 'Type A1 Emergency Fixture',
            'A1/E': 'Type A1 with Emergency Battery Backup',
            'EM-1': 'Emergency Type 1 Fixture',
            'EM-2': 'Emergency Type 2 Fixture',
            'EXIT-EM': 'Exit Sign with Emergency Backup',
            'EL': 'Emergency Light'
        }
        
        # Additional keywords that might indicate emergency lighting
        self.emergency_keywords = [
            'EM', 'EMERGENCY', 'EXIT', 'UNSWITCHED', 'BATTERY', 
            'BACKUP', 'EGRESS', 'EVACUATION', 'SAFETY'
        ]
        
        # Define keywords for identifying notes and tables
        self.note_keywords = [
            'NOTE', 'GENERAL NOTE', 'LIGHTING NOTE', 'EMERGENCY LIGHTING', 
            'REQUIREMENT', 'CODE', 'SPECIFICATION'
        ]
        
        self.table_keywords = [
            'SCHEDULE', 'LIGHTING SCHEDULE', 'FIXTURE SCHEDULE', 'SYMBOL', 
            'TYPE', 'DESCRIPTION', 'MANUFACTURER', 'CATALOG', 'MOUNT', 'VOLTAGE', 'LUMENS'
        ]
    
    def pdf_to_images(self, pdf_path):
        """Convert PDF pages to images"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2, 2)  # Scale factor for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            images.append({
                'image': img,
                'page_num': page_num,
                'sheet_name': f"Sheet_{page_num + 1}"
            })
        
        doc.close()
        return images
    
    def detect_emergency_lights(self, image, sheet_name):
        """Detect emergency lighting fixtures in the image"""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        # Method 1: Detect shaded rectangular areas (emergency lights)
        # Use faster thresholding method
        _, thresh_regular = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Skip adaptive thresholding and morphological operations for speed
        thresh_cleaned = thresh_regular
        
        # Find contours for shaded areas - use RETR_LIST for faster processing
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Limit the number of contours to process for speed
        max_contours = 50
        if len(contours) > max_contours:
            # Sort contours by area (largest first) and take only the top ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Narrower area range to focus on most likely fixtures
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it's rectangular (emergency light fixture)
                if 0.2 < aspect_ratio < 5.0:  # Wider aspect ratio range
                    # Calculate a smaller region of interest for faster text extraction
                    text_roi_y_min = max(0, y - 20)  # Reduced from 50 to 20
                    text_roi_y_max = min(gray.shape[0], y + h + 20)  # Reduced from 50 to 20
                    text_roi_x_min = max(0, x - 20)  # Reduced from 50 to 20
                    text_roi_x_max = min(gray.shape[1], x + w + 20)  # Reduced from 50 to 20
                    
                    # Extract text near the fixture
                    roi = gray[text_roi_y_min:text_roi_y_max, text_roi_x_min:text_roi_x_max]
                    
                    # Skip OCR for very small ROIs to save processing time
                    roi_area = (text_roi_y_max - text_roi_y_min) * (text_roi_x_max - text_roi_x_min)
                    if roi_area < 100:  # Skip tiny ROIs
                        continue
                    
                    # Only use OCR if Tesseract is available
                    if TESSERACT_AVAILABLE:
                        try:
                            # Use a faster OCR configuration
                            text_nearby = self.extract_text_from_roi(roi, fast_mode=True)
                            # Check if it's an emergency light based on nearby text
                            if self.is_emergency_fixture(text_nearby):
                                symbol = self.identify_symbol(text_nearby)
                                confidence = 0.7 + (0.2 * (1 if symbol != 'UNKNOWN' else 0))
                                
                                detections.append({
                                    'symbol': symbol,
                                    'bounding_box': [x, y, x+w, y+h],
                                    'text_nearby': text_nearby,
                                    'source_sheet': sheet_name,
                                    'confidence': confidence
                                })
                        except Exception as e:
                            logger.error(f"Error in OCR processing: {str(e)}")
                            # Fall back to shape-based detection on OCR error, but assign a specific symbol
                            # based on aspect ratio to avoid 'UNKNOWN' labels
                            if 0.8 < aspect_ratio < 1.2:  # Square-ish shape
                                symbol = 'EL'  # Emergency Light
                            elif aspect_ratio < 0.5:  # Tall and narrow
                                symbol = 'EXIT-EM'  # Exit sign with emergency backup
                            elif aspect_ratio > 2.0:  # Wide and short
                                symbol = 'A1E'  # Exit/Emergency Combo Unit
                            else:  # Other rectangular shapes
                                symbol = 'EM-1'  # Default to Emergency Type 1 Fixture
                                
                            detections.append({
                                'symbol': symbol,
                                'bounding_box': [x, y, x+w, y+h],
                                'text_nearby': [],
                                'source_sheet': sheet_name,
                                'confidence': 0.6,
                                'detection_method': 'shape_based_fallback'
                            })
                    else:
                        # If Tesseract is not available, use shape-based detection but assign a specific symbol
                        # based on aspect ratio to avoid 'UNKNOWN' labels
                        if 0.8 < aspect_ratio < 1.2:  # Square-ish shape
                            symbol = 'EL'  # Emergency Light
                        elif aspect_ratio < 0.5:  # Tall and narrow
                            symbol = 'EXIT-EM'  # Exit sign with emergency backup
                        elif aspect_ratio > 2.0:  # Wide and short
                            symbol = 'A1E'  # Exit/Emergency Combo Unit
                        else:  # Other rectangular shapes
                            symbol = 'EM-1'  # Default to Emergency Type 1 Fixture
                            
                        detections.append({
                            'symbol': symbol,
                            'bounding_box': [x, y, x+w, y+h],
                            'text_nearby': [],
                            'source_sheet': sheet_name,
                            'confidence': 0.6,
                            'detection_method': 'shape_based'
                        })
        
        # Method 2: Look for symbols and then check if they're emergency fixtures
        # This helps catch fixtures that might not be shaded but are marked with symbols
        # Only run this method if Tesseract is available
        if TESSERACT_AVAILABLE:
            try:
                # Use Tesseract to find all text in the image
                h, w = gray.shape
                # Process the image in larger sections for faster processing
                section_size = 2000  # Process in 2000x2000 sections (larger sections = fewer iterations)
                
                # Only process the center section of the image where symbols are most likely to be
                center_y = max(0, (h // 2) - section_size)
                center_x = max(0, (w // 2) - section_size)
                
                # Process just one large section in the center of the image
                for y_start in [center_y]:
                    for x_start in [center_x]:
                        y_end = min(y_start + section_size, h)
                        x_end = min(x_start + section_size, w)
                        
                        section = gray[y_start:y_end, x_start:x_end]
                        if section.size == 0:  # Skip empty sections
                            continue
                            
                        # Skip processing very large sections to save time
                        if section.shape[0] * section.shape[1] > 4000000:  # Skip sections larger than 2000x2000
                            continue
                            
                        # Use faster text extraction with minimal processing
                        try:
                            # Use a simpler and faster OCR approach
                            section_text = pytesseract.image_to_string(section, config='--psm 11 --oem 0')
                            words = [word.strip().upper() for word in section_text.split() if word.strip()]
                            
                            # Process only the first few words to save time
                            max_words = 20  # Limit word processing
                            for i, word in enumerate(words[:max_words]):
                                if not word:  # Skip empty words
                                    continue
                                    
                                # Check if this word is a potential emergency symbol
                                if word in self.emergency_symbols or any(keyword in word for keyword in self.emergency_keywords):
                                    # Since we're using image_to_string instead of image_to_data,
                                    # we don't have bounding box information for individual words.
                                    # Instead, we'll use a region around the center of the section
                                    
                                    # Calculate an approximate position for this word
                                    # Use the center of the section as an approximation
                                    x = x_start + section.shape[1] // 2 - 50
                                    y = y_start + section.shape[0] // 2 - 50
                                    w = 100  # Use a fixed width
                                    h = 100  # Use a fixed height
                                    
                                    # Skip context extraction to save processing time
                                    context_text = []  # Skip additional OCR to speed up processing
                                    
                                    # If this is an emergency fixture, add it to detections
                                    if self.is_emergency_fixture([word] + context_text):
                                        symbol = self.identify_symbol([word] + context_text)
                                        
                                        # Check if this detection overlaps with existing ones
                                        new_bbox = [x, y, x+w, y+h]
                                        if not any(self._bboxes_overlap(new_bbox, d['bounding_box']) for d in detections):
                                            detections.append({
                                                'symbol': symbol,
                                                'bounding_box': new_bbox,
                                                'text_nearby': [word] + context_text,
                                                'source_sheet': sheet_name,
                                                'confidence': 0.75,
                                                'detection_method': 'text_based'
                                            })
                        except Exception as e:
                            logger.error(f"Error in section OCR processing: {str(e)}")
                            # Continue with next section on error
                            continue
            except Exception as e:
                logger.error(f"Error in text-based detection: {str(e)}")
                # Continue with shape-based detections only
                pass
        else:
            logger.warning("Tesseract OCR is not available. Using shape-based detection only.")
        
        # If no detections were found and Tesseract is not available, try to detect based on shape patterns only
        if len(detections) == 0:
            logger.info("No detections found with primary methods. Trying shape-based detection only.")
            # Find potential emergency light fixtures based on shape characteristics
            # Look for rectangular shapes with specific aspect ratios
            for contour in contours:
                area = cv2.contourArea(contour)
                if 30 < area < 15000:  # Wider area range for fallback
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Emergency lights and exit signs often have these aspect ratios
                    if (0.2 < aspect_ratio < 0.5) or (2.0 < aspect_ratio < 5.0) or (0.8 < aspect_ratio < 1.2):
                        # Assign a specific symbol based on aspect ratio to avoid 'UNKNOWN' labels
                        if 0.8 < aspect_ratio < 1.2:  # Square-ish shape
                            symbol = 'EL'  # Emergency Light
                        elif aspect_ratio < 0.5:  # Tall and narrow
                            symbol = 'EXIT-EM'  # Exit sign with emergency backup
                        elif aspect_ratio > 2.0:  # Wide and short
                            symbol = 'A1E'  # Exit/Emergency Combo Unit
                        else:  # Other rectangular shapes
                            symbol = 'EM-1'  # Default to Emergency Type 1 Fixture
                            
                        detections.append({
                            'symbol': symbol,
                            'bounding_box': [x, y, x+w, y+h],
                            'text_nearby': [],
                            'source_sheet': sheet_name,
                            'confidence': 0.5,  # Lower confidence for shape-only detection
                            'detection_method': 'fallback_shape_only'
                        })
        
        return detections
        
    def _bboxes_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap significantly"""
        # Convert to [x1, y1, x2, y2] format
        box1 = [bbox1[0], bbox1[1], bbox1[2], bbox1[3]]
        box2 = [bbox2[0], bbox2[1], bbox2[2], bbox2[3]]
        
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return False  # No overlap
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU (Intersection over Union)
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou > threshold
    
    def extract_text_from_roi(self, roi, fast_mode=False):
        """Extract text from a region of interest with enhanced processing
        
        Args:
            roi: Region of interest image
            fast_mode: If True, use faster processing with minimal enhancement
        """
        if roi is None or roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return []
            
        # Check if Tesseract is available
        if not TESSERACT_AVAILABLE:
            # If Tesseract is not available, return empty list
            # This will be handled by the calling method
            logger.warning("Tesseract OCR is not available. Skipping text extraction.")
            return []
            
        try:
            # In fast mode, skip enhancement to save processing time
            if fast_mode:
                enhanced = roi  # Use original image without enhancement
            else:
                # Apply contrast enhancement (only in normal mode)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi)
            
            words = []
            
            try:
                # Use fastest PSM mode in fast mode
                config = '--psm 11 --oem 0' if fast_mode else '--psm 11'
                text = pytesseract.image_to_string(enhanced, config=config)
                words.extend([word.strip().upper() for word in text.split() if word.strip()])
            except Exception as e:
                logger.warning(f"Error in text extraction: {str(e)}")
            
            # Skip rotation attempts to speed up processing
            # We already have words, no need to try rotations
            
            # The following code was removed to improve performance:
            # Rotation code that tried different angles (90, 180, 270)
            # This significantly speeds up processing
            
            # Remove duplicates while preserving order
            unique_words = []
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)
            
            return unique_words
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            return []
    
    def is_emergency_fixture(self, text_list):
        """Check if the text indicates an emergency fixture"""
        if not text_list:
            return False
            
        text_str = ' '.join(text_list).upper()
        
        # First check for negative indicators
        negative_indicators = ['NOT EMERGENCY', 'REGULAR', 'STANDARD', 'NO EMERGENCY']
        for indicator in negative_indicators:
            if indicator in text_str:
                return False
        
        # Check for exact emergency symbol matches
        for symbol in self.emergency_symbols:
            if symbol in text_str.split():
                return True
        
        # Check for emergency symbols in the text
        for symbol in self.emergency_symbols:
            if symbol in text_str and len(symbol) > 1:  # Avoid single letter matches like 'E'
                return True
                
        # Check for emergency keywords - must be exact matches to avoid false positives
        for keyword in self.emergency_keywords:
            if keyword in text_str.split():
                return True
                
        # Check for patterns like "A1-E" or "A1/E"
        if any(re.search(r'[A-Z][0-9][-/]?E', word.upper()) for word in text_list):
            return True
            
        # Check for specific emergency-related phrases
        emergency_phrases = ['EMERGENCY LIGHT', 'EXIT SIGN', 'BATTERY BACKUP']
        for phrase in emergency_phrases:
            if phrase in text_str:
                return True
                
        return False
    
    def identify_symbol(self, text_list):
        """Identify the fixture symbol from nearby text"""
        if not text_list:
            return 'UNKNOWN'
            
        text_str = ' '.join(text_list).upper()
        
        # First check if this is actually an emergency fixture
        if not self.is_emergency_fixture(text_list):
            return 'UNKNOWN'
        
        # Special case for A1-E and A1/E as first word
        if text_list and text_list[0] in ['A1-E', 'A1/E']:
            return text_list[0]
        
        # Check for exact symbol matches
        for symbol in self.fixture_types.keys():
            if symbol in text_str.split():
                return symbol
        
        # Check for specific patterns - order matters for correct identification
        # First check for exact matches of the specific symbols
        for word in text_list:
            word = word.upper()
            if word == 'A1E':
                return 'A1E'
            elif word == 'A1-E':
                return 'A1-E'
            elif word == 'A1/E':
                return 'A1/E'
            elif word == 'EM-1':
                return 'EM-1'
            elif word == 'EM-2':
                return 'EM-2'
            elif word == 'EXIT-EM':
                return 'EXIT-EM'
            elif word == 'EL':
                return 'EL'
        
        # Then check for patterns in the full text
        if 'A1E' in text_str or ('A1' in text_str and 'EXIT' in text_str):
            return 'A1E'
        elif 'A1-E' in text_str.split() or re.search(r'\bA1-E\b', text_str) or (text_list[0].upper() == 'A1-E'):
            return 'A1-E'
        elif 'A1/E' in text_str.split() or re.search(r'\bA1/E\b', text_str) or (text_list[0].upper() == 'A1/E'):
            return 'A1/E'
        elif 'EM-1' in text_str.split() or re.search(r'\bEM-1\b', text_str) or (text_list[0].upper() == 'EM-1'):
            return 'EM-1'
        elif 'EM-2' in text_str.split() or re.search(r'\bEM-2\b', text_str) or (text_list[0].upper() == 'EM-2'):
            return 'EM-2'
        elif 'EXIT-EM' in text_str.split() or re.search(r'\bEXIT-EM\b', text_str) or (text_list[0].upper() == 'EXIT-EM'):
            return 'EXIT-EM'
        elif 'EL' in text_str.split() or (text_list[0].upper() == 'EL'):
            return 'EL'
        elif 'A1' in text_str.split() and self.is_emergency_fixture(text_list):
            return 'A1'
        elif re.search(r'2[\s]*[X\*]\s*4', text_str) or '2X4' in text_str:
            return '2X4'
        elif 'EXIT' in text_str and 'SIGN' in text_str:
            return 'EX'
        elif ('W' in text_str.split() and 'WALL' in text_str) or 'WALL-MOUNTED' in text_str:
            return 'W'
        elif 'WP' in text_str or 'WALLPACK' in text_str or 'PHOTOCELL' in text_str:
            return 'WP'
        elif 'EM' in text_str.split() and 'EMERGENCY' in text_str:
            return 'E'
        
        # Check for pattern matches like A1-E, A1/E, EM-1, EM-2, etc.
        for word in text_list:
            # Check for A1-E or A1/E pattern
            match_a1e = re.search(r'([A-Z][0-9])[-/]E', word)
            if match_a1e:
                base_symbol = match_a1e.group(1)
                separator = '-' if '-' in word else '/'
                return f"{base_symbol}{separator}E"
            
            # Check for EM-1 or EM-2 pattern
            match_em = re.search(r'(EM)[-]([12])', word)
            if match_em:
                return f"EM-{match_em.group(2)}"
            
            # Check for EXIT-EM pattern
            if re.search(r'EXIT[-]EM', word):
                return 'EXIT-EM'
            
            # Check for EL pattern (Emergency Light)
            if word == 'EL':
                return 'EL'
        
        return 'UNKNOWN'
    
    def extract_static_content(self, images):
        """Extract static content like notes and schedules using advanced methods"""
        rulebook = []
        
        # First, try to extract from PDF directly if available
        try:
            pdf_path = os.getenv('CURRENT_PDF_PATH')
            if pdf_path and os.path.exists(pdf_path):
                # Extract from PDF directly using PyMuPDF
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    sheet_name = f"Sheet_{page_num + 1}"
                    
                    # Extract text for notes
                    text = page.get_text()
                    notes = self._extract_notes_from_text(text, sheet_name)
                    rulebook.extend(notes)
                    
                    # Extract tables using PyMuPDF's improved table detection
                    try:
                        tables = page.find_tables()
                        if tables and tables.tables:
                            # Process each table found in the page
                            for table in tables.tables:
                                # Check if this looks like a lighting schedule
                                header_cells = []
                                for i in range(min(table.header_count, table.cols)):
                                    cell = tables.cells[i] if i < len(tables.cells) else None
                                    if cell:
                                        header_cells.append(cell.text.upper())
                                
                                header_text = ' '.join(header_cells)
                                if any(keyword in header_text for keyword in self.table_keywords):
                                    # This is a lighting-related table, extract its rows
                                    table_rows = self._extract_table_rows(tables, sheet_name)
                                    rulebook.extend(table_rows)
                    except Exception as table_err:
                        logger.warning(f"PyMuPDF table detection error: {str(table_err)}")
                        # Try alternative table detection method if available
                        # Only proceed if Tesseract is available
                        if TESSERACT_AVAILABLE:
                            try:
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                                alt_table_rows = self._detect_tables_in_image(gray_img, text, sheet_name)
                                rulebook.extend(alt_table_rows)
                            except Exception as ocr_err:
                                logger.warning(f"OCR-based table detection error: {str(ocr_err)}")
                        else:
                            # Skip table detection if Tesseract is not available
                            logger.warning("Tesseract OCR is not available. Skipping OCR-based table detection.")
                            continue
                
                doc.close()
                return {'rulebook': rulebook}
        except Exception as e:
            logger.warning(f"Could not extract from PDF directly: {str(e)}. Falling back to image-based extraction.")
        
        # Fall back to image-based extraction if PDF extraction fails
        for img_data in images:
            image = img_data['image']
            sheet_name = img_data['sheet_name']
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding before CLAHE
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Apply CLAHE for better text extraction
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Extract text and detect tables only if Tesseract is available
            if TESSERACT_AVAILABLE:
                try:
                    # Extract all text from the image with multiple PSM modes
                    text1 = pytesseract.image_to_string(enhanced, config='--psm 6')
                    text2 = pytesseract.image_to_string(enhanced, config='--psm 3')
                    text = text1 + "\n" + text2  # Combine results
                    
                    # Extract notes
                    notes = self._extract_notes_from_text(text, sheet_name)
                    rulebook.extend(notes)
                    
                    # Try to detect tables in the image
                    table_rows = self._detect_tables_in_image(enhanced, text, sheet_name)
                    rulebook.extend(table_rows)
                except Exception as ocr_err:
                    logger.error(f"Error in OCR-based extraction: {str(ocr_err)}")
                    # Add a basic note about OCR failure
                    rulebook.append({
                        'type': 'note',
                        'content': 'OCR processing failed. Some text content may not be available.',
                        'source_sheet': sheet_name
                    })
            else:
                # If Tesseract is not available, add a note about limited functionality
                logger.warning("Tesseract OCR is not available. Skipping text and table extraction from images.")
                rulebook.append({
                    'type': 'note',
                    'content': 'OCR functionality is limited because Tesseract is not installed.',
                    'source_sheet': sheet_name
                })
                
                # Try to extract basic visual features without OCR
                # This is a simplified approach that doesn't rely on text recognition
                try:
                    # Look for table-like structures based on lines and grid patterns
                    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
                    
                    if lines is not None and len(lines) > 10:  # If we detect many lines, it might be a table
                        rulebook.append({
                            'type': 'note',
                            'content': 'Potential table or structured content detected, but text extraction is limited without OCR.',
                            'source_sheet': sheet_name
                        })
                except Exception as vis_err:
                    logger.error(f"Error in visual feature extraction: {str(vis_err)}")
        
        return {'rulebook': rulebook}
    
    def _extract_notes_from_text(self, text, sheet_name=None):
        """Extract notes from text content"""
        notes = []
        lines = text.split('\n')
        in_note_section = False
        current_note = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a note section
            if any(keyword in line.upper() for keyword in self.note_keywords) and not in_note_section:
                in_note_section = True
                current_note = line
            # If we're in a note section, add lines until we hit something that looks like the end
            elif in_note_section:
                # Check if this line might be the end of the note section
                if len(line) < 3 or any(keyword in line.upper() for keyword in self.table_keywords):
                    # End of note, add it to our collection
                    if current_note:
                        notes.append({
                            'type': 'note',
                            'content': current_note.strip(),
                            'source_sheet': sheet_name
                        })
                    in_note_section = False
                    current_note = ""
                else:
                    # Continue the note
                    current_note += " " + line
        
        # Add the last note if we're still in a note section
        if in_note_section and current_note:
            notes.append({
                'type': 'note',
                'content': current_note.strip(),
                'source_sheet': sheet_name
            })
            
        return notes
    
    def _map_column_to_field(self, header: str) -> str:
        """Maps table headers to standardized field names"""
        header = header.lower()
        
        # Map common column headers to standard field names
        if any(term in header for term in ['symbol', 'type', 'fixture', 'mark']):
            return 'symbol'
        elif any(term in header for term in ['desc', 'description', 'name']):
            return 'description'
        elif any(term in header for term in ['mount', 'mounting', 'location']):
            return 'mount_type'
        elif any(term in header for term in ['volt', 'voltage', 'v']):
            return 'voltage'
        elif any(term in header for term in ['lumen', 'output', 'brightness']):
            return 'lumens'
        elif any(term in header for term in ['watt', 'power', 'w']):
            return 'watts'
        elif any(term in header for term in ['manufacturer', 'mfr', 'brand']):
            return 'manufacturer'
        elif any(term in header for term in ['model', 'catalog', 'part']):
            return 'model'
        
        # If no match, return the original header as the field name
        return header
    
    def _extract_table_rows(self, tables, sheet_name):
        """Extract rows from PyMuPDF table object"""
        table_rows = []
        
        # Process each table in the page
        for table_idx, table in enumerate(tables.tables):
            # Get header row for column mapping
            header_row = []
            for col_idx in range(table.cols):
                if col_idx < len(tables.cells):
                    cell_text = tables.cells[col_idx].text.strip()
                    header_row.append(cell_text)
            
            # Check if this is a lighting schedule table
            is_lighting_table = any(keyword in ' '.join(header_row).upper() 
                                   for keyword in self.table_keywords)
            
            if not is_lighting_table:
                continue
            
            # Process data rows
            for row_idx in range(1, table.rows):  # Skip header row
                row_data = {'type': 'table_row'}
                
                for col_idx in range(table.cols):
                    cell_idx = row_idx * table.cols + col_idx
                    if cell_idx < len(tables.cells):
                        header = header_row[col_idx] if col_idx < len(header_row) else f"Column {col_idx}"
                        cell_text = tables.cells[cell_idx].text.strip()
                        
                        # Map columns to standard fields
                        field_name = self._map_column_to_field(header)
                        if field_name and cell_text:
                            row_data[field_name] = cell_text
                
                # Add source sheet information
                if row_data and len(row_data) > 1:  # Ensure we have more than just the type
                    row_data['source_sheet'] = sheet_name
                    table_rows.append(row_data)
        
        return table_rows
    
    def _detect_tables_in_image(self, image, text, sheet_name):
        """Detect tables in images using text patterns"""
        # If Tesseract is not available, we can't extract text from images
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract OCR is not available. Skipping table detection from images.")
            # Try to detect tables using visual features instead
            try:
                # Look for table-like structures based on lines and grid patterns
                edges = cv2.Canny(image, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
                
                if lines is not None and len(lines) > 10:  # If we detect many lines, it might be a table
                    # Return a placeholder table row to indicate a table was detected
                    return [{
                        'type': 'table_row',
                        'source_sheet': sheet_name,
                        'note': 'Table detected visually, but text extraction is limited without OCR.',
                        'detection_method': 'visual_only'
                    }]
            except Exception as e:
                logger.error(f"Error in visual table detection: {str(e)}")
            # Return empty list if visual detection fails
            return []
            
        table_rows = []
        
        try:
            # Look for schedule tables in text
            if 'schedule' in text.lower() or 'symbol' in text.lower():
                lines = text.split('\n')
                table_lines = [line.strip() for line in lines if line.strip() and len(line.split()) > 2]
                
                # Try to identify table header
                header_line = None
                for line in table_lines:
                    if any(keyword in line.upper() for keyword in self.table_keywords):
                        header_line = line
                        break
                        
                if header_line:
                    # Process lines after header as table rows
                    in_table = False
                    for line in table_lines:
                        if line == header_line:
                            in_table = True
                            continue
                        
                        if in_table:
                            parts = line.split()
                            if len(parts) >= 2:
                                # First part is likely the symbol
                                symbol = parts[0]
                                description = ' '.join(parts[1:])
                                
                                # Try to extract additional information
                                mount_type = ''
                                voltage = ''
                                lumens = ''
                                
                                # Look for specific patterns in the description
                                if 'MOUNT' in description.upper():
                                    mount_parts = description.split('MOUNT', 1)
                                    if len(mount_parts) > 1:
                                        mount_type = mount_parts[1].strip()
                                
                                if 'VOLT' in description.upper():
                                    volt_parts = description.split('VOLT', 1)
                                    if len(volt_parts) > 1:
                                        voltage = volt_parts[1].strip()
                                
                                if 'LUMEN' in description.upper():
                                    lumen_parts = description.split('LUMEN', 1)
                                    if len(lumen_parts) > 1:
                                        lumens = lumen_parts[1].strip()
                                
                                table_rows.append({
                                    'type': 'table_row',
                                    'symbol': symbol,
                                    'description': description,
                                    'mount': mount_type,
                                    'voltage': voltage,
                                    'lumens': lumens,
                                    'source_sheet': sheet_name
                                })
        except Exception as e:
            logger.error(f"Error in table text analysis: {str(e)}")
            return []
        
        return table_rows
        table_rows = []
        
        # Look for schedule tables in text
        if 'schedule' in text.lower() or 'symbol' in text.lower():
            lines = text.split('\n')
            table_lines = [line.strip() for line in lines if line.strip() and len(line.split()) > 2]
            
            # Try to identify table header
            header_line = None
            for line in table_lines:
                if any(keyword in line.upper() for keyword in self.table_keywords):
                    header_line = line
                    break
            
            if header_line:
                # Process lines after header as table rows
                in_table = False
                for line in table_lines:
                    if line == header_line:
                        in_table = True
                        continue
                    
                    if in_table:
                        parts = line.split()
                        if len(parts) >= 2:
                            # First part is likely the symbol
                            symbol = parts[0]
                            description = ' '.join(parts[1:])
                            
                            # Try to extract additional information
                            mount_type = ''
                            voltage = ''
                            lumens = ''
                            
                            # Look for specific patterns in the description
                            if 'MOUNT' in description.upper():
                                mount_parts = description.split('MOUNT', 1)
                                if len(mount_parts) > 1:
                                    mount_type = mount_parts[1].strip()
                            
                            if 'VOLT' in description.upper():
                                volt_parts = description.split('VOLT', 1)
                                if len(volt_parts) > 1:
                                    voltage = volt_parts[1].strip()
                            
                            if 'LUMEN' in description.upper():
                                lumen_parts = description.split('LUMEN', 1)
                                if len(lumen_parts) > 1:
                                    lumens = lumen_parts[1].strip()
                            
                            table_rows.append({
                                'type': 'table_row',
                                'symbol': symbol,
                                'description': description,
                                'mount': mount_type,
                                'voltage': voltage,
                                'lumens': lumens,
                                'source_sheet': sheet_name
                            })
        
        return table_rows
    
    def group_lighting_fixtures(self, detections, rulebook):
        """Group lighting fixtures by type and count them"""
        summary = {}
        symbol_counts = {}
        fixture_descriptions = {}
        
        # First pass: collect all unique descriptions for each symbol
        for detection in detections:
            symbol = detection['symbol']
            if symbol not in fixture_descriptions:
                fixture_descriptions[symbol] = set()
            
            # Get description from fixture types or rulebook
            description = self.get_fixture_description(detection)
            fixture_descriptions[symbol].add(description)
            
            # Count detections by symbol
            if symbol in symbol_counts:
                symbol_counts[symbol] += 1
            else:
                symbol_counts[symbol] = 1
        
        # Create enhanced summary with better categorization
        light_counter = 1
        for symbol, count in symbol_counts.items():
            # Get the most descriptive name for this symbol
            if symbol in fixture_descriptions and fixture_descriptions[symbol]:
                description = max(fixture_descriptions[symbol], key=len)  # Use the longest description
            else:
                description = self.fixture_types.get(symbol, f"Unknown Fixture Type ({symbol})")
            
            # For unknown symbols, try to create more specific descriptions
            if symbol == 'UNKNOWN':
                # Check if we can determine a better description from the emergency keywords
                for detection in detections:
                    if detection['symbol'] == symbol:
                        text_nearby = ' '.join(detection.get('text_nearby', [])).upper()
                        if 'EXIT' in text_nearby:
                            description = 'Exit/Emergency Combo Unit'
                            break
                        elif 'WALL' in text_nearby:
                            description = 'Wall-Mounted Emergency LED'
                            break
                        elif any(keyword in text_nearby for keyword in self.emergency_keywords):
                            description = 'Emergency Lighting Fixture'
                            break
            
            summary[f"Lights{light_counter:02d}"] = {
                'count': count,
                'description': description,
                'symbol': symbol
            }
            light_counter += 1
        
        return {'summary': summary}
        
    def get_fixture_description(self, fixture):
        """Get a detailed description for a fixture based on its symbol, nearby text, and characteristics"""
        symbol = fixture.get('symbol', 'UNKNOWN')
        text_nearby = fixture.get('text_nearby', [])
        
        # Join nearby text for easier searching
        text = ' '.join(text_nearby).upper() if text_nearby else ''
        
        # Check if we have a known fixture type
        if symbol in self.fixture_types:
            base_description = self.fixture_types[symbol]
            
            # Enhance description with emergency status if not already included
            if ('EMERGENCY' not in base_description.upper() and 'EM' not in base_description.upper()) and \
               (any(keyword in text for keyword in self.emergency_keywords) or \
                any(sym in symbol for sym in self.emergency_symbols)):
                return f"Emergency {base_description}"
            return base_description
        
        # Try to determine from symbol pattern
        if re.search(r'[A-Z0-9]+-?E', symbol) or 'EX' in symbol:
            if 'EXIT' in text:
                return 'Exit/Emergency Combo Unit'
            elif '2X4' in text or '2X4' in symbol:
                return '2x4 LED Emergency Fixture'
            else:
                return 'Emergency Lighting Fixture'
        
        # Try to determine from nearby text
        if 'EXIT' in text and ('EMERGENCY' in text or 'EM' in text):
            return 'Exit/Emergency Combo Unit'
        elif 'EXIT' in text:
            return 'Exit Sign'
        elif 'EMERGENCY' in text or 'EM' in text:
            if 'WALL' in text or 'PHOTOCELL' in text:
                return 'Wall-Mounted Emergency LED'
            elif '2X4' in text:
                return '2x4 LED Emergency Fixture'
            else:
                return 'Emergency Lighting Fixture'
        elif '2X4' in text:
            return '2x4 LED Fixture'
        elif 'WALL' in text or 'PHOTOCELL' in text:
            return 'Wall-Mounted Fixture'
        
        # If symbol contains numbers and letters, it might be a specific fixture code
        if re.match(r'^[A-Z][0-9]', symbol):
            return f"Fixture Type {symbol}"
        
        return "Unknown Fixture"

def process_pdf_background(job_id, pdf_path, pdf_name):
    """Background processing function"""
    try:
        # Update job status
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE processing_jobs SET status = ? WHERE id = ?', ('processing', job_id))
        conn.commit()
        
        # Use the stored PDF path from environment if available
        if 'CURRENT_PDF_PATH' in os.environ and os.environ['CURRENT_PDF_PATH']:
            filepath = os.environ['CURRENT_PDF_PATH']
            # Clear the environment variable after use
            os.environ['CURRENT_PDF_PATH'] = ''
        
        # Initialize detector
        detector = EmergencyLightingDetector()
        
        # Convert PDF to images
        images = detector.pdf_to_images(pdf_path)
        
        # Detect emergency lights
        all_detections = []
        for img_data in images:
            detections = detector.detect_emergency_lights(img_data['image'], img_data['sheet_name'])
            all_detections.extend(detections)
        
        # Extract static content
        static_content = detector.extract_static_content(images)
        
        # Group fixtures
        grouped_result = detector.group_lighting_fixtures(all_detections, static_content['rulebook'])
        
        # Store extracted data in database
        conn = sqlite3.connect('emergency_lighting.db')
        cursor = conn.cursor()
        
        # Store rulebook data (static content)
        for item in static_content['rulebook']:
            cursor.execute('''
                INSERT INTO extracted_data (pdf_name, data_type, content, source_sheet)
                VALUES (?, ?, ?, ?)
            ''', (pdf_name, item['type'], json.dumps(item), item.get('source_sheet', '')))
        
        # Store detection results
        for detection in all_detections:
            cursor.execute('''
                INSERT INTO extracted_data (pdf_name, data_type, content, source_sheet)
                VALUES (?, ?, ?, ?)
            ''', (pdf_name, 'detection', json.dumps(detection), detection['source_sheet']))
        
        # Update job with final result including summary of fixtures
        cursor.execute('''
            UPDATE processing_jobs 
            SET status = ?, result = ?, completed_at = ? 
            WHERE id = ?
        ''', ('complete', json.dumps(grouped_result['summary']), datetime.now(), job_id))
        
        conn.commit()
        conn.close()
        
        # Create annotation image
        create_annotation_image(images[0]['image'], all_detections, pdf_name)
        
        logger.info(f"Processing completed for job {job_id}, static content extracted and stored")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        # Update job status to error
        conn = sqlite3.connect('emergency_lighting.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE processing_jobs SET status = ? WHERE id = ?', ('error', job_id))
        conn.commit()
        conn.close()

def create_annotation_image(image, detections, pdf_name):
    """Create annotated image showing detected emergency lights"""
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Draw bounding boxes and labels
    for detection in detections:
        bbox = detection['bounding_box']
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw label
        label = f"{detection['symbol']} ({detection.get('confidence', 0.8):.2f})"
        draw.text((x1, y1-20), label, fill='red')
    
    # Save annotated image
    annotation_path = os.path.join(app.config['ANNOTATIONS_FOLDER'], f"{pdf_name}_annotated.png")
    annotated_image.save(annotation_path)

@app.route('/blueprints/upload', methods=['POST'])
def upload_blueprint():
    """Upload PDF and trigger background processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store the PDF path in environment for static content extraction
        os.environ['CURRENT_PDF_PATH'] = filepath
        
        # Create processing job
        job_id = str(uuid.uuid4())
        
        conn = sqlite3.connect('emergency_lighting.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO processing_jobs (id, pdf_name, status)
            VALUES (?, ?, ?)
        ''', (job_id, filename, 'uploaded'))
        conn.commit()
        conn.close()
        
        # Start background processing
        thread = threading.Thread(
            target=process_pdf_background,
            args=(job_id, filepath, filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'uploaded',
            'pdf_name': filename,
            'job_id': job_id,
            'message': 'Processing started in background.'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/blueprints/result', methods=['GET'])
def get_result():
    """Get processed result for a PDF"""
    try:
        pdf_name = request.args.get('pdf_name')
        if not pdf_name:
            return jsonify({'error': 'pdf_name parameter is required'}), 400
        
        conn = sqlite3.connect('emergency_lighting.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT status, result FROM processing_jobs 
            WHERE pdf_name = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (pdf_name,))
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'error': 'PDF not found'}), 404
        
        status, result_data = result
        
        if status == 'complete':
            # Get static content (rulebook) if available
            cursor.execute('''
                SELECT content FROM extracted_data 
                WHERE pdf_name = ? AND data_type IN ('note', 'table_row')
            ''', (pdf_name,))
            
            static_content = [json.loads(row[0]) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({
                'pdf_name': pdf_name,
                'status': 'complete',
                'result': json.loads(result_data) if result_data else {},
                'static_content': static_content
            })
        elif status == 'error':
            conn.close()
            return jsonify({
                'pdf_name': pdf_name,
                'status': 'error',
                'message': 'Processing failed. Please try again.'
            })
        else:
            conn.close()
            return jsonify({
                'pdf_name': pdf_name,
                'status': 'in_progress',
                'message': 'Processing is still in progress. Please try again later.'
            })
            
    except Exception as e:
        logger.error(f"Result retrieval error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/blueprints/annotation/<pdf_name>', methods=['GET'])
def get_annotation(pdf_name):
    """Get annotation image for a PDF"""
    try:
        annotation_path = os.path.join(app.config['ANNOTATIONS_FOLDER'], f"{pdf_name}_annotated.png")
        if os.path.exists(annotation_path):
            return send_file(annotation_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Annotation not found'}), 404
    except Exception as e:
        logger.error(f"Annotation retrieval error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    import io
    app.run(host='0.0.0.0', port=5000, debug=False)