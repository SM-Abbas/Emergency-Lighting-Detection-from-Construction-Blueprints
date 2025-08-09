import os
import cv2
import numpy as np
import json
from PIL import Image
import shutil
from app import EmergencyLightingDetector
from analyze_symbols import detect_shapes
import random
from pathlib import Path

# Create necessary directories
def create_directory_structure():
    # Create main directories
    data_dir = os.path.join(os.getcwd(), 'data')
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Create class subdirectories
    class_names = ['A1E', 'A1-E', 'A1/E', 'EM-1', 'EM-2', 'EXIT-EM', 'EL', 'UNKNOWN', 'NOT_EMERGENCY']
    for class_name in class_names:
        os.makedirs(os.path.join(images_dir, class_name), exist_ok=True)
    
    return images_dir, annotations_dir

# Extract potential emergency lighting symbols from blueprint images
def extract_symbols_from_blueprints(blueprint_dir, output_dir, detector):
    # Get all image files in the blueprint directory
    image_files = [f for f in os.listdir(blueprint_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(blueprint_dir, f))]
    
    if not image_files:
        print(f"No image files found in {blueprint_dir}")
        return []
    
    extracted_symbols = []
    
    for image_file in image_files:
        print(f"Processing {image_file}...")
        image_path = os.path.join(blueprint_dir, image_file)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect potential symbols using shape detection
        potential_symbols = detect_shapes(gray)
        
        # Process each potential symbol
        for i, symbol_info in enumerate(potential_symbols):
            bbox = symbol_info['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Extract the region of interest
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # Save the extracted symbol
            symbol_filename = f"{os.path.splitext(image_file)[0]}_symbol_{i}.png"
            symbol_path = os.path.join(output_dir, 'UNKNOWN', symbol_filename)
            cv2.imwrite(symbol_path, roi)
            
            # Add to the list of extracted symbols
            extracted_symbols.append({
                'image_file': symbol_filename,
                'source_image': image_file,
                'bbox': [x, y, w, h],
                'area': symbol_info['area'],
                'aspect_ratio': symbol_info['aspect_ratio'],
                'path': symbol_path
            })
    
    return extracted_symbols

# Use the existing detector to label extracted symbols
def label_extracted_symbols(extracted_symbols, detector, images_dir, annotations_dir):
    labeled_symbols = []
    
    for symbol_info in extracted_symbols:
        # Read the symbol image
        symbol_path = symbol_info['path']
        symbol_image = cv2.imread(symbol_path)
        
        if symbol_image is None:
            continue
        
        # Convert to grayscale for text extraction
        gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
        
        # Try to extract text from the symbol
        try:
            # Use the detector's text extraction method if available
            if hasattr(detector, 'extract_text_from_roi'):
                text = detector.extract_text_from_roi(gray)
            else:
                # Fallback to a simple approach
                text = []
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            text = []
        
        # Determine the symbol type based on text or shape
        if text and hasattr(detector, 'identify_symbol'):
            symbol_type = detector.identify_symbol(text)
        else:
            # Use aspect ratio to guess the symbol type
            aspect_ratio = symbol_info['aspect_ratio']
            if 0.8 < aspect_ratio < 1.2:  # Square-ish shape
                symbol_type = 'EL'  # Emergency Light
            elif aspect_ratio < 0.5:  # Tall and narrow
                symbol_type = 'EXIT-EM'  # Exit sign with emergency backup
            elif aspect_ratio > 2.0:  # Wide and short
                symbol_type = 'A1E'  # Exit/Emergency Combo Unit
            else:  # Other rectangular shapes
                symbol_type = 'EM-1'  # Default to Emergency Type 1 Fixture
        
        # Move the symbol to the appropriate class directory
        if symbol_type in ['A1E', 'A1-E', 'A1/E', 'EM-1', 'EM-2', 'EXIT-EM', 'EL']:
            dest_dir = os.path.join(images_dir, symbol_type)
        else:
            dest_dir = os.path.join(images_dir, 'UNKNOWN')
        
        # Create a new filename
        new_filename = f"{symbol_type}_{os.path.basename(symbol_path)}"
        new_path = os.path.join(dest_dir, new_filename)
        
        # Copy the file to the new location
        shutil.copy(symbol_path, new_path)
        
        # Update the symbol info
        symbol_info['label'] = symbol_type
        symbol_info['text'] = text
        symbol_info['new_path'] = new_path
        
        labeled_symbols.append(symbol_info)
    
    return labeled_symbols

# Generate negative examples (non-emergency lighting fixtures)
def generate_negative_examples(blueprint_dir, images_dir, num_samples=100):
    # Get all image files in the blueprint directory
    image_files = [f for f in os.listdir(blueprint_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(blueprint_dir, f))]
    
    if not image_files:
        print(f"No image files found in {blueprint_dir}")
        return
    
    negative_dir = os.path.join(images_dir, 'NOT_EMERGENCY')
    os.makedirs(negative_dir, exist_ok=True)
    
    count = 0
    for image_file in image_files:
        if count >= num_samples:
            break
        
        image_path = os.path.join(blueprint_dir, image_file)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Get random regions from the image
        h, w = image.shape[:2]
        for _ in range(5):  # Try to get 5 samples from each image
            if count >= num_samples:
                break
            
            # Random region size
            region_w = random.randint(30, 100)
            region_h = random.randint(30, 100)
            
            # Random position
            x = random.randint(0, max(0, w - region_w - 1))
            y = random.randint(0, max(0, h - region_h - 1))
            
            # Extract region
            region = image[y:y+region_h, x:x+region_w]
            
            # Save as negative example
            negative_filename = f"negative_{image_file}_{count}.png"
            negative_path = os.path.join(negative_dir, negative_filename)
            cv2.imwrite(negative_path, region)
            
            count += 1
    
    print(f"Generated {count} negative examples")

# Create annotation files for training
def create_annotations(labeled_symbols, annotations_dir):
    # Group symbols by source image
    symbols_by_source = {}
    for symbol in labeled_symbols:
        source = symbol['source_image']
        if source not in symbols_by_source:
            symbols_by_source[source] = []
        symbols_by_source[source].append(symbol)
    
    # Create annotation file for each source image
    for source, symbols in symbols_by_source.items():
        annotation_data = {
            'image_file': source,
            'annotations': []
        }
        
        for symbol in symbols:
            annotation_data['annotations'].append({
                'bbox': symbol['bbox'],
                'label': symbol['label'],
                'text': symbol['text'] if 'text' in symbol else []
            })
        
        # Save annotation file
        annotation_file = os.path.join(annotations_dir, f"{os.path.splitext(source)[0]}_annotations.json")
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)

# Augment the training data to increase dataset size
def augment_training_data(images_dir):
    for class_dir in os.listdir(images_dir):
        class_path = os.path.join(images_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            # Apply various augmentations
            # 1. Rotation
            for angle in [90, 180, 270]:
                rotated = rotate_image(image, angle)
                aug_filename = f"aug_rot{angle}_{image_file}"
                cv2.imwrite(os.path.join(class_path, aug_filename), rotated)
            
            # 2. Flip
            flipped_h = cv2.flip(image, 1)  # Horizontal flip
            cv2.imwrite(os.path.join(class_path, f"aug_fliph_{image_file}"), flipped_h)
            
            # 3. Brightness/contrast adjustment
            brightened = adjust_brightness_contrast(image, alpha=1.2, beta=10)
            cv2.imwrite(os.path.join(class_path, f"aug_bright_{image_file}"), brightened)
            
            darkened = adjust_brightness_contrast(image, alpha=0.8, beta=-10)
            cv2.imwrite(os.path.join(class_path, f"aug_dark_{image_file}"), darkened)

# Helper function for rotation
def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

# Helper function for brightness/contrast adjustment
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Main function
def main():
    # Create directory structure
    images_dir, annotations_dir = create_directory_structure()
    
    # Initialize the detector
    detector = EmergencyLightingDetector()
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Extract symbols from blueprint images
    blueprint_dir = current_dir  # Assuming blueprint images are in the current directory
    extracted_symbols = extract_symbols_from_blueprints(blueprint_dir, images_dir, detector)
    
    if not extracted_symbols:
        print("No symbols were extracted. Check if there are blueprint images in the directory.")
        return
    
    print(f"Extracted {len(extracted_symbols)} potential symbols")
    
    # Label the extracted symbols
    labeled_symbols = label_extracted_symbols(extracted_symbols, detector, images_dir, annotations_dir)
    
    print(f"Labeled {len(labeled_symbols)} symbols")
    
    # Generate negative examples
    generate_negative_examples(blueprint_dir, images_dir)
    
    # Create annotation files
    create_annotations(labeled_symbols, annotations_dir)
    
    # Augment the training data
    print("Augmenting training data...")
    augment_training_data(images_dir)
    
    print("Training data preparation complete!")
    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")

if __name__ == "__main__":
    main()