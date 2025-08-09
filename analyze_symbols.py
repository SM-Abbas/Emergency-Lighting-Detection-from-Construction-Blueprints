import cv2
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Function to detect shapes in an image that might be emergency lighting symbols
def detect_shapes(image):
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape
    potential_symbols = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:  # Filter by area
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if it's rectangular (emergency light fixture)
            if 0.2 < aspect_ratio < 5.0:
                potential_symbols.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
    
    return potential_symbols

# Function to detect if an image contains a legend or schedule based on visual features
def detect_legend_or_schedule(image):
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Look for table-like structures based on lines and grid patterns
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    has_tables = lines is not None and len(lines) > 10
    
    # Check for grid-like patterns (common in legends and schedules)
    horizontal_lines = 0
    vertical_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                horizontal_lines += 1
            else:  # Vertical line
                vertical_lines += 1
    
    # If we have both horizontal and vertical lines, it might be a grid/table
    has_grid = horizontal_lines > 3 and vertical_lines > 3
    
    return has_tables, has_grid

# Function to analyze a PDF page for emergency lighting symbols using visual features
def analyze_page(image_path):
    print(f"Analyzing {image_path}...")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Detect potential symbols
    potential_symbols = detect_shapes(image)
    
    # Detect if the image contains a legend or schedule
    has_tables, has_grid = detect_legend_or_schedule(image)
    
    # Determine if this is likely a legend or schedule page
    # Legends and schedules typically have many symbols and grid-like structures
    is_legend = has_grid and len(potential_symbols) > 5
    is_schedule = has_tables and has_grid
    
    # Check if this page might contain emergency lighting information
    # Emergency lighting symbols often have specific aspect ratios
    contains_emergency = False
    emergency_symbol_count = 0
    
    for symbol in potential_symbols:
        # Emergency exit signs often have aspect ratios around 2:1 to 3:1
        if 1.8 < symbol['aspect_ratio'] < 3.2:
            emergency_symbol_count += 1
        # Emergency lights often have aspect ratios around 1:1 to 1:2
        elif 0.5 < symbol['aspect_ratio'] < 1.2:
            emergency_symbol_count += 1
    
    # If we found multiple potential emergency symbols, mark this page
    if emergency_symbol_count >= 3:
        contains_emergency = True
    
    # Create a visualization of the detected symbols
    visualization = image.copy()
    for symbol in potential_symbols:
        x, y, x2, y2 = symbol['bbox']
        # Draw rectangle around the symbol
        cv2.rectangle(visualization, (x, y), (x2, y2), (0, 255, 0), 2)
    
    # Save the visualization
    vis_path = image_path.replace('.png', '_analysis.png')
    cv2.imwrite(vis_path, visualization)
    
    return {
        'image_path': image_path,
        'visualization_path': vis_path,
        'is_legend': is_legend,
        'is_schedule': is_schedule,
        'contains_emergency': contains_emergency,
        'has_tables': has_tables,
        'has_grid': has_grid,
        'potential_symbols': len(potential_symbols),
        'emergency_symbol_count': emergency_symbol_count
    }

# Function to extract emergency lighting symbols from the PDF
def extract_emergency_symbols():
    # Analyze all PDF pages
    results = []
    
    # Get all PDF page images
    pdf_pages = [f for f in os.listdir() if f.startswith('pdf_page') and f.endswith('.png')]
    pdf_pages.sort(key=lambda x: int(x.replace('pdf_page', '').replace('.png', '')))
    
    for page in pdf_pages:
        result = analyze_page(page)
        if result:
            results.append(result)
    
    # Save results to a JSON file
    with open('pdf_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Analysis complete. Results saved to pdf_analysis.json")
    
    # Print summary
    print("\nSummary:")
    legend_pages = [r['image_path'] for r in results if r['is_legend']]
    schedule_pages = [r['image_path'] for r in results if r['is_schedule']]
    emergency_pages = [r['image_path'] for r in results if r['contains_emergency']]
    
    print(f"Legend pages: {legend_pages}")
    print(f"Schedule pages: {schedule_pages}")
    print(f"Pages with emergency lighting info: {emergency_pages}")
    
    # Extract new emergency symbols and fixture types
    new_emergency_symbols = []
    new_fixture_types = {}
    
    # Check Legend.png and Lighting Fixture.png for symbols
    special_images = ['Legend.png', 'Lighting Fixture.png']
    for img_path in special_images:
        if os.path.exists(img_path):
            print(f"\nAnalyzing special image: {img_path}")
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Detect potential symbols
            potential_symbols = detect_shapes(image)
            print(f"Found {len(potential_symbols)} potential symbols in {img_path}")
            
            # Create a visualization
            visualization = image.copy()
            for i, symbol in enumerate(potential_symbols):
                x, y, x2, y2 = symbol['bbox']
                # Draw rectangle and label
                cv2.rectangle(visualization, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(visualization, f"Symbol {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save the visualization
            vis_path = img_path.replace('.png', '_analysis.png')
            cv2.imwrite(vis_path, visualization)
            print(f"Visualization saved as {vis_path}")
    
    # Update app.py with new emergency symbols and fixture types
    update_emergency_detector(new_emergency_symbols, new_fixture_types)
    
    return results

# Function to update the EmergencyLightingDetector class in app.py
def update_emergency_detector(new_symbols, new_fixture_types):
    # Read the current app.py file
    app_py_path = 'app.py'
    if not os.path.exists(app_py_path):
        print(f"Error: {app_py_path} not found")
        return
    
    with open(app_py_path, 'r') as f:
        app_py_content = f.read()
    
    # Find the EmergencyLightingDetector class initialization
    detector_init_start = app_py_content.find('class EmergencyLightingDetector:')
    if detector_init_start == -1:
        print("Error: Could not find EmergencyLightingDetector class in app.py")
        return
    
    # Find the emergency_symbols and fixture_types definitions
    symbols_start = app_py_content.find('self.emergency_symbols = [', detector_init_start)
    symbols_end = app_py_content.find(']', symbols_start) + 1
    
    fixture_types_start = app_py_content.find('self.fixture_types = {', detector_init_start)
    fixture_types_end = app_py_content.find('}', fixture_types_start) + 1
    
    if symbols_start == -1 or fixture_types_start == -1:
        print("Error: Could not find emergency_symbols or fixture_types in app.py")
        return
    
    # Extract current symbols and fixture types
    current_symbols_str = app_py_content[symbols_start:symbols_end]
    current_fixture_types_str = app_py_content[fixture_types_start:fixture_types_end]
    
    print("\nCurrent emergency symbols and fixture types:")
    print(current_symbols_str)
    print(current_fixture_types_str)
    
    # For now, we'll just print the current values since we don't have text recognition
    # In a real implementation, we would update these with new symbols from the PDF
    print("\nBased on visual analysis, we've identified potential emergency lighting symbols.")
    print("To fully train the model with new symbols, manual annotation of the symbols would be required.")
    print("The visualizations created (*_analysis.png files) show potential symbols that could be added.")

# Run the analysis
if __name__ == "__main__":
    results = extract_emergency_symbols()