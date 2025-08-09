import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from app import EmergencyLightingDetector
from analyze_symbols import detect_shapes
from ml_model import EmergencyLightingMLModel

class MLIntegrator:
    def __init__(self, model_path=None, class_indices_path=None):
        # Default paths
        self.model_dir = os.path.join(os.getcwd(), 'models')
        
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'best_model.h5')
        
        if class_indices_path is None:
            class_indices_path = os.path.join(self.model_dir, 'class_indices.json')
        
        # Load the model if it exists
        self.model = None
        self.class_indices = None
        self.class_names = None
        self.image_size = (64, 64)  # Default image size
        
        try:
            self.load_model(model_path, class_indices_path)
            print(f"ML model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load ML model: {str(e)}")
            print("ML detection will be disabled. Train a model first using train_model.py")
        
        # Initialize the traditional detector as fallback
        self.traditional_detector = EmergencyLightingDetector()
    
    def load_model(self, model_path, class_indices_path):
        """Load the trained model and class indices"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(class_indices_path):
            raise FileNotFoundError(f"Class indices file not found: {class_indices_path}")
        
        # Load the model
        self.model = load_model(model_path)
        
        # Get input shape from model
        input_shape = self.model.input_shape[1:3]
        if input_shape[0] is not None and input_shape[1] is not None:
            self.image_size = input_shape
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping
        self.class_names = {v: k for k, v in self.class_indices.items()}
    
    def preprocess_image(self, image):
        """Preprocess an image for model prediction"""
        # Resize image to match model input size
        resized = cv2.resize(image, self.image_size)
        
        # Convert to RGB if grayscale
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:  # RGBA
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict_symbol(self, image):
        """Predict the symbol class for an image"""
        if self.model is None or self.class_names is None:
            return None, 0.0
        
        # Preprocess the image
        preprocessed = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(preprocessed)[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # Get the class name
        predicted_class = self.class_names.get(predicted_class_idx, 'UNKNOWN')
        
        return predicted_class, float(confidence)
    
    def detect_emergency_lights_ml(self, image, confidence_threshold=0.7):
        """Detect emergency lights using the ML model"""
        if self.model is None:
            print("ML model not loaded. Using traditional detection method.")
            return self.traditional_detector.detect_emergency_lights(image)
        
        # Convert PIL image to OpenCV format if needed
        if hasattr(image, 'convert'):
            # PIL image
            image_np = np.array(image.convert('RGB'))
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            # Assume it's already an OpenCV image
            image_cv = image
        
        # Convert to grayscale for shape detection
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect potential symbols using shape detection
        potential_symbols = detect_shapes(gray)
        
        # List to store detected emergency lights
        detections = []
        
        # Process each potential symbol
        for symbol_info in potential_symbols:
            bbox = symbol_info['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Extract the region of interest
            roi = image_cv[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
            
            # Predict using ML model
            symbol_type, confidence = self.predict_symbol(roi)
            
            # Skip if confidence is below threshold or not an emergency light
            if confidence < confidence_threshold or symbol_type == 'NOT_EMERGENCY':
                continue
            
            # Add to detections
            detection = {
                'type': symbol_type,
                'bbox': [x, y, x+w, y+h],
                'confidence': confidence,
                'description': self.get_symbol_description(symbol_type)
            }
            
            detections.append(detection)
        
        # If no detections with ML, fall back to traditional method
        if not detections and hasattr(self.traditional_detector, 'detect_emergency_lights'):
            print("No ML detections found. Falling back to traditional method.")
            return self.traditional_detector.detect_emergency_lights(image)
        
        return detections
    
    def get_symbol_description(self, symbol_type):
        """Get the description for a symbol type"""
        # Use the traditional detector's descriptions if available
        if hasattr(self.traditional_detector, 'fixture_types') and symbol_type in self.traditional_detector.fixture_types:
            return self.traditional_detector.fixture_types[symbol_type]
        
        # Default descriptions
        descriptions = {
            'A1E': 'Exit/Emergency Combo Unit',
            'A1-E': 'Exit/Emergency Combo Unit',
            'A1/E': 'Exit/Emergency Combo Unit',
            'EM-1': 'Emergency Type 1 Fixture',
            'EM-2': 'Emergency Type 2 Fixture',
            'EXIT-EM': 'Exit Sign with Emergency Backup',
            'EL': 'Emergency Light',
            'UNKNOWN': 'Unknown Emergency Fixture',
            'NOT_EMERGENCY': 'Not an Emergency Fixture'
        }
        
        return descriptions.get(symbol_type, 'Unknown Fixture Type')

# Function to patch the existing detector with ML capabilities
def patch_detector():
    """Patch the EmergencyLightingDetector class with ML detection capabilities"""
    # Create an instance of the ML integrator
    ml_integrator = MLIntegrator()
    
    # Store the original detect_emergency_lights method
    original_detect = EmergencyLightingDetector.detect_emergency_lights
    
    # Define a new method that uses ML detection if available
    def detect_emergency_lights_with_ml(self, image, use_ml=True):
        if use_ml and ml_integrator.model is not None:
            print("Using ML-based detection")
            return ml_integrator.detect_emergency_lights_ml(image)
        else:
            print("Using traditional detection")
            return original_detect(self, image)
    
    # Replace the method
    EmergencyLightingDetector.detect_emergency_lights = detect_emergency_lights_with_ml
    
    # Add the ML integrator as an attribute
    EmergencyLightingDetector.ml_integrator = ml_integrator
    
    print("EmergencyLightingDetector patched with ML capabilities")

# Main function to test the integration
def main():
    # Patch the detector
    patch_detector()
    
    # Create a detector instance
    detector = EmergencyLightingDetector()
    
    # Test with an image if available
    test_images = [f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if test_images:
        test_image_path = test_images[0]
        print(f"Testing with image: {test_image_path}")
        
        # Load the image
        image = cv2.imread(test_image_path)
        
        if image is not None:
            # Detect emergency lights using ML
            detections_ml = detector.detect_emergency_lights(image, use_ml=True)
            print(f"ML Detections: {len(detections_ml)}")
            for i, detection in enumerate(detections_ml):
                print(f"  {i+1}. {detection['type']} ({detection['confidence']:.2f}): {detection['description']}")
            
            # Detect emergency lights using traditional method
            detections_trad = detector.detect_emergency_lights(image, use_ml=False)
            print(f"Traditional Detections: {len(detections_trad)}")
            for i, detection in enumerate(detections_trad):
                print(f"  {i+1}. {detection['type']}: {detection['description']}")
        else:
            print(f"Could not read image: {test_image_path}")
    else:
        print("No test images found in the current directory")

if __name__ == "__main__":
    main()