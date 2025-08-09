import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our ML model
from ml_model import EmergencyLightingMLModel
from integrate_ml_model import MLIntegrator

def test_ml_model():
    """Test the ML model by loading it and making predictions on test images"""
    print("Testing ML model...")
    
    # Check if model exists
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "emergency_lighting_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return False
    
    # Load the model
    try:
        model = EmergencyLightingMLModel()
        model.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model classes: {model.class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Test the integrator
    try:
        integrator = MLIntegrator()
        print(f"ML Integrator initialized: {integrator.model_loaded}")
    except Exception as e:
        print(f"Error initializing integrator: {e}")
        return False
    
    return True

def test_prediction_on_sample(image_path):
    """Test prediction on a sample image"""
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
    
    # Load the model
    model = EmergencyLightingMLModel()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "emergency_lighting_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model.load_model(model_path)
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)
    
    # Get the predicted class
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = model.class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    print(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array)
    plt.title(f"Predicted: {predicted_class} ({confidence:.4f})")
    plt.axis('off')
    plt.show()

def main():
    """Main function to run tests"""
    if test_ml_model():
        print("ML model test passed!")
        
        # Test on sample images if available
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
        if os.path.exists(sample_dir):
            sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if sample_images:
                print(f"\nTesting prediction on sample image: {sample_images[0]}")
                test_prediction_on_sample(os.path.join(sample_dir, sample_images[0]))
            else:
                print("No sample images found in the samples directory.")
        else:
            print("Samples directory not found. Create a 'samples' directory with test images.")
    else:
        print("ML model test failed!")

if __name__ == "__main__":
    main()