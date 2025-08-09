import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our ML model
from ml_model import EmergencyLightingMLModel

def evaluate_model(test_data_dir=None):
    """Evaluate the ML model on a test dataset"""
    print("Evaluating ML model...")
    
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
    
    # If no test data directory is provided, use the default
    if test_data_dir is None:
        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test")
    
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found at {test_data_dir}")
        return False
    
    # Create a data generator for the test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load the test data
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    evaluation = model.model.evaluate(test_generator)
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Test Accuracy: {evaluation[1]:.4f}")
    
    # Get predictions
    predictions = model.model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes
    true_classes = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the confusion matrix
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Show the plot if running interactively
    plt.show()
    
    return True

def plot_sample_predictions(test_data_dir=None, num_samples=5):
    """Plot sample predictions from the test dataset"""
    # Check if model exists
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "emergency_lighting_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return False
    
    # Load the model
    model = EmergencyLightingMLModel()
    model.load_model(model_path)
    
    # If no test data directory is provided, use the default
    if test_data_dir is None:
        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test")
    
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found at {test_data_dir}")
        return False
    
    # Create a data generator for the test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load the test data without shuffling to maintain order
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=1,  # Process one image at a time
        class_mode='categorical',
        shuffle=True  # Shuffle to get random samples
    )
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Create a figure to display sample predictions
    plt.figure(figsize=(15, num_samples * 3))
    
    for i in range(num_samples):
        # Get a batch (single image)
        img, true_label = next(test_generator)
        
        # Make prediction
        prediction = model.model.predict(img)
        predicted_class_idx = np.argmax(prediction[0])
        true_class_idx = np.argmax(true_label[0])
        
        # Get class names
        predicted_class = class_names[predicted_class_idx]
        true_class = class_names[true_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        # Display the image with prediction
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img[0])
        
        # Set title color based on correctness
        title_color = 'green' if predicted_class_idx == true_class_idx else 'red'
        plt.title(f"True: {true_class}, Predicted: {predicted_class} ({confidence:.4f})", 
                 color=title_color, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the sample predictions
    samples_path = os.path.join(model_dir, "sample_predictions.png")
    plt.savefig(samples_path)
    print(f"Sample predictions saved to {samples_path}")
    
    # Show the plot if running interactively
    plt.show()
    
    return True

def main():
    """Main function to run evaluation"""
    # Evaluate the model
    if evaluate_model():
        print("\nModel evaluation completed successfully!")
        
        # Plot sample predictions
        print("\nGenerating sample predictions...")
        plot_sample_predictions(num_samples=5)
    else:
        print("Model evaluation failed!")

if __name__ == "__main__":
    main()