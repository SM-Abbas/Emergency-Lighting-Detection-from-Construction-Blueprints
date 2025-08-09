import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from pathlib import Path
import pickle
from datetime import datetime

class EmergencyLightingMLModel:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = [
            'A1E', 'A1-E', 'A1/E', 'EM-1', 'EM-2', 'EXIT-EM', 'EL', 'UNKNOWN', 'NOT_EMERGENCY'
        ]
        self.input_shape = (64, 64, 3)  # Default input shape for the model
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self, model_type='simple_cnn'):
        """Build the neural network model architecture"""
        if model_type == 'simple_cnn':
            self.model = Sequential([
                # First convolutional block
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Second convolutional block
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Third convolutional block
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Fully connected layers
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(len(self.class_names), activation='softmax')
            ])
            
            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        elif model_type == 'transfer_learning':
            # Use a pre-trained model like VGG16 or ResNet50
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
            
            # Freeze the base model layers
            for layer in base_model.layers:
                layer.trainable = False
            
            # Add custom layers on top
            x = base_model.output
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            predictions = Dense(len(self.class_names), activation='softmax')(x)
            
            # Create the model
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return self.model
    
    def prepare_data(self, image_dir, annotations_dir=None, augment=True):
        """Prepare training and validation data from images and annotations"""
        X = []
        y = []
        
        # If annotations directory is provided, use it to load labeled data
        if annotations_dir and os.path.exists(annotations_dir):
            # Load annotations and corresponding images
            for annotation_file in os.listdir(annotations_dir):
                if annotation_file.endswith('.json'):
                    with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
                        annotation_data = json.load(f)
                    
                    # Get corresponding image file
                    image_file = annotation_data.get('image_file')
                    if image_file and os.path.exists(os.path.join(image_dir, image_file)):
                        image = cv2.imread(os.path.join(image_dir, image_file))
                        if image is None:
                            continue
                        
                        # Process each annotation (bounding box)
                        for annotation in annotation_data.get('annotations', []):
                            bbox = annotation.get('bbox')
                            label = annotation.get('label')
                            
                            if bbox and label and label in self.class_names:
                                # Extract the region of interest
                                x, y, w, h = bbox
                                roi = image[y:y+h, x:x+w]
                                
                                if roi.size == 0:
                                    continue
                                
                                # Resize to the input shape expected by the model
                                roi = cv2.resize(roi, (self.input_shape[0], self.input_shape[1]))
                                
                                # Add to dataset
                                X.append(roi)
                                y.append(self.class_names.index(label))
        else:
            # If no annotations, use a simpler approach with directory structure
            # Assume each subdirectory in image_dir is named after a class
            for class_name in self.class_names:
                class_dir = os.path.join(image_dir, class_name)
                if os.path.exists(class_dir) and os.path.isdir(class_dir):
                    for image_file in os.listdir(class_dir):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(class_dir, image_file)
                            image = cv2.imread(image_path)
                            
                            if image is None:
                                continue
                            
                            # Resize to the input shape expected by the model
                            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
                            
                            # Add to dataset
                            X.append(image)
                            y.append(self.class_names.index(class_name))
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # One-hot encode the labels
        y = to_categorical(y, num_classes=len(self.class_names))
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Data augmentation
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            datagen.fit(X_train)
        
        return X_train, X_val, y_train, y_val, datagen if augment else None
    
    def train(self, X_train, y_train, X_val, y_val, datagen=None, epochs=50, batch_size=32):
        """Train the model with the prepared data"""
        if self.model is None:
            self.build_model()
        
        # Create model checkpoint to save the best model
        checkpoint_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'emergency_lighting_model_{epoch:02d}_{val_accuracy:.4f}.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr]
        
        # Train the model
        if datagen:
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )
        
        return history
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Build or load a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save class names
        with open(filepath.replace('.h5', '_classes.pkl'), 'wb') as f:
            pickle.dump(self.class_names, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model
        self.model = load_model(filepath)
        
        # Load class names if available
        class_file = filepath.replace('.h5', '_classes.pkl')
        if os.path.exists(class_file):
            with open(class_file, 'rb') as f:
                self.class_names = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
    
    def predict(self, image):
        """Predict the class of an input image"""
        if self.model is None:
            raise ValueError("No model available. Build or load a model first.")
        
        # Preprocess the image
        if isinstance(image, str) and os.path.exists(image):
            # Load image from file
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Invalid image input")
        
        # Resize to the input shape expected by the model
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Expand dimensions to create a batch of size 1
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'symbol': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_confidences': {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        }
    
    def detect_symbols_in_image(self, image, confidence_threshold=0.5):
        """Detect emergency lighting symbols in a full image"""
        if self.model is None:
            raise ValueError("No model available. Build or load a model first.")
        
        # Load image if it's a file path
        if isinstance(image, str) and os.path.exists(image):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Invalid image input")
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it's rectangular (potential emergency light fixture)
                if 0.2 < aspect_ratio < 5.0:
                    # Extract the region of interest
                    roi = image[y:y+h, x:x+w]
                    
                    if roi.size == 0:
                        continue
                    
                    # Make prediction on this ROI
                    prediction = self.predict(roi)
                    
                    # Only add detections above the confidence threshold and not classified as NOT_EMERGENCY
                    if prediction['confidence'] > confidence_threshold and prediction['symbol'] != 'NOT_EMERGENCY':
                        detections.append({
                            'symbol': prediction['symbol'],
                            'bounding_box': [x, y, x+w, y+h],
                            'confidence': prediction['confidence'],
                            'detection_method': 'ml_model'
                        })
        
        return detections

# Function to create and train the model
def create_and_train_model(image_dir, annotations_dir=None, model_type='simple_cnn', epochs=50):
    # Initialize the model
    model = EmergencyLightingMLModel()
    
    # Build the model architecture
    model.build_model(model_type=model_type)
    
    # Prepare the data
    X_train, X_val, y_train, y_val, datagen = model.prepare_data(image_dir, annotations_dir)
    
    # Train the model
    history = model.train(X_train, y_train, X_val, y_val, datagen, epochs=epochs)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join('models', f'emergency_lighting_model_{model_type}_{timestamp}.h5')
    model.save_model(model_path)
    
    return model, history, model_path

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy

# Function to visualize training history
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Main function to run if this script is executed directly
if __name__ == "__main__":
    # Check if we have training data
    image_dir = 'data/images'
    annotations_dir = 'data/annotations'
    
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory {image_dir} not found. Please create it and add training images.")
        # Create the directory structure
        os.makedirs(image_dir, exist_ok=True)
        for class_name in ['A1E', 'A1-E', 'A1/E', 'EM-1', 'EM-2', 'EXIT-EM', 'EL', 'UNKNOWN', 'NOT_EMERGENCY']:
            os.makedirs(os.path.join(image_dir, class_name), exist_ok=True)
    
    if not os.path.exists(annotations_dir):
        print(f"Warning: Annotations directory {annotations_dir} not found. Using directory structure for training.")
        os.makedirs(annotations_dir, exist_ok=True)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if we have enough data to train
    has_data = False
    for root, dirs, files in os.walk(image_dir):
        if files:
            has_data = True
            break
    
    if has_data:
        print("Starting model training...")
        model, history, model_path = create_and_train_model(image_dir, annotations_dir, model_type='simple_cnn', epochs=30)
        
        # Plot training history
        plot_training_history(history)
        
        print(f"Model training complete. Model saved to {model_path}")
    else:
        print("No training data found. Please add images to the data/images directory.")
        print("Each class should have its own subdirectory with example images.")