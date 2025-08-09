import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from ml_model import EmergencyLightingMLModel

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)  # Small size for faster training
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Paths
DATA_DIR = os.path.join(os.getcwd(), 'data', 'images')
MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if GPU is available
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU is available: {gpus}")
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU for training")

# Prepare data generators
def prepare_data_generators():
    print(f"Preparing data from {DATA_DIR}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}. Run prepare_training_data.py first.")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = valid_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

# Train the model
def train_model(train_generator, validation_generator):
    # Get the number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Initialize the model
    model_instance = EmergencyLightingMLModel()
    
    # Build the model with transfer learning
    model = model_instance.build_model(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=num_classes,
        use_transfer_learning=True
    )
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
    
    # Save class indices
    import json
    with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    return model, history, train_generator.class_indices

# Evaluate the model
def evaluate_model(model, validation_generator, class_indices):
    # Get the true labels
    validation_generator.reset()
    y_true = validation_generator.classes
    
    # Get the predicted labels
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get class names
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    # Calculate accuracy
    accuracy = np.sum(y_true == y_pred_classes) / len(y_true)
    print(f"\nValidation Accuracy: {accuracy:.4f}")

# Plot training history
def plot_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))

# Main function
def main():
    # Check GPU availability
    check_gpu()
    
    # Prepare data generators
    try:
        train_generator, validation_generator = prepare_data_generators()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run prepare_training_data.py first to generate the training data.")
        return
    
    # Train the model
    print("\nTraining model...")
    model, history, class_indices = train_model(train_generator, validation_generator)
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_model(model, validation_generator, class_indices)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_history(history)
    
    print(f"\nTraining complete! Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()