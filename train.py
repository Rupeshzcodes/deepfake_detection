import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, LSTM, Reshape, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def build_hybrid_model(input_shape=(224, 224, 3)):
    # 1. Base Model: ResNet50
    # Include_top=False gives us the 7x7x2048 feature map
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze all layers first
    base_model.trainable = False
    
    # Unfreeze the last 20 layers
    # ResNet50 has ~175 layers.
    for layer in base_model.layers[-20:]:
        layer.trainable = True
        
    inputs = Input(shape=input_shape)
    x = base_model(inputs) # Shape: (Batch, 7, 7, 2048)
    
    # 2. Reshape for LSTM
    # We treat the 7x7 spatial grid as a sequence of 49 steps
    target_shape = (49, 2048)
    x = Reshape(target_shape)(x)
    
    # 3. LSTM Layer
    # Return last state only implies we are classifying the whole image based on the sequence
    x = LSTM(128, return_sequences=False)(x)
    
    # 4. Dense Layers
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x) # Binary Classification
    
    model = Model(inputs, outputs, name="ResNet50_LSTM_Hybrid")
    return model

def train():
    # Settings
    BATCH_SIZE = 32
    EPOCHS = 10
    IMG_SIZE = (224, 224)
    DATA_DIR = "dataset" # Assumes structure: dataset/train/real, dataset/train/fake

    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL ERROR: Dataset directory '{DATA_DIR}' not found.")
        print("Please ensure your dataset is organized as 'dataset/train/real' and 'dataset/train/fake'.")
        return

    # Data Generators
    print("Initializing Data Generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Use 20% for validation
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'], # real=0, fake=1
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'], # real=0, fake=1
        subset='validation'
    )

    # Build Model
    print("Building Hybrid Model...")
    model = build_hybrid_model()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        "deepfake_model.h5", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    # Train
    print("Starting Training...")
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint]
        )
        
        # Plot Results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig("training_results.png")
        print("Training complete. Plot saved as training_results.png")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    train()
