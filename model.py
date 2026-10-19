import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_model():
    """
    Constructs the Hybrid CNN-LSTM model.
    CNN: ResNet50 (frozen)
    Bridge: Reshape (7x7x2048 -> 49x2048)
    LSTM: 64 units, dropout=0.2
    Classifier: Dense(1, sigmoid)
    """
    # CNN Base
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze CNN layers initially, but unfreeze the last 20 for fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Build the Hybrid Architecture
    model = Sequential()
    model.add(base_model)
    
    # ResNet50 output is (None, 7, 7, 2048)
    # Reshape to (None, 49, 2048) for LSTM (Sequence Length=49, Features=2048)
    model.add(Reshape((49, 2048)))
    
    # LSTM Layer
    model.add(LSTM(64, dropout=0.2))
    
    # Classifier
    model.add(Dense(1, activation='sigmoid'))

    # Compile with low learning rate for fine-tuning
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
