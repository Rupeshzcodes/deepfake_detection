import sys

print("Checking imports...")
try:
    import streamlit
    print(f"Streamlit: {streamlit.__version__} [OK]")
except ImportError as e:
    print(f"Streamlit: [FAIL] - {e}")

try:
    import mediapipe
    print(f"Mediapipe: {mediapipe.__version__} [OK]")
except ImportError as e:
    print(f"Mediapipe: [FAIL] - {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__} [OK]")
except ImportError as e:
    print(f"OpenCV: [FAIL] - {e}")

try:
    import imutils
    print(f"Imutils: {imutils.__version__} [OK]")
except ImportError as e:
    print(f"Imutils: [FAIL] - {e}")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__} [OK]")
except ImportError as e:
    print(f"TensorFlow: [FAIL] - {e}")

try:
    from tensorflow.keras.models import load_model
    print("Keras Load Model: [OK]")
except ImportError as e:
    print(f"Keras Load Model: [FAIL] - {e}")

print("Verification complete.")
