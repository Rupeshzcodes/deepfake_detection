import os
import cv2
import numpy as np
from imutils import paths

def load_data(dataset_path="dataset"):
    """
    Iterates through dataset/Real (Label 0) and dataset/Fake (Label 1).
    Limits loading to 500 images per class.
    Resizes images to (224, 224) and normalizes pixel values.
    """
    print("Loading data...")
    image_paths_real = list(paths.list_images(os.path.join(dataset_path, "Real")))
    image_paths_fake = list(paths.list_images(os.path.join(dataset_path, "Fake")))

    data = []
    labels = []

    # Process Real Images (Label 0)
    count = 0
    for image_path in image_paths_real:
        #if count >= 500:
        #    break
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (224, 224))
                image = image / 255.0  # Normalize
                data.append(image)
                labels.append(0)
                count += 1
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    print(f"Loaded {count} Real images.")

    # Process Fake Images (Label 1)
    count = 0
    for image_path in image_paths_fake:
        #if count >= 500:
        #    break
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (224, 224))
                image = image / 255.0  # Normalize
                data.append(image)
                labels.append(1)
                count += 1
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            
    print(f"Loaded {count} Fake images.")

    X = np.array(data, dtype="float32")
    y = np.array(labels, dtype="int")

    print(f"Data loaded successfully: {X.shape}")
    return X, y

if __name__ == "__main__":
    # Test the function
    if os.path.exists("dataset"):
        load_data()
    else:
        print("Dataset directory not found for testing.")
