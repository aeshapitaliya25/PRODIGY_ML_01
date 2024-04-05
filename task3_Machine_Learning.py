import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score # type: ignore
from skimage.io import imread # type: ignore
from skimage.transform import resize # type: ignore
from tqdm import tqdm # type: ignore

# Function to extract features from images
def extract_features(images, target_size=(100, 100)):
    features = []
    for image_path in tqdm(images):
        image = imread(image_path)
        image = resize(image, target_size)
        features.append(image.flatten())
    return np.array(features)

# Path to your dataset
dataset_path = "/path/to/your/dataset"

# Load image paths
cat_image_paths = [os.path.join(dataset_path, "cats", filename) for filename in os.listdir(os.path.join(dataset_path, "cats"))]
dog_image_paths = [os.path.join(dataset_path, "dogs", filename) for filename in os.listdir(os.path.join(dataset_path, "dogs"))]

# Combine cat and dog image paths
image_paths = cat_image_paths + dog_image_paths

# Labels for cats (0) and dogs (1)
labels = [0] * len(cat_image_paths) + [1] * len(dog_image_paths)

# Extract features from images
X = extract_features(image_paths)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')

# Train SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict labels for test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)