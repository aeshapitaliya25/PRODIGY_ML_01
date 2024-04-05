import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def predict_food_item(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    predictions = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
    
    return predictions

# Example usage
image_path = 'path_to_your_image.jpg'
predictions = predict_food_item(image_path)
for pred in predictions:
    print(pred[1], pred[2])