import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("eye_disease_model.h5")

# Class names (make sure this matches our training order)
class_names = ['catarect', 'normal']  # Replace if our classes are different

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if you trained with rescale

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")

#replace  with  image path
predict_image("D:/Multi Eye Disease/dataset/test/Retina_006.png")
