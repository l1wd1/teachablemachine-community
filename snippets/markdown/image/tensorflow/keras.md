from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import tensorflow as tf

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# تعريف دالة custom_objects لتخطي معلمة groups
def custom_objects():
    return {
        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D  # استبدال طبقة DepthwiseConv2D بtf.keras
    }

# Load the model with custom objects to avoid issues with 'groups'
model = load_model("keras_Model.h5", compile=False, custom_objects=custom_objects())

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")  # Display class name (removing extra space)
print("Confidence Score:", confidence_score)
