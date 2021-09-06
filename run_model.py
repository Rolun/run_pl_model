import tensorflow as tf
import os
from PIL import Image
import numpy as np
from tensorflow import keras

mapping = {0: "Normal", 1: "Defect"}
path_to_model = "Pills_model"

def load_image(path):
    image = Image.open(path)
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

#Load some images
defected = load_image("C:/Users/Robert/Documents/PerceptiLabs/Default/pills/defect/pic.6.571.0.png")
normal = load_image("C:/Users/Robert/Documents/PerceptiLabs/Default/pills/normal/pic.6.443.0.png")

#Loads the model
model = keras.models.load_model(path_to_model)

#Makes some predictions and catogirizes them
prediction1 = model(defected)
print(prediction1)
print(mapping[np.asarray(prediction1['labels']).argmax()])
prediction2 = model(normal)
print(prediction2)
print(mapping[np.asarray(prediction2['labels']).argmax()])