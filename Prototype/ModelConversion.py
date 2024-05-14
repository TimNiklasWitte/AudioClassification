import tensorflow as tf
from Classifier import *


# Convert the model
classifier = Classifier()
x = tf.zeros(shape=(32, 124, 129, 1))
classifier(x)
classifier.build(input_shape=(1, 124, 129, 1))
classifier.summary()

classifier.load_weights(f"./saved_models/epoch_100.weights.h5")
classifier.export('./saved_models/model_full')

converter = tf.lite.TFLiteConverter.from_saved_model('./saved_models/model_full') # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.
with open('./saved_models/model.tflite', 'wb') as f:
  f.write(tflite_model)