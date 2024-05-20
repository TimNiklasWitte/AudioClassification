from SpeechMovementCommandClassifier import *
from ModelConversion import *
# Convert the model
classifier = SpeechMovementCommandClassifier()
x = tf.zeros(shape=(32, 124, 129, 1))
classifier(x)
classifier.build(input_shape=(1, 124, 129, 1))
classifier.summary()

classifier.load_weights(f"./saved_models/epoch_20.weights.h5")

convert_model_to_tflite(classifier, target_path="./saved_models", model_name= model_full)