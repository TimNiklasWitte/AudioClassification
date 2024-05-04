from Classifier import *

classifier = Classifier()

x = tf.zeros(shape=(32, 124, 129, 1))
classifier(x)
    
classifier.build(input_shape=(1, 124, 129, 1))
classifier.summary()

classifier.load_weights(f"./saved_models/epoch_20.weights.h5")#.expect_partial()