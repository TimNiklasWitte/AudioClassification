import tensorflow as tf

from IdentityBlock import *
from ConvBlock import *

class SpeechMovementCommandClassifier(tf.keras.Model):

    def __init__(self):
        super(SpeechMovementCommandClassifier, self).__init__()

        self.layer_list = [
            IdentityBlock(16),
            IdentityBlock(16),
            IdentityBlock(16),
            ConvBlock(32),

            IdentityBlock(32),
            IdentityBlock(32),
            IdentityBlock(32),
            ConvBlock(64),

            IdentityBlock(64),
            IdentityBlock(64),
            IdentityBlock(64),
     
            tf.keras.layers.GlobalMaxPool2D(),

            tf.keras.layers.Dense(20, activation="tanh"),
            tf.keras.layers.Dense(20, activation="tanh"),
            tf.keras.layers.Dense(5, activation="softmax")
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.cce_loss = tf.keras.losses.CategoricalCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function
    def call(self, x, training=False):
        #print(x.shape)
        for layer in self.layer_list:
            x = layer(x, training=training)
            #print(x.shape)

        return x

    @tf.function
    def train_step(self, x, target):

        with tf.GradientTape() as tape:
            prediction = self(x, training=True)
            loss = self.cce_loss(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        label = tf.argmax(target, axis=-1)
        prediction = tf.argmax(prediction, axis=-1)
        
        self.metric_accuracy.update_state(label, prediction)

    def test_step(self, dataset):
          
        self.metric_loss.reset_state()
        self.metric_accuracy.reset_state()

        for x, target in dataset:
            prediction = self(x)

            loss = self.cce_loss(target, prediction)
            
            self.metric_loss.update_state(loss)

            label = tf.argmax(target, axis=-1)
            prediction = tf.argmax(prediction, axis=-1)
            
            self.metric_accuracy.update_state(label, prediction)
