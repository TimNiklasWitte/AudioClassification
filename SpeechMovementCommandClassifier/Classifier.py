import tensorflow as tf

class Classifier(tf.keras.Model):

    def __init__(self):
        super(Classifier, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=2, padding='same', activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation="relu"),
            tf.keras.layers.BatchNormalization(),
        
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.GlobalMaxPool2D(),

            tf.keras.layers.Dense(15, activation="tanh"),
            tf.keras.layers.Dense(5, activation="softmax")
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.cce_loss = tf.keras.losses.CategoricalCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")

        self.metric_f1_score = tf.keras.metrics.F1Score(name="F1Score")


    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
         
        return x

    @tf.function
    def train_step(self, x, target):

        with tf.GradientTape() as tape:
            prediction = self(x, training=True)
            loss = self.cce_loss(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        self.metric_f1_score.update_state(target, prediction)

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

            self.metric_f1_score.update_state(target, prediction)


            label = tf.argmax(target, axis=-1)
            prediction = tf.argmax(prediction, axis=-1)
            
            self.metric_accuracy.update_state(label, prediction)
