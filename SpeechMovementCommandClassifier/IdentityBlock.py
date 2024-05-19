import tensorflow as tf

# feature_map_size_input = feature_map_size_output
class IdentityBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters):
        super(IdentityBlock, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.BatchNormalization()
        ]

 
    @tf.function
    def call(self, x, training=False):

        x_skip = x
        
        for layer in self.layer_list:
            x = layer(x, training=training)

        x = x_skip + x
        x = tf.nn.relu(x)

        return x