import tensorflow as tf

# feature_map_size_output = feature_map_size_input / 2
class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=1, padding='same'),
            tf.keras.layers.BatchNormalization()
        ]

        self.conv = tf.keras.layers.Conv2D(num_filters, kernel_size=(1, 1), strides=2, padding='same')
    
    @tf.function
    def call(self, x, training=False):

        x_skip = x
        
        for layer in self.layer_list:
            x = layer(x, training=training)

        x_skip = self.conv(x_skip)
    
        x = x_skip + x
        x = tf.nn.relu(x)

        return x