import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.constraints import NonNeg

KERNEL_REGUL = tf.keras.regularizers.L2(1e-8)
CONV_PARAMS = {
    'padding': 'same',
    'kernel_initializer': 'glorot_normal',
    'kernel_regularizer': KERNEL_REGUL,
}

# FACTORS = [1/2, 1/2, 1/4, 1/4, 1/8, 1/8]

FACTORS = [1, 1, 1/2, 1/2, 1/4, 1/8]

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, feature, activation='relu'):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(feature, 3, **CONV_PARAMS)
        # self.bn = BatchNormalization()
        self.activation = Activation(activation)

    def call(self, inputs):
        x = self.conv(inputs)
        # x = self.bn(x)
        x = self.activation(x)
        return x

class EncodeLayer(tf.keras.layers.Layer):
    def __init__(self, feature, name='Encoder', factors=FACTORS, **kwargs):
        super(EncodeLayer, self).__init__(name=name, **kwargs)

        self.convs = []

        for factor in factors[:-1]:
            ifeature = int(factor*feature)
            conv = ConvBlock(ifeature)
            self.convs.append(conv)

        self.out_features = int(feature*factors[-1])
        encode = ConvBlock(self.out_features, activation=None)
        self.convs.append(encode)

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        return x


class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self, feature, L, name='Decoder', factors=FACTORS, **kwargs):
        super(DecodeLayer, self).__init__(name=name, **kwargs)

        self.convs = []
        self.L = L

        for factor in factors[-2::-1]:
            ifeature = int(factor*feature)
            conv = ConvBlock(ifeature)
            self.convs.append(conv)

        decode = Conv2D(L, 3, activation='relu', **CONV_PARAMS)
        self.convs.append(decode)

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        return x
    

class multiplyLayer(tf.keras.layers.Layer):
    def __init__(self, name='Layer1'):
        super(multiplyLayer, self).__init__(name=name)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(
            input_shape[1], input_shape[2], input_shape[3]),
            initializer=tf.keras.initializers.glorot_normal,
            trainable=True)

    def call(self, input):
        return tf.multiply(input, self.kernel)