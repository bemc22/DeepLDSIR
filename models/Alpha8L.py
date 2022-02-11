import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from utils import spacial_tv2, dd_cassi, tf_dwt


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


class HSI_net:
    def __init__(self, coded_aperture,
                 pretrained_weights=None, input_size=(512, 512, 31),
                 feature=64, denoiser="spacial_tv2"):

        self.regul_term = {
            "spacial_tv2": spacial_tv2
        }

        self.coded_aperture = coded_aperture
        self.pretrained_weights = pretrained_weights
        self.input_size = input_size
        self.denoiser_dims = (1,) +  input_size
        self.encode_size = (input_size[0:2]) + (int(feature/8),)

        self.krnl_regul = tf.keras.regularizers.L2(1e-8)
        self.feature = feature
        self.L = input_size[2]
        self.denoiser_fun = self.regul_term[denoiser]
        self.autoencoder = self.get_autoencoder()
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.modelR = self.get_modelR(self.denoiser_fun)
        self.priorA = self.get_prior_autoencoder(self.denoiser_fun)

        if(pretrained_weights):
            for layer in self.autoencoder.layers[7:13]:
                self.decoder.get_layer(layer.name).set_weights(
                    self.autoencoder.get_layer(layer.name).get_weights())

            for layer in self.encoder.layers[1:7]:
                self.encoder.get_layer(layer.name).set_weights(
                    self.autoencoder.get_layer(layer.name).get_weights())

            for layer in self.modelR.layers:
                if isinstance(layer, keras.layers.Conv2D):
                    self.modelR.get_layer(layer.name).set_weights(
                        self.autoencoder.get_layer(layer.name).get_weights())
                    layer.trainable = False

    def E(self, inputs):
        feature, krnl_regul = self.feature, self.krnl_regul

        conv1 = Conv2D(feature, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv1')(inputs)

        conv2 = Conv2D(feature, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv2')(conv1)

        conv3 = Conv2D(feature/2, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv3')(conv2)
        conv4 = Conv2D(feature/2, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv4')(conv3)

        conv5 = Conv2D(feature/4, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv5')(conv4)

        Eh = Conv2D(feature/8, 3, activation=None, padding='same',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=krnl_regul,
                    name='encode')(conv5)
        return Eh

    def D(self, inputs):
        feature, krnl_regul = self.feature, self.krnl_regul

        conv6 = Conv2D(feature/4, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv6')(inputs)

        conv7 = Conv2D(feature/2, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv7')(conv6)

        conv8 = Conv2D(feature/2, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv8')(conv7)

        conv9 = Conv2D(feature, 3, activation='relu', padding='same',
                       kernel_initializer='glorot_normal',
                       kernel_regularizer=krnl_regul,
                       name='conv9')(conv8)

        conv10 = Conv2D(feature, 3, activation='relu', padding='same',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=krnl_regul,
                        name='conv10')(conv9)

        Dh = Conv2D(self.L, 3, activation='relu', padding='same',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=krnl_regul,
                    name='decode')(conv10)
        return Dh

    def get_autoencoder(self):
        inputs = Input(self.input_size)
        Eh = self.E(inputs)
        DEh = self.D(Eh)    
        model = Model(inputs, DEh, name='Autoencoder')  
        if (self.pretrained_weights):
            model.load_weights(self.pretrained_weights)
        return model

    def get_prior_autoencoder(self, denoiser):
        inputs = Input(self.input_size)
        Eh = self.E(inputs)
        DEh = self.D(Eh)
        TV = Lambda(lambda x: denoiser(x), name='TV')(DEh)
        model = Model(inputs, [DEh, TV], name='PriorAutoencoder')
        if (self.pretrained_weights):
            model.load_weights(self.pretrained_weights)
        return model

    def get_encoder(self):
        inputs = Input(self.input_size)
        Eh = self.E(inputs)
        model = Model(inputs, Eh, name='Encoder')
        return model

    def get_decoder(self):
        inputs = Input(self.encode_size)
        DEh = self.D(inputs)
        model = Model(inputs, DEh, name='Decoder')
        return model

    def get_modelR(self, denoiser):
        inputs = Input(self.encode_size, name="Input", batch_size=1)
        layer = multiplyLayer(name='Layer1')(inputs)
        decode = self.D(layer)
        Iest = Lambda(lambda x: dd_cassi(
            x, self.coded_aperture), name="I")(decode)
        encode = self.E(decode)
        P = Subtract(name='P')([layer, encode])
        T = Lambda(lambda x: denoiser(x), name='TV')(decode)
        T2 = Lambda(lambda x: tf_dwt(x), name='W')(layer)
        model = Model(inputs, [Iest, P, T, T2], name='ReconsNet')
        return model
