import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import  Model
from ldsr.layers import EncodeLayer, DecodeLayer, FACTORS
from ldsr.utils import dd_cassi, dd_icassi, tv_prior, ImgGrad , ImgGradT


class MainModel:
    def __init__(self, input_size=(512,512,31), features=64, weights=None, factors=FACTORS, training=False):
        
        N , M , L = input_size        
        self.input_size = input_size
        self.unrolled_size = (N , M , 1)
        self.features = features  

        self.factors = factors
        self.autoencoder = self.get_autoencoder(input_size, features, training=training)


    def get_autoencoder(self, input_size, features, training=False):

        L = input_size[-1]
        inputs = Input(input_size)
        alpha = EncodeLayer(features, factors=self.factors)(inputs)
        decode = DecodeLayer(features, L, factors=self.factors)(alpha)
        prior = Lambda( lambda x: tv_prior(x) , name='Prior')(decode)

        if training:
            model = Model(inputs, [decode, prior], name="Autoencoder")
        else:
            model = Model(inputs, decode, name="Autoencoder")

        return model
    
    def get_encoder(self, input_size, features):

        inputs  = Input(input_size)
        encoder = EncodeLayer(features, factors=self.factors)
        alpha   = encoder(inputs)
        model   = Model(inputs, alpha , name='Encoder')

        encoder.set_weights(self.encoder_weights)
        encoder.trainable = False

        return model

    def get_decoder(self, input_size, features, L):

        inputs = Input(input_size)
        decoder = DecodeLayer(features, L, factors=self.factors)
        decode = decoder(inputs)
        model = Model(inputs, decode, name='Decoder')

        decoder.set_weights(self.decoder_weights)
        decoder.trainable = False

        return model    



