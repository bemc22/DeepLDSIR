import os
import scipy.io as sio
import random
import tensorflow as tf

from tensorflow.data import AUTOTUNE
from ldsr.utils import coded2DTO3D


def get_list_imgs(data_path):
    list_imgs = os.listdir(data_path)
    list_imgs = [ os.path.join(data_path, img) for img in list_imgs ]
    random.shuffle(list_imgs)
    return list_imgs

def generate_H(coded_size, transmittance):
    H = tf.random.uniform(coded_size, dtype=tf.float32)
    H = tf.cast( H > transmittance, dtype=tf.float32)*1
    H = coded2DTO3D(H)
    return H

def csi_mapping(x, coded_size, transmittance=0.5):
    batch = x.shape[0]
    coded_size = (batch,) + coded_size
    H = generate_H(coded_size, transmittance)
    return (x, H), x


class DataGen(tf.data.Dataset):

    def _generator(self, data_path):  

        list_imgs = get_list_imgs(data_path) 

        for img_path in list_imgs:
            x = sio.loadmat(img_path)['img']
            yield x

    def __new__(cls, input_size=(512, 512, 31), data_path="../data/kaist/train"):
        output_signature = tf.TensorSpec(shape = input_size, dtype = tf.float32)

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = output_signature,
            args=(data_path,)
        )


def get_csi_pipeline(data_path, input_size=(512,512,31), batch_size=32, buffer_size=3, cache_dir=''):

    M, N, L = input_size
    coded_size = (N , M + L - 1 , 1)
    map_fun = lambda x: csi_mapping(x, coded_size)

    dataset = DataGen(input_size=input_size, data_path = data_path)

    pipeline_data = (
    dataset
    .batch(batch_size, drop_remainder=True)
    .cache(cache_dir) # cache_dir='' guarda el cache en RAM
    .shuffle(buffer_size)
    .map(map_fun, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
    )

    return pipeline_data

def get_pipeline(data_path, input_size=(512,512,31), batch_size=32, buffer_size=3, cache_dir=''):

    dataset = DataGen(input_size=input_size, data_path = data_path)
    map_fun = lambda x:  (x, x)

    pipeline_data = (
    dataset
    .cache(cache_dir)
    # .shuffle(batch_size) # cache_dir='' guarda el cache en RAM
    .batch(batch_size, drop_remainder=True)
    .map(map_fun, num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size)
    )

    return pipeline_data
