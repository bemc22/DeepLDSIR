from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pywt


def wavelet(inputs, filters, lvl):
    outputs = tf.nn.conv3d(inputs, filters, padding='VALID',strides=[1, 1, 2, 2, 1])
    outputs = tf.split(outputs, int(outputs.shape[-1]), -1)

    if lvl != 0:
        outputs[0] = wavelet(outputs[0], filters, lvl-1)

    outputs = tf.concat(outputs, -1)
    outputs = tf.transpose(outputs, perm=[1, 0, 4, 2, 3])
    outputs = tf.unstack(outputs, int(outputs.shape[0]), 0)
    c, _, nrows, ncols = outputs[0].shape
    h = w = nrows*2
    outputs = [tf.reshape(i, [c, h//nrows, -1, nrows, ncols]) for i in outputs]
    outputs = [tf.transpose(i, perm=[0, 1, 3, 2, 4]) for i in outputs]
    outputs = [tf.reshape(i, [c, h, w]) for i in outputs]
    outputs = tf.stack(outputs, 1)
    outputs = tf.expand_dims(outputs, -1)
    return outputs


def tf_dwt(inputs, lvl=7):
    wave = 'Haar'
    w = pywt.Wavelet(wave)
    ll = np.outer(w.dec_lo, w.dec_lo)
    lh = np.outer(w.dec_hi, w.dec_lo)
    hl = np.outer(w.dec_lo, w.dec_hi)
    hh = np.outer(w.dec_hi, w.dec_hi)
    d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    d_temp[::-1, ::-1, 0, 0] = ll
    d_temp[::-1, ::-1, 0, 1] = lh
    d_temp[::-1, ::-1, 0, 2] = hl
    d_temp[::-1, ::-1, 0, 3] = hh

    filts = d_temp.astype('float32')
    filts = filts[None, :, :, :, :]
    filts = tf.convert_to_tensor(filts, name='filter')
    
    sz = 2 * (len(w.dec_lo) // 2 - 1)
    inputs = tf.pad(inputs, tf.constant(
        [[0, 0], [sz, sz], [sz, sz], [0, 0]]), mode="REFLECT")
    inputs = tf.expand_dims(inputs, 1)

    inputs = tf.split(inputs, [1]*int(inputs.shape.dims[4]), 4)
    inputs = tf.concat([x for x in inputs], 1)
    inputs = wavelet(inputs, filts, lvl)
    inputs = tf.transpose(inputs, perm=[0, 2, 3, 1, 4])[:, :, :, :, 0]
    return inputs

def plot_img(imagen, canales, p=1):
    img = imagen[0, :, :, :]
    img = img[:, :, canales]
    img = img / np.max(img)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.power(img, p))
    plt.show()


def sof_tresh(V, tau2, ro=1):
    x = tau2/ro
    V1 = (V > x)*(V - x)
    V2 = (V < -x)*(V + x)
    resul = V1 + V2
    return resul


def dd_cassi(inputs, CA):
    L = inputs.shape[3]
    M = inputs.shape[2]
    Y = None
    for i in range(L):
        aux = tf.expand_dims(inputs[:, :, :, i], -1)
        Temp = tf.multiply(aux, CA[:, :, i:M+i, :, 0])
        if Y is None:
            Y = Temp
        else:
            Y = tf.concat([Y, Temp], axis=3)

    Y = tf.reduce_sum(Y, 3, keepdims=True)
    Y = tf.cast(Y, tf.float32)
    return Y


def spacial_tv2(inputs):
    dy, dx = tf.image.image_gradients(inputs)
    tv = tf.add(tf.abs(dy), tf.abs(dx))
    return tv
