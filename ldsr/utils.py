import numpy as np
import tensorflow as tf


TV_KERNEL = np.zeros((3,3,1,1))

TV_KERNEL[:,:,0,0] = np.array(
    [
    [ 0,-1, 0],
    [-1, 2, 0],    
    [ 0, 0, 0], 
    ]
)

@tf.function
def sof_tresh(V, tau):    
    V1 = (V > tau)*(V - tau)
    V2 = (V < -tau)*(V + tau)
    resul = V1 + V2
    return resul

@tf.function
def tv_prior(inputs):
    dy, dx = tf.image.image_gradients(inputs)
    tv = tf.add(tf.abs(dy), tf.abs(dx))
    return tv

@tf.function
def dd_cassi(x):
    inputs, H = x
    y = tf.multiply(H, inputs)
    y = tf.reduce_sum(y, -1, keepdims=True)
    return y

@tf.function
def dd_icassi(x):
    y, H = x
    H = tf.divide(H, tf.add(tf.reduce_sum(H, -1, keepdims=True), 1e-12))
    y = tf.tile(y, [1, 1, 1, H.shape[-1] ])
    return  tf.multiply(H, y) 

@tf.function
def ChannelwiseConv2D(inputs, kernel):
    inputs = tf.split(inputs , [1]*inputs.shape[-1] , axis=-1)
    output = [tf.nn.conv2d(i, kernel, strides=[1,1,1,1], padding="SAME") for i in inputs] 
    output = tf.concat(output,-1)
    return output

@tf.function
def ImgGrad(inputs):
    kernel = TV_KERNEL
    output = ChannelwiseConv2D(inputs, kernel)
    return output 

@tf.function
def ImgGradT(inputs):
    kernel = TV_KERNEL[::-1,::-1,:,:]
    output = ChannelwiseConv2D(inputs, kernel)
    return output 

@tf.function
def coded2DTO3D(CA, input_shape=None):

    if input_shape:
        M, N, L = input_shape
    else:
        _ , N, M, _ = CA.shape
        L = M - N + 1

    H = tf.concat([CA[:, :, i:N+i, :] for i in range(L)], -1)
    return H
