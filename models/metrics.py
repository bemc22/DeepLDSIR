import tensorflow as tf
from utils import spacial_tv2

def prior_loss(y_true, y_pred):
    y_true = spacial_tv2(y_true)
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))