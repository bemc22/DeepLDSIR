import tensorflow as tf
from ldsr.utils import tv_prior

def prior_loss(y_true, y_pred):
    y_true = tv_prior(y_true)
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))