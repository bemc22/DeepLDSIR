import tensorflow as tf
import numpy as np
from models.utils import dd_cassi, tf_dwt, sof_tresh

def myPSNR(y_true, y_pred):
  return tf.image.psnr(y_true,y_pred,1)


class ADMM(object):
  def __init__(self,main_model):
    self.decode = main_model.decoder
    self.main_model = main_model
    self.denoiser_dims = main_model.denoiser_dims
    self.alpha_dims =  (1,) + main_model.encode_size 
    self.denoiser_fun = main_model.denoiser_fun
    self.A = tf.constant( np.ones((self.alpha_dims)) , dtype=tf.float32 , name='A')
    self.P = tf.constant( np.zeros((self.alpha_dims)), dtype=tf.float32 , name='P')
    self.alpha_ones = tf.constant( np.ones(shape=(self.alpha_dims))*0.0001 , dtype=tf.float32 , name='alpha_ones' )  

    optimizad = tf.keras.optimizers.Adam(learning_rate=1e-4)

    self.decode.compile(optimizer=optimizad, loss='mean_squared_error')  
    
  def __call__(self,I,epocas,steps,params):  
    tau1 , Tau , ro = params     
    tau2 = Tau*ro

    Tau2 = 0.1
    tau3 = Tau2*ro#
    recons = self.main_model.modelR 
    optimizad2 = tf.keras.optimizers.Adam(learning_rate=5e-2,beta_2=0.9)  

    recons.compile(
    optimizer=optimizad2,         
    loss={
    "I": tf.keras.losses.mean_squared_error,
    "P": tf.keras.losses.mean_squared_error,
    "TV":  tf.keras.losses.mean_squared_error,  
    "W":  tf.keras.losses.mean_squared_error,   
    } , loss_weights=[0.05 , tau1, ro/2, ro/5],) 

    z = np.zeros(self.denoiser_dims) 
    u = np.zeros(self.denoiser_dims)
    z2 = np.zeros(self.alpha_dims) 
    u2 = np.zeros(self.alpha_dims)
          
    alpha = recons.get_layer('Layer1').set_weights(self.alpha_ones) 
    temp_old = np.zeros(self.denoiser_dims)
    temp2_old = np.zeros(self.alpha_dims)

    for i in range(0,steps): 
      T = tf.constant( z - u, dtype=tf.float32 , name='T')
      T2 = tf.constant( z2 - u2, dtype=tf.float32 , name='Wp')

      Y = (I,self.P,T,T2) 
      train_set = tf.data.Dataset.from_tensor_slices( (self.A,Y))
      train_set = train_set.batch(1)  


      recons.fit(train_set ,batch_size=1 ,epochs=epocas*(i+1) , verbose=0 , initial_epoch=i*epocas)

      alpha = recons.get_layer('Layer1').get_weights()    
      temp = self.decode.predict( np.expand_dims(alpha[0] , 0) , batch_size=1)
      temp = self.denoiser_fun(tf.constant(temp, tf.float32 , name='temp'))
      temp = temp.numpy()

      temp2 = np.expand_dims( alpha[0], 0 )
      temp2 = tf_dwt(temp2).numpy()

      z = sof_tresh(temp + u, tau2/ro)      
      u = u + temp - z
      
      z2 = sof_tresh(temp2 + u2, tau3/ro)
      u2 = u2 + temp2 - z2


      temp_res = np.linalg.norm(temp.flatten() - temp_old.flatten(),2)
      temp_old = temp
      temp2_res = np.linalg.norm(temp2.flatten() - temp2_old.flatten(),2)
      temp2_old = temp2

      ro = ro*1.01

      print('||temp-t_old||'+str(temp_res) +'||temp2-t2_old||'+str(temp2_res))


    alpha = alpha[0]
    alpha = np.expand_dims(alpha,0)
    h_rec = self.decode.predict(alpha,batch_size=1)
    print('recons done')

    return h_rec