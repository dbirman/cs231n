import numpy as np


def contrast_norm_forward(x):
  """    
  Computes the forward pass for a contrast normalization layer.  
  output:
  The input consists of N data points, each at t times, height H and width
  W (N x t x H x W)
      for each data point, output is:
          out(i,j,t) = [x(i,j,t) - np.mean(x)]/np.mean(x)  
  """
  #when one data point (e.g., one image)
  if len(x.shape)==3:
      T, H, W = x.shape
      x_resh = np.reshape(x,(T,H*W))            
      a = np.mean(x_resh,axis=1)
      b = np.transpose(np.tile(a,(H*W,1)))
      means = np.reshape(b,(T,H*W))
    
  #when many data points
  elif len(x.shape)==4:
      N, T, H, W = x.shape
      x_resh = np.reshape(x,(N,T,H*W)) 
      a = np.mean(x_resh,axis=2) 
      b = np.transpose(np.tile(a,(H*W,1,1)))        
      means = np.reshape(b,(N,T,H*W))    
      
  #contrast normalize  
  out = (x_resh - means)/means

  #shape back   
  out = np.reshape(out,x.shape)    
        
  return out
    
# def contrast_norm_backward(dout, cache):    
    
    

