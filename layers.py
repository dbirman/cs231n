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


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  D = 1
  for i in range(1,len(x.shape)):
    D = D * x.shape[i]
  X = np.reshape(x,(x.shape[0],D))
  out = np.add(np.dot(X,w), b)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  D = 1
  for i in range(1,len(x.shape)):
    D = D * x.shape[i]
  X = np.reshape(x,(x.shape[0],D))
  dx = np.dot(dout,w.T)
  dx = np.reshape(dx,x.shape)
  dw = np.dot(X.T,dout)
  db = np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.matrix.copy(x)
  out[out<0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x<=0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def conv_forward(x, w, b, conv_param):
    """
  Computes the forward pass for a layer of 3d convolution.
   """
    from theano.tensor.nnet.Conv3D import computeH
    out = computeH(x,w,b,conv_param['stride'])
    cache = (x,w,b,conv_param['stride'])
    return out, cache

def conv_backward(dout, cache):
    from theano.tensor.nnet.Conv3D import Conv3D
    dx, dw, db, da = Conv3D().grad(cache,(dout,))
    return dx, dw, db


