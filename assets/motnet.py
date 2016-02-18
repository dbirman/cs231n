import numpy as np

from layers import *

class MotionNet(object):
  """
  An N-dimensional convolutional network with the following architecture:
  
    [conv-relu] x N - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(32,32,16), num_channels=4, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               num_conv_layers=3, num_affine_layers=1, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.fs = filter_size
    self.ncl = num_conv_layers
    self.nal = num_affine_layers
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #     [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
    H,W,T = input_dim
    
    for c in np.arange(self.ncl):
        dim_ = 1 if c==0 else num_channels
        self.params['W'+`c+1`] = np.random.randn(num_channels,filter_size,filter_size,filter_size,dim_) * weight_scale / np.sqrt(3 / filter_size*filter_size*filter_size*dim_)
        self.params['b'+`c+1`] = np.zeros((1,num_channels))
        
        # todo: implement for self.nal > 1
    #for a in np.arange(self.nal):
    #    sz = num_filters*H/(2**self.ncl)*W/(2**self.nal) if a==0 else hidden_dim
    #    self.params['A'+`a+1`]
    
    self.params['A'+`self.nal`] = np.random.randn(num_channels*H*W*T,num_classes)
    self.params['Ab'+`self.nal`] = np.zeros((1,num_classes))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = {'stride': 1, 'pad': (self.fs - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out = X
    ccache = []
    acache = []
    for c in np.arange(self.ncl):
        out,cache = conv3d_relu_forward(out,self.params['W'+`c+1`],self.params['b'+`c+1`],conv_param)
        ccache.append(cache)
    for a in np.arange(self.nal):
        out,cache = affine_forward(out,self.params['A'+`a+1`],self.params['Ab'+`a+1`])
        acache.append(cache)
        
    scores = out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dloss = softmax_loss(scores,y)
    for c in np.arange(self.ncl):
        loss += self.reg*.5*np.sum(self.params['W'+`c+1`]**2)
    for a in np.arange(self.nal):
        loss += self.reg*.5*np.sum(self.params['A'+`a+1`]**2)

    ##TODO
    dout = dloss
    
    for a in np.flipud(np.arange(self.nal)):
        dout, dw, db = affine_backward(dout,acache[a])
        grads['A'+`a+1`] = dw
        grads['Ab'+`a+1`] = db
    for c in np.flipud(np.arange(self.ncl)):
        cache = ccache[c]
        dout,dw,db = conv3d_relu_backward(dout,cache)
        grads['W'+`c+1`] = dw
        grads['b'+`c+1`] = db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
