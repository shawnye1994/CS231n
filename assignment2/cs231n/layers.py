import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_rows = x.reshape(x.shape[0],-1)
  out  = np.dot(x_rows,w) + b
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
  x_shape = x.shape
  dw = np.dot(x.reshape(x.shape[0],-1).T,dout)
  dx = np.dot(dout,w.T).reshape(x_shape)
  db = np.dot(np.ones(dout.shape[0]),dout)
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
  out = np.maximum(x,0)
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
  binary = np.maximum(x,0)
  binary[binary > 0] = 1
  dx = binary*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pad,stride = conv_param['pad'],conv_param['stride']
  H,HH = x.shape[2],w.shape[2]
  W,WW = x.shape[3],w.shape[3]
  N,F = x.shape[0],w.shape[0]
  C = x.shape[1]
  
  H_1 = 1 + (H + 2 * pad - HH) / stride
  W_1 = 1 + (W + 2 * pad - WW) / stride
  x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')  #the default pading constant is zero
  out = np.zeros((N,F,H_1,W_1))
  for i in range(H_1):
    for j in range(W_1):
      w_temp = w.reshape(F,-1)
      x_temp = x_pad[:,:,i*stride:HH+i*stride,j*stride:WW+j*stride].reshape(N,-1)
      temp_dot = np.dot(x_temp,w_temp.T).reshape(N,F,1,1) + b.reshape(F,1,1)
      out[:,:,i:i+1,j:j+1] = temp_dot
        
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x,w,b,conv_param = cache[0],cache[1],cache[2],cache[3]
  pad,stride = conv_param['pad'],conv_param['stride']
  H,HH = x.shape[2],w.shape[2]
  W,WW = x.shape[3],w.shape[3]
  N,F = x.shape[0],w.shape[0]
  C = x.shape[1]
  
  H_1 = 1 + (H + 2 * pad - HH) / stride
  W_1 = 1 + (W + 2 * pad - WW) / stride
  x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')

  dx = np.zeros(x_pad.shape)
  dw = np.zeros((F,C,HH,WW))
  db = np.zeros(F)
  for i in range(H_1):
    for j in range(W_1):
      w_temp = w.reshape(F,-1)
      x_temp = x_pad[:,:,i*stride:HH+i*stride,j*stride:WW+j*stride].reshape(N,-1)
       
      d_temp = dout[:,:,i:i+1,j:j+1].reshape(N,F)
      dx_temp = np.dot(d_temp,w_temp)
      dw_temp = np.dot(d_temp.T,x_temp)
    
      dx[:,:,i*stride:HH+i*stride,j*stride:WW+j*stride] += dx_temp.reshape(N,C,HH,WW)  #notice that there would be "+='', rather than "="
      db += np.dot(np.ones(N),d_temp)
      dw += dw_temp.reshape(F,C,HH,WW)
  dx = dx[:,:,pad:H+pad,pad:W+pad]
      
      
      
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  HH,WW,stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  N,C,H,W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  out_H = (H - HH)/stride + 1
  out_W = (W - WW)/stride + 1
  out = np.zeros((N,C,out_H,out_W))
  for i in range(out_H):
    for j in range(out_W):
      temp_x = x[:,:,i*stride:HH + i*stride,j*stride:WW + j*stride]
      temp_max = np.amax(temp_x,axis = 2)
      temp_max = np.amax(temp_max,axis = 2)
      out[:,:,i:i+1,j:j+1] = temp_max.reshape(N,C,1,1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param = cache[0],cache[1]
  HH,WW,stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  N,C,H,W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  out_H = (H - HH)/stride + 1
  out_W = (W - WW)/stride + 1
  dx = np.zeros(x.shape)
  for i in range(out_H):
    for j in range(out_W):
      temp_x = x[:,:,i*stride:HH + i*stride,j*stride:WW + j*stride]
      for n in range(N):
        for c in range(C):
          max_index = np.unravel_index(np.argmax(temp_x[n,c]),(HH,WW))
          dx[:,:,i*stride:HH + i*stride,j*stride:WW + j*stride][n,c,max_index[0],max_index[1]] = dout[n,c,i,j]
          
      
      
      
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

