import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  e_scores = np.e**scores
  p_scores = e_scores/np.sum(e_scores,axis = 1)[:,np.newaxis]
  for i in xrange(num_train):
    loss -= np.log(p_scores[i,y[i]])
    for j in xrange(num_classes):
      if j != y[i]:
        dW[:,j] += X[i,:]*e_scores[i,j]*(e_scores[i,j]/(np.sum(e_scores[i,:])**2))/p_scores[i,j]/num_train
      else:
        dW[:,j] += X[i,:]*e_scores[i,j]*(np.sum(e_scores[i,:])-e_scores[i,j])/(-(np.sum(e_scores[i,:])**2))/p_scores[i,j]/num_train
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #dW /= num_train
  dW += 0.5*reg*2*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  e_scores = np.e**scores
  correct_e_scores = e_scores[np.arange(num_train),y]
  p_scores = e_scores/np.sum(e_scores,axis = 1)[:,np.newaxis]
  correct_p_scores = p_scores[np.arange(num_train),y]

  grad_mat = -correct_e_scores[:,np.newaxis]*np.ones((num_train,num_classes))
  e_sum_mat = np.sum(e_scores,axis = 1)
  temp_mat = np.zeros((num_train,num_classes))
  temp_mat[np.arange(num_train),y] = e_sum_mat
  grad_mat += temp_mat
  grad_mat = (1/(e_sum_mat**2))[:,np.newaxis]*grad_mat
  grad_mat = (-1/correct_p_scores)[:,np.newaxis]*grad_mat
  grad_mat *= e_scores/num_train
  dW = X.T.dot(grad_mat)
  dW += 0.5*reg*2*W
  
  loss = np.sum(-np.log(correct_p_scores))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

