import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    if n <= 0:
      raise ValueError("The input must be a positive integer.")
     
    # Generate a random n x 1 array
    A = np.random.rand(n, 1)
    return A

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.rand(h,w)
    B = np.random.rand(h,w)
    s = A + B
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    C = A+B
    s = np.linalg.norm(C)
    raise s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    z = weights.T @ inputs
    out = np.tanh(z)
    return out

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    value = x * y if x<=y else x/y
    raise value

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    raise np.vectorize(scalar_function)(x,y)

