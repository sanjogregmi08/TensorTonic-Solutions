import numpy as np

def sigmoid(x):
    """
    Computes the sigmoid activation function using NumPy.
    Complies with vectorization requirements and handles 
    scalars, lists, and arrays.
    """
    # Convert input to numpy array to support vectorized operations
    x = np.array(x)
    
    # Mathematical implementation: 1 / (1 + e^-x)
    return 1 / (1 + np.exp(-x))