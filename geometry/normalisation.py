from sklearn import preprocessing
import numpy as np


def normalise(values, norm='l1', axis=1):
    """
    Applies the Euclidean norm to the values.

    Params:
        - values (np.array): the values to be normalised

    Returns:
        A numpy array with the normalised values.
    """
    if norm == 'l0':
        if values.ndim == 2:
            return np.count_nonzero(values, axis=1)
        return np.count_nonzero(values)
    
    is_one_dim = False
    if len(values) == 0:
        return values
    
    if values.ndim == 1:
        is_one_dim = True
        values = [values]

    normalised_vector = preprocessing.normalize(X=values, norm=norm, axis=axis)
    if is_one_dim:
        return normalised_vector[0]
    return normalised_vector
