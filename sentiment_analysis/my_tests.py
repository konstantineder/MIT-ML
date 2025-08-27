from string import punctuation, digits
import numpy as np
#import random

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    z = label*(np.inner(theta,feature_vector)+theta_0)
    if z<=1:
        return 1-z
    else:
        return 0


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """
    l = 0
    n = np.shape(feature_matrix)[0]
    for i in range(n-1):
        feature_vector = feature_matrix[i,:]
        label = labels[i]
        l = l + hinge_loss_single(feature_vector, label, theta, theta_0)
    return l/n


feature_vector = np.array([[1, 2, 4], [1, 2, 3]])
label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
print(np.shape(feature_vector)[0])
print(theta[0])
print(feature_vector[:,0])