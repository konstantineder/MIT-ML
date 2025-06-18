import numpy as np

# It's possible, given the mistakes of each training data, to calculate the parameters when using Linear Perceptron Algorithm as follows:
#
# [Writen in LaTeX]
# \theta := theta_{initial} + \sum_{i=1}^{n} \beta^{i}y^{i}x^{i}
# \theta_{0} := theta_{0, initial} + \sum_{i=1}^{n} \beta^{i}y^{i}
#
# where \beta^{i} is the number of mistakes of ith training point

# Initialize the parameters as 0
theta_exam, theta_0_exam = np.zeros((1, 2)), 0

# Number os mistakes
history_exam = {'x1': 0,'x2': 1, 'x3': 9, 'x4': 10, 
                'x5': 5, 'x6': 9, 'x7': 11, 'x8': 3, 'x9': 1, 'x10': 1}

# Training set
x = np.array([[5, 2],
              [0, 0],
              [2, 0],
              [3, 0],
              [0, 2],
              [2, 2],
              [5, 1],
              [2, 4],
              [4, 4],
              [5, 5]])

y = np.array([[+1],
              [-1],
              [-1],
              [-1],
              [-1],
              [-1],
              [1],
              [1],
              [1],
              [1]])

# Evaluate the parameters
for i in range(len(history_exam)):
    theta_exam += history_exam.get('x' + str(i + 1))*y[i]*x[i]
    theta_0_exam += history_exam.get('x' + str(i + 1))*y[i]

print(f"Theta Exam: {theta_exam} | Theta 0 Exam: {theta_0_exam} \n")

import numpy as np
from sklearn.svm import SVC

# Points (features)
X = np.array([
    [0, 0], [2, 0], [3, 0], [0, 2], [2, 2],
    [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]
])

# Labels
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

# Fit the model
model = SVC(kernel='linear', C=1e10)
model.fit(X, y)

# Extract the parameters
w = model.coef_[0]
b = model.intercept_[0]

import numpy as np

# Points (features)
X = np.array([
    [0, 0], [2, 0], [3, 0], [0, 2], [2, 2],
    [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]
])

# Labels
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

# Parameters
w = np.array([0.9996875/2, 1.0/2])
b = -4.9990625/2

# Compute the Hinge loss for each example
hinge_losses = np.maximum(0, 1 - y * (np.dot(X, w) + b))

# Sum of Hinge losses
total_hinge_loss = np.sum(hinge_losses)
print("Total Hinge Loss:", total_hinge_loss)


import numpy as np

# It's possible, given the mistakes of each training data, to calculate the parameters when using Kernel Perceptron Algorithm as follows:
#
# [Writen in LaTeX]
# \theta := theta_{initial} + \sum_{i=1}^{n} \beta^{i}y^{i}\phi(x^{i})
# \theta_{0} := theta_{0, initial} + \sum_{i=1}^{n} \beta^{i}y^{i}
#
# where \beta^{i} is the number of mistakes of ith training point

# Initialize the parameters as 0
theta_exam, theta_0_exam = np.zeros(3), 0

# Number os mistakes
history_exam = {'x1': 1, 'x2': 65, 'x3': 11, 'x4': 31,
                'x5': 72, 'x6': 30, 'x7': 0, 'x8': 21, 'x9': 4, 'x10': 15}

# Training set
x = np.array([[0, 0],
              [2, 0],
              [1, 1],
              [0, 2],
              [3, 3],
              [4, 1],
              [5, 2],
              [1, 4],
              [4, 4],
              [5, 5]])

y = np.array([[-1],
              [-1],
              [-1],
              [-1],
              [-1],
              [1],
              [1],
              [1],
              [1],
              [1]])


def Kernel(x):
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])


# Evaluate the parameters
for i in range(len(history_exam)):
    theta_exam += history_exam.get('x' + str(i + 1))*y[i]*Kernel(x[i])
    theta_0_exam += history_exam.get('x' + str(i + 1))*y[i]

print(
    f"Kernel Vector:\n {np.array([Kernel(x[i]) for i in range(len(history_exam))])}\n")
print(f"Theta Exam: {theta_exam} | Theta 0 Exam: {theta_0_exam} \n")


# Classify the Training Points considering the Parameters above:
prediction = np.empty((y.shape))
for i, point in enumerate(x):
    phi = Kernel(x[i])
    prediction[i] = np.where(np.dot(theta_exam, phi) + theta_0_exam < 0, -1, 1)

if (y == prediction).sum() == y.shape[0]:
    print(
        f"The Kernel Perceptron correctly classified all points")
else:
    print(
        f"The Kernel Perceptron incorrectly classified {(y != prediction).sum()} points")
