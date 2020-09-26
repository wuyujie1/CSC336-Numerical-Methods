#CSC 336 HW#4 starter code

import scipy.linalg as sla
import numpy as np

#Q1 - set these to their correct values
M_35 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

P_36_a = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
P_36_b = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

A_q1c = np.array([[2, 3, -6], [1/2, -7.5, 11], [3/2, 13/15, 7/15]])
y_q1c = np.array([-8, 11, 7/15])
x_q1c = np.array([-1, 0, 1])

#Q3
def q3(A,B,C,b):
    M = sla.solve(B, 2 * A + np.identity(A.shape[0]))
    N = sla.solve(C, b)
    x = np.dot(M, N + np.dot(A, b))

    return x

if __name__ == '__main__':
    pass