#CSC336 Assignment #1 starter code for the report question
import numpy as np
import matplotlib.pyplot as plt

"""
See the examples in class this week if you
aren't sure how to start writing the code
to run the experiment and produce the plot for Q1.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.

A few things you'll likely find useful:

import matplotlib.pyplot as plt

hs = np.logspace(-15,-1,15)
plt.figure()
plt.loglog(hs,rel_errs)
plt.show() #displays the figure
plt.savefig("myplot.png")



a_cmplx_number = 1j #the j in python is the complex "i"

try to reuse code where possible ("2 or more, use a for")

"""

#example function header, you don't have to use this
def fd(f, x, h):
    """
    Return the forward finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    if type(h) != np.array and type(h) != list:
        return (f(x + h) - f(x)) / h
    else:
        return [(f(x + item) - f(x)) / item for item in h]
#your code here

def cd(f, x, h):
    """
    Return the centred difference approximation to the derivative of
    f(x), using the step size(s) in h.
    """

    if type(h) != np.array and type(h) != list:
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        return [(f(x + item) - f(x - item)) / (2 * item) for item in h]

def complexStep(f, x, h):
    if type(h) != np.array and type(h) != list:
        return np.imag((f(x + 1j * h) - f(x)) / h)
    else:
        return [np.imag((f(x + 1j * item) - f(x)) / item) for item in h]
    
def f(x):
    return x ** 2


if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    arr = []
    for i in range(1, 15):
        arr.append(1 * 10 ** (-i))


    forward = fd(f, 1, arr)
    forward_error = [abs(np.divide(float(format((item - 1), '.15f')), 1)) for item in forward]

    center = cd(f, 1, arr)
    center_error = [abs(np.divide(float(format((ite - 1), '.15f')), 1)) for ite in center]

    image = complexStep(f, 1, arr)
    image_error = [abs(np.divide(float(format((it - 1), '.15f')), 1)) for it in image]
    plt.plot(arr, forward_error, 'r', arr, center_error, 'b', arr, image_error, 'y--')
    plt.xlabel("h")
    plt.ylabel("relative error")
    plt.legend(("Forward difference", "Centered difference", "Complex Step"))

    plt.savefig("a1-report.png")
    plt.show()  # displays the figure
