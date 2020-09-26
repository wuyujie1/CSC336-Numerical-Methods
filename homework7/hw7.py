import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.interpolate

#q2
def graph(formula, x_range, x_range_end):
    x = np.arange(x_range, x_range_end, 0.001)
    y = abs(eval(formula))
    plt.plot(x, y)

#q3
def origin(x):
    return 1/(1+25*x**2)

def a_generator(n, i_start, i_end):
    h = (i_end - i_start) / (n-1)
    hs = []
    B = []
    for i in range(1, n + 1):
        hs.append(i_start + (i-1)*h)
        B.append([hs[-1] ** j for j in range(n)])
    B = np.array(B)
    f = np.array([1/(1+25 * x ** 2) for x in hs])
    a = LA.solve(B, f)

    return a

def cubic_denerator(n, i_start, i_end):
    h = (i_end - i_start) / (n-1)
    hs = []
    fx = []
    for i in range(1, n + 1):
        hs.append(i_start + (i-1)*h)
        fx.append(origin(hs[-1]))
    return scipy.interpolate.CubicSpline(hs, fx)

def poly_est(x, a):
    i = 0
    y = 0
    for item in a:
        y += item * x ** i
        i+= 1
    return y

if __name__ == "__main__":
    #q2
    graph('0.99631689*x+0.01995143*x**2-0.20358546*x**3+0.02871423*x**4 -np.sin(x)', 0, np.pi/2)
    x = np.arange(0, np.pi/2, 0.001)
    y = 0.0014863448 * np.ones(x.shape[0])
    plt.plot(x,y)
    plt.show()

    #q3
    # n = 11
    a = a_generator(11, -1, 1)

    x_val_11 = []
    original_fun_11 = []
    poly_11 = []

    for i in np.arange(-1, 1, 0.001):
        x_val_11.append(i)
        original_fun_11.append(origin(i))
        poly_11.append(poly_est(i, a))
    cubic_11 = cubic_denerator(11, -1, 1)

    plt.plot(x_val_11, original_fun_11, "r", x_val_11, poly_11, "--b", x_val_11, cubic_11(x_val_11), '--g')
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend(("f(t)", "poly 11", "cubic 11"))
    plt.show()

    # n = 21
    a = a_generator(21, -1, 1)

    x_val_21 = []
    original_fun_21 = []
    poly_21 = []

    for i in np.arange(-1, 1, 0.001):
        x_val_21.append(i)
        original_fun_21.append(origin(i))
        poly_21.append(poly_est(i, a))
    cubic_21 = cubic_denerator(21, -1, 1)

    plt.plot(x_val_21, original_fun_21, "r", x_val_21, poly_21, "--b", x_val_21, cubic_21(x_val_21), '--g')
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend(("f(t)", "poly 21", "cubic 21"))
    plt.show()