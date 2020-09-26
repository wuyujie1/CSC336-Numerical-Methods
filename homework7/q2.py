import numpy as np
import matplotlib.pyplot as plt


def graph(formula, x_range, x_range_end):
    x = np.arange(x_range, x_range_end, 0.001)
    y = abs(eval(formula))
    plt.plot(x, y)

if __name__ == "__main__":
    graph('0.99631689*x+0.01995143*x**2-0.20358546*x**3+0.02871423*x**4 -np.sin(x)', 0, np.pi/2)
    x = np.arange(0, np.pi/2, 0.001)
    y = 0.0014863448 * np.ones(x.shape[0])
    plt.plot(x,y)
    plt.show()