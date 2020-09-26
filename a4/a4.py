import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import time
import scipy.linalg as LA


def newton_poly_degree_8(coeff, x, t):
    return coeff[0] + coeff[1] * (x - t[0]) + coeff[2] * (x - t[0]) * (x - t[1]) + coeff[3] * (x - t[0]) * (x - t[1]) * (x - t[2])\
           + coeff[4] * (x - t[0]) * (x - t[1]) * (x - t[2]) * (x - t[3]) + coeff[5] * (x - t[0]) * (x - t[1]) * (x - t[2]) * (x - t[3]) * (x - t[4])\
           + coeff[6] * (x - t[0]) * (x - t[1]) * (x - t[2]) * (x - t[3]) * (x - t[4]) * (x - t[5])\
           + coeff[7] * (x - t[0]) * (x - t[1]) * (x - t[2]) * (x - t[3]) * (x - t[4]) * (x - t[5]) * (x - t[6])\
           + coeff[8] * (x - t[0]) * (x - t[1]) * (x - t[2]) * (x - t[3]) * (x - t[4]) * (x - t[5]) * (x - t[6]) * (x - t[7])

def newton_poly_degree_9(x, t, p8, t10, y10, coeff):
    x10 = (y10 - p8(coeff, t10, t[:-1])) / ((t10 - t[0]) * (t10 - t[1]) * (t10 - t[2]) * (t10 - t[3]) * (t10 - t[4]) * (t10 - t[5]) * (t10 - t[6]) * (t10 - t[7]) * (t10 - t[8]))
    return p8(coeff, x, t[:-1]) + x10 * (x - t[0]) * (x - t[1]) * (x - t[2]) * (x - t[3]) * (x - t[4]) * (x - t[5]) * (x - t[6]) * (x - t[7]) * (x - t[8])


if __name__ == "__main__":
    # #tt = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
    tt = np.linspace(1900, 1980, 9)
    # a)

    one = np.vander(tt, 9)
    two = np.vander(tt - 1900, 9)
    three = np.vander(tt - 1940, 9)
    four = np.vander((tt - 1940) / 40, 9)

    print("condition number for 1: " + str(format(np.linalg.cond(one), '.4e')))
    print("condition number for 2: " + str(format(np.linalg.cond(two), '.4e')))
    print("condition number for 3: " + str(format(np.linalg.cond(three), '.4e')))
    print("condition number for 4: " + str(format(np.linalg.cond(four), '.4e')))

    # b)
    actual_y = np.arr0ay([76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])
    coef = np.linalg.solve(four, actual_y)

    x = np.arange(1900, 1980, 1)
    est = np.polyval(coef, (x - 1940) / 40)

    plt.plot(tt, actual_y, "*k", x, est)
    plt.xlabel("year")
    plt.ylabel("population")
    plt.legend(("actual data", "est by the fourth set of basis function"))
    plt.show()

    # c)
    pchip = scipy.interpolate.PchipInterpolator(tt, actual_y)
    est_pchip = pchip(x)
    plt.plot(tt, actual_y, "*k", x, est, x, est_pchip, "--")
    plt.xlabel("year")
    plt.ylabel("population")
    plt.legend(("actual data", "est by the fourth set of basis function", "scipy.interpolate.PchipInterpolator"))
    plt.show()

    # d)
    cubic_spline = scipy.interpolate.CubicSpline(tt, actual_y, bc_type="clamped")
    est_cubic = cubic_spline(x)
    plt.plot(tt, actual_y, "*k", x, est, x, est_pchip, "--", x, est_cubic, "--")
    plt.xlabel("year")
    plt.ylabel("population")
    plt.legend(("actual data", "est by the fourth set of basis function", "scipy.interpolate.PchipInterpolator", "clamped cubic spline"))
    plt.show()

    # e)
    x = np.arange(1900, 1990, 1)
    est = np.polyval(coef, (x - 1940) / 40)
    est_pchip = pchip(x)
    est_cubic = cubic_spline(x)
    plt.plot(np.append(tt, 1990), np.append(actual_y, 248709873), "*k", x, est, x, est_pchip, "--", x, est_cubic, "--")
    plt.xlabel("year")
    plt.ylabel("population")
    plt.legend(("actual data", "est by the fourth set of basis function", "scipy.interpolate.PchipInterpolator", "clamped cubic spline"))
    plt.show()
    print(est_pchip[-1])

    # f)
    x = np.arange(1900, 1980, 1)

    time_ho = 0
    for i in range(5):
        start_ho = time.perf_counter()
        est = np.polyval(coef, (x - 1940) / 40)
        end_ho = time.perf_counter()
        time_ho += end_ho - start_ho
    time_ho /= 5

    est_pchip = pchip(x)

    time_cubic = 0
    for ii in range(5):
        start_cubic = time.perf_counter()
        est_cubic = cubic_spline(x)
        end_cubic = time.perf_counter()
        time_cubic += end_cubic - start_cubic
    time_cubic /= 5

    lag = scipy.interpolate.BarycentricInterpolator(tt, actual_y)

    time_la = 0
    for iii in range(5):
        start_lag = time.perf_counter()
        est_lag = lag(x)
        end_lag = time.perf_counter()
        time_la += end_lag - start_lag
    time_la /= 5

    plt.plot(tt, actual_y, "*k", x, est, x, est_pchip, "--", x, est_cubic, "--", x, est_lag, "--")
    plt.xlabel("year")
    plt.ylabel("population")
    plt.legend(("actual data", "est by the fourth set of basis function", "scipy.interpolate.PchipInterpolator", "clamped cubic spline", "lagrange"))
    plt.show()

    print("time for horner's evaluation scheme (polyval): {}\ntime for cubic spline: {}\ntime for lagrange: {}".format(
        str(format(time_ho, '.4e')), str(format(time_cubic, '.4e')), str(format(time_la, '.4e'))))

    # g)
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980
    lo = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, t2-t1, 0, 0, 0, 0, 0, 0, 0],
                   [1, t3-t1, (t3-t1)*(t3-t2), 0, 0, 0, 0, 0, 0],
                   [1, t4-t1, (t4-t1)*(t4-t2), (t4-t1)*(t4-t2)*(t4-t3), 0, 0, 0, 0, 0],
                   [1, t5-t1, (t5-t1)*(t5-t2), (t5-t1)*(t5-t2)*(t5-t3),(t5-t1)*(t5-t2)*(t5-t3)*(t5-t4), 0, 0, 0, 0],
                   [1, t6-t1, (t6-t1)*(t6-t2), (t6-t1)*(t6-t2)*(t6-t3),(t6-t1)*(t6-t2)*(t6-t3)*(t6-t4), (t6-t1)*(t6-t2)*(t6-t3)*(t6-t4)*(t6-t5), 0, 0, 0],
                   [1, t7-t1, (t7-t1)*(t7-t2), (t7-t1)*(t7-t2)*(t7-t3),(t7-t1)*(t7-t2)*(t7-t3)*(t7-t4), (t7-t1)*(t7-t2)*(t7-t3)*(t7-t4)*(t7-t5), (t7-t1)*(t7-t2)*(t7-t3)*(t7-t4)*(t7-t5)*(t7-t6), 0, 0],
                   [1, t8-t1, (t8-t1)*(t8-t2), (t8-t1)*(t8-t2)*(t8-t3),(t8-t1)*(t8-t2)*(t8-t3)*(t8-t4), (t8-t1)*(t8-t2)*(t8-t3)*(t8-t4)*(t8-t5), (t8-t1)*(t8-t2)*(t8-t3)*(t8-t4)*(t8-t5)*(t8-t6), (t8-t1)*(t8-t2)*(t8-t3)*(t8-t4)*(t8-t5)*(t8-t6)*(t8-t7), 0],
                   [1, t9-t1, (t9-t1)*(t9-t2), (t9-t1)*(t9-t2)*(t9-t3),(t9-t1)*(t9-t2)*(t9-t3)*(t9-t4), (t9-t1)*(t9-t2)*(t9-t3)*(t9-t4)*(t9-t5), (t9-t1)*(t9-t2)*(t9-t3)*(t9-t4)*(t9-t5)*(t9-t6), (t9-t1)*(t9-t2)*(t9-t3)*(t9-t4)*(t9-t5)*(t9-t6)*(t9-t7), (t9-t1)*(t9-t2)*(t9-t3)*(t9-t4)*(t9-t5)*(t9-t6)*(t9-t7)*(t9-t8)]])
    coef_new = LA.solve_triangular(lo, actual_y, lower=True)

    x = np.arange(1900, 1990, 1)
    est_newton_9_points = newton_poly_degree_8(coef_new, x, tt)
    est_newton_10_points = newton_poly_degree_9(x, np.linspace(1900, 1990, 10), newton_poly_degree_8, 1990, 248709873, coef_new)
    plt.plot(np.append(tt, 1990), np.append(actual_y, 248709873), "*k", x, est_newton_9_points, x, est_newton_10_points, "--")
    plt.xlabel("year")
    plt.ylabel("population")
    plt.legend(("actual data", "Newton's polynomial degree 8", "Newton's polynomial degree 9"))
    plt.show()

    # h)
    actual_y = np.array(
        [76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])
    round_7 = np.round(actual_y, -7)
    round_5 = np.round(actual_y, -5)
    round_4 = np.round(actual_y, -4)
    round_3 = np.round(actual_y, -3)
    round_2 = np.round(actual_y, -2)
    round_1 = np.round(actual_y, -1)

    coef_7 = np.linalg.solve(four, round_7)
    coef_5 = np.linalg.solve(four, round_5)
    coef_4 = np.linalg.solve(four, round_4)
    coef_3 = np.linalg.solve(four, round_3)
    coef_2 = np.linalg.solve(four, round_2)
    coef_1 = np.linalg.solve(four, round_1)


    x = np.arange(1900, 1980, 1)
    actual_y_rounded = np.round(actual_y, -6)
    coef_b = np.linalg.solve(four, actual_y)
    coef_h = np.linalg.solve(four, actual_y_rounded)

    print("coefficient for part b: " + str(coef_b))
    print("coefficient for round -1: " + str(coef_1))
    print("coefficient for round -2: " + str(coef_2))
    print("coefficient for round -3: " + str(coef_3))
    print("coefficient for round -4: " + str(coef_4))
    print("coefficient for round -5: " + str(coef_5))
    print("coefficient for part h: " + str(coef_h))
    print("coefficient for round -7: " + str(coef_7))

    bound_1 = np.linalg.cond(four) * (np.linalg.norm(actual_y - round_1) / np.linalg.norm(actual_y))
    bound_2 = np.linalg.cond(four) * (np.linalg.norm(actual_y - round_2) / np.linalg.norm(actual_y))
    bound_3 = np.linalg.cond(four) * (np.linalg.norm(actual_y - round_3) / np.linalg.norm(actual_y))
    bound_4 = np.linalg.cond(four) * (np.linalg.norm(actual_y - round_4) / np.linalg.norm(actual_y))
    bound_5 = np.linalg.cond(four) * (np.linalg.norm(actual_y - round_5) / np.linalg.norm(actual_y))
    bound_6 = np.linalg.cond(four) * (np.linalg.norm(actual_y - actual_y_rounded) / np.linalg.norm(actual_y))
    bound_7 = np.linalg.cond(four) * (np.linalg.norm(actual_y - round_7) / np.linalg.norm(actual_y))

    est_1 = bound_1 * np.linalg.norm(coef_b)
    est_2 = bound_2 * np.linalg.norm(coef_b)
    est_3 = bound_3 * np.linalg.norm(coef_b)
    est_4 = bound_4 * np.linalg.norm(coef_b)
    est_5 = bound_5 * np.linalg.norm(coef_b)
    est_6 = bound_6 * np.linalg.norm(coef_b)
    est_7 = bound_6 * np.linalg.norm(coef_b)
    print("delta coef from -1 to -7)")
    print(str(format(est_1, '.4e')))
    print(str(format(est_2, '.4e')))
    print(str(format(est_3, '.4e')))
    print(str(format(est_4, '.4e')))
    print(str(format(est_5, '.4e')))
    print(str(format(est_6, '.4e')))
    print(str(format(est_7, '.4e')))

    print('\n')
    print(format(np.linalg.norm(coef_1 - coef_b), '4e'))
    print(format(np.linalg.norm(coef_2 - coef_b), '4e'))
    print(format(np.linalg.norm(coef_3 - coef_b), '4e'))
    print(format(np.linalg.norm(coef_4 - coef_b), '4e'))
    print(format(np.linalg.norm(coef_4 - coef_b), '4e'))
    print(format(np.linalg.norm(coef_h - coef_b), '4e'))
    print(format(np.linalg.norm(coef_7 - coef_b), '4e'))




