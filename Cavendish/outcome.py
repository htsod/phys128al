# outcome.py
# created 05/08/2023
# Max Liang
# Description:
#
#


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def func(t, a, lamb, gamma, phi, c):
    return a * np.exp(-lamb*t) * np.cos(gamma*t + phi) + c


def get_fit_para(x, y, y_uncer):


    popt, pcov = curve_fit(func, x, y, p0=[2, 0.001, 0.01, np.pi/2.5, 51], 
                        sigma=y_uncer, absolute_sigma=True)
    
    para_names = ["amplitude", "decay_const", "ang_freq", "phase", "const"]
    for i in range(len(para_names)):
        print(f"{para_names[i]} = {popt[i]} +- {np.sqrt(pcov[i][i])}")

    ang_freq = popt[2]
    const = popt[4]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, 
                yerr=y_uncer, ecolor="k", 
                markerfacecolor="b", markeredgecolor="b", 
                markersize=1, fmt="o", 
                capthick=1.5, capsize=2, 
                label="Experiment Data")
    pred = func(x, popt[0], popt[1], popt[2], popt[3], popt[4])
    ax.plot(x, pred, label="Prediction")

    ax.set_title("Prediction")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Displacement / m")
    ax.legend()

    
    return ang_freq, const


def get_G(delta_S, T, b, d, r, m1, L):
    factor = np.pi ** (2) * delta_S * b ** (2)
    nominator = d**(2) + (2/5) * r ** 2
    denominator = T**2 * m1 * L * d
    return factor * nominator / denominator


