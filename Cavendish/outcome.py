# outcome.py
# created 05/08/2023
# Max Liang
# Description:
#
#


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt




def get_fit_para(func, x, y, y_uncer, name="Prediction", save_fig=False):


    popt, pcov = curve_fit(func, x, y, p0=[2, 0.001, 0.01, np.pi/2.5, 51], 
                        sigma=y_uncer, absolute_sigma=True)
    
    para_names = ["amplitude", "decay_const", "ang_freq", "phase", "const"]
    for i in range(len(para_names)):
        print(f"{para_names[i]} = {popt[i]} +- {np.sqrt(pcov[i][i])}")

    ang_freq = popt[2]
    ang_freq_uncer = np.sqrt(pcov[2][2])
    const = popt[4]
    const_uncer = np.sqrt(pcov[4][4])

    # select_data = np.arange(0, len(x), 15)
    # x_p = np.array(x)[select_data]
    # y_p = np.array(y)[select_data]
    # y_uncer = np.array(y_uncer)[select_data]
    # ax.errorbar(x_p, y_p, 
    #             yerr=y_uncer, ecolor="k", 
    #             markerfacecolor="b", markeredgecolor="b", 
    #             markersize=1, fmt="o", 
    #             capthick=1.5, capsize=2, 
    #             label="Experiment Data")

    fig, ax = plt.subplots(figsize=(10, 5))


    ax.fill_between(x, y+y_uncer,
                         y-y_uncer, facecolor='b',
                         alpha=0.2, label="Measurement Uncertainty")

    
    pred = func(x, popt[0], popt[1], popt[2], popt[3], popt[4])
    ax.plot(x, pred, "r", label=r"$Ae^{-\lambda}cos(\omega t+\phi) + c$")

    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Laser Position / m")
    ax.legend()


    if save_fig:
        plt.savefig(name)

    
    return ang_freq, const, ang_freq_uncer, const_uncer



def compute_theta(s1, s2):
    c = 31
    L = 1.74625
    l1 = np.sqrt(L**(2) + (s1 - c)**(2))
    l2 = np.sqrt(L**(2) + (s2 - c)**(2))
    nominator = (s1 - s2)**(2) - l1**(2) - l2**(2)
    denominator = 2 * l1 * l2
    theta = np.arccos(nominator/denominator)/2
    return theta



def get_G_new(theta, T, L):
    b = 42.2 / 1000
    d = 50 / 1000
    r = 9.55 / 1000
    m1 = 1.5
    factor = np.pi ** (2) * (4 * L * theta) * b ** (2)
    nominator = d**(2) + (2/5) * r ** 2
    denominator = T**2 * m1 * L * d
    beta = b**(3) / (b**(2) + 4 * d**(2))**(3/2)
    return factor * nominator / denominator





def get_G(delta_S, T, L):
    b = 42.2 / 1000
    d = 50 / 1000
    r = 9.55 / 1000
    m1 = 1.5
    factor = np.pi ** (2) * delta_S * b ** (2)
    nominator = d**(2) + (2/5) * r ** 2
    denominator = T**2 * m1 * L * d
    beta = b**(3) / (b**(2) + 4 * d**(2))**(3/2)
    return factor * nominator / denominator


def cal_corr():
    b = 42.2 / 1000
    d = 50 / 1000
    r = 9.55 / 1000
    m1 = 1.5
    beta = b**(3) / (b**(2) + 4 * d**(2))**(3/2)
    return beta




def linear_fit(func, x, y, y_uncer, name="Prediction", save_fig=False):


    popt, pcov = curve_fit(func, x, y, p0=[0, 0], 
                        sigma=y_uncer, absolute_sigma=True)
    
    para_names = ["gradient", "constant"]
    for i in range(len(para_names)):
        print(f"{para_names[i]} = {popt[i]} +- {np.sqrt(pcov[i][i])}")

    grad = popt[0]
    const = popt[1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, 
                yerr=y_uncer, ecolor="k", 
                markerfacecolor="k", markeredgecolor="k", 
                markersize=3, fmt="o", 
                capthick=1.5, capsize=2, 
                label="Data Points")
    pred = func(x, popt[0], popt[1])
    ax.plot(x, pred, label=r"$mx+b$")

    ax.set_title(name, fontweight="bold")
    ax.set_xlabel(r"$Time^{2}$ / $s^{2}$", fontsize=10)
    ax.set_ylabel("Displacement / m")
    ax.legend()
    if save_fig:
        plt.savefig(name)

    
    return grad, const


def get_G_by_a(grad, L):
    b = 42.2 / 1000
    d = 50 / 1000
    m1 = 1.5
    a = (grad / 100) * d / L
    G = b**(2) * a / (2 * m1)
    return G


