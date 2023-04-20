# calculate.py
# Max Liang
# created 04/11/2023
# Description:
#

import numpy as np

H = 1.096776 * 10 ** (7)
D = 1.097074 * 10 ** (7)

H_gamma = []
D_gamma = []
gamma_diff = []

for i in range(3, 9):
    int_val = 1 / 4 - 1 / (i ** 2)
    h_gamma = (int_val * H) ** (-1) * 10 ** (9)
    d_gamma = (int_val * D) ** (-1) * 10 ** (9)
    diff = abs(h_gamma - d_gamma)
    H_gamma.append(h_gamma)
    D_gamma.append(D_gamma)
    gamma_diff.append(diff)

    print(f"For n = {i}", "\n", "Hydrogen wavelength =", "{:.4f}".format(h_gamma),
          "\n", "Deuterium wavelength =", "{:.4f}".format(d_gamma),
          "\n", "Difference =", "{:.4f}".format(diff))

P_1 = 100 * (10 * 10 ** (-7)) / (4 * np.pi * (0.1) ** (2))
P_2 = P_1 * (10 * 10**(-7)) / (4 * np.pi)
E_p = (6.626 * 10**(-34)) * (3 * 10**(8)) / (500 * 10**(-9))
N_p = P_2 / E_p
I = N_p * 10**(7) * (1.602 * 10**(-19))


print(f"P_1 = {P_1}, P_2 = {P_2}, E_p = {E_p}, N_p = {N_p}, I = {I}, ")

