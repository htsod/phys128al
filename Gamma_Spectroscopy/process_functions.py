# pocess_function.py
# Max Liang
# created 04/27/23
# Description:
# Contains all the function to process and visualize data
# The first four functions are devoted to spectral analysis
# The rest of the functions are devoted to Least Square Prediction on Composition
# The last three functions are incomplete
# Need evaluation method to the least square model


import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Spectral Analysis


def select_reg(xmin, xmax, x_data, y_data):
    # Get the indices of the x values within the range we want to fit
    selected_region = (xmin <= x_data) & (x_data <= xmax)
    # convert pandas.series to list
    # x_data = pandas.Series.tolist(x_data[selected_region])
    # y_data = pandas.Series.tolist(y_data[selected_region])
    x_data,y_data = x_data[selected_region],y_data[selected_region]
    return x_data,y_data


def find_max(xmin, xmax, x_data, y_data):
    selected_region = (xmin <= x_data) & (x_data <= xmax)
    x_data,y_data = x_data[selected_region],y_data[selected_region]
    return  x_data[np.argmax(y_data)], np.max(y_data)


def gaussian(x, a, N, center, stddev,b):
    """
    Returns the value of a Gaussian function at a given point x.

    Parameters:
    x (float): The point at which to evaluate the function.
    amplitude (float): The amplitude of the Gaussian function.
    center (float): The center of the Gaussian function.
    stddev (float): The standard deviation of the Gaussian function.

    Returns:
    float: The value of the Gaussian function at the given point.
    """
    return (a*x+b) + N/(2*math.pi*stddev**2)**(1/2) * np.exp(-(x - center) ** 2 / (2 * stddev ** 2))


def gaussian_fit(xmin, xmax, x_data, y_data): # Define the range of x values to fit the Gaussian to
    # Plot the data
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x_data, y_data, label="Data",color='red',s=20)
    x_data,y_data = select_reg(xmin, xmax, x_data, y_data)
    # Use the curve_fit function to fit the Gaussian to the selected data
    y_max = max(y_data)
    popt, pcov = curve_fit(gaussian, x_data, y_data,
                           sigma=(y_data)**(1/2), absolute_sigma=True,
                           p0=[1.0, y_max, (xmax+xmin)/2, 1.0,1.0])
    # c, N, center, stddev
    ## Print the results
    print('N:' ,popt[1],'with uncertainty =',(pcov[1][1])**(1/2))
    print('Center:' ,popt[2],'with uncertainty =',(pcov[2][2])**(1/2))
    print('Peak:' ,popt[1]/(2*math.pi*popt[3]**2)**(1/2))

    # Plot the fitted Gaussian function
    ax.plot(x_data, gaussian(x_data, *popt), label="Fit",color = 'grey',linewidth = 3)
    # Add labels and a legend to the plot
    ax.set_xlabel("Energy(KeV)",fontsize = 17)
    ax.set_ylabel("Counts",fontsize = 17)
    ax.legend(fontsize = 17)
    ax.tick_params(labelsize=15)
    # ax.axvline(x = popt[2], linestyle = '--')
    plt.errorbar(popt[2], popt[1]+popt[0], xerr=pcov[2][2], yerr=pcov[1][1], fmt='o')
    # Add the coordinate text to the plot
    plt.annotate('({}KeV, {})'.format(round(popt[2],2), round(popt[1]+popt[0],2)), xy=(popt[2], popt[1]+popt[0]),
                 xytext=(popt[2] + 20, popt[1]+popt[0] + 100), fontsize=17,
                 arrowprops=dict(facecolor='black', shrink=0.05))
    # Show the plot
    plt.xlim(xmin-50, xmax+50)
    plt.ylim(0,y_max*1.2)
    plt.show()



# Least Square Prediction on Composition


def func(x, a, b, c, d, e, f, g, h, i):
    w = np.array([a, b, c, d, e, f, g, h, i])
    return np.dot(w, x)


def least_square_weights(file_name, energy, countss, countss_uncer, cond=0.05, plotting=True):
    unknown = countss[0]
    samples = np.array(countss[1:])
    popt, pcov = curve_fit(func, samples,
                           unknown, p0=[1, 0, 0, 0, 0, 0, 0, 0, 0], 
                           sigma=np.sqrt(unknown + 1), bounds=(0, 5))

    unknown_sample = countss[0]
    sample_counts = np.array(countss[1:])
    sample_counts_uncer = np.array(countss_uncer[1:])
    pred_name = np.array(file_name[1:])
    condition = np.where(popt >= cond)

    print("The predictions are", pred_name[condition])
    print("With proportion of", np.round(popt[condition], 3))

    select_prop = popt[condition]
    select_samples = sample_counts[condition]
    select_samples_uncer = sample_counts_uncer[condition]

    pred = np.zeros(len(unknown_sample))
    pred_uncer = np.zeros(len(unknown_sample))
    for i in range(len(select_samples)):
        pred = pred + select_prop[i] * select_samples[i]
        pred_uncer = np.sqrt(np.square(pred_uncer) + np.square(select_samples_uncer[i]))

    
    pred_uncer = pred_uncer / pred
    pred = np.log(pred)
    unknown_sample = np.log(unknown_sample)


    pick_points = np.arange(int(len(pred)/15), int(len(pred)/1.7), int(len(pred)/30))
    x = energy[pick_points]
    y = pred[pick_points]
    y_err = pred_uncer[pick_points]

    if plotting == True:
        breakpoint = int(len(energy)*3/5)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(energy[:breakpoint], unknown_sample[:breakpoint], label="Unknown", alpha=0.65)
        ax.plot(energy[:breakpoint], pred[:breakpoint], label="Predition from least square", alpha=1)
        ax.errorbar(x, 
                    y, 
                    yerr=y_err, 
                    ecolor="k", 
                    markerfacecolor="black", 
                    markeredgecolor="black", 
                    markersize=0.2,
                    fmt="o", 
                    capthick=1.5, 
                    capsize=2)
        ax.set_title("Multi Digression Prediction with Unknown Source")
        ax.set_xlabel("Energy / keV")
        ax.set_ylabel("log(Counts)")
        ax.legend()
        








