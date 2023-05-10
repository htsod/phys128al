# pocess_function.py
# Max Liang
# created 04/27/23
# Description:
# Contains all the function to process and visualize data
# The first four functions are devoted to spectral analysis credit giving credit to Eobard Ding
# The rest of the functions are devoted to Least Square Prediction on Composition


import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter

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


def least_square_weights(file_name, energy, countss, countss_uncer, cond1=0.05, cond2=0.05, plotting=True):
    unknown = countss[0]
    unknown_uncer = np.sqrt(unknown + 1)
    samples = np.array(countss[1:])
    popt, pcov = curve_fit(func, samples,
                           unknown, p0=[1, 0, 0, 0, 0, 0, 0, 0, 0], 
                           sigma=unknown_uncer, absolute_sigma=True, bounds=(0, 5))

    sample_counts = np.array(countss[1:])
    sample_counts_uncer = np.array(countss_uncer[1:])
    pred_name = np.array(file_name[1:])
    condition = np.where(popt >= cond1)

    select_prop = popt[condition]
    select_samples = sample_counts[condition]
    select_samples_uncer = sample_counts_uncer[condition]


    print("The predictions are", pred_name[condition])


    print("\nWith proportion of", np.round(popt[condition], 3))
    print("uncertainties =", end=" ")
    prop_uncer = []
    for i in range(len(pred_name[condition])):
        cond = condition[0][i]
        prop_uncer.append(round(pcov[cond][cond] ** (1/2), 4))
    print(prop_uncer)
    print(f"percentage uncertainty of {np.round(prop_uncer/popt[condition]*100, 3)}")
 

    print(f"\nThe equation that predicts y is stated as", end=" ")
    for i in range(len(pred_name[condition])):
        end_str = " + "
        if i == int(len(pred_name[condition])-1):
            end_str = " = "
        print(f"{np.round(popt[condition][i], 3)}*{pred_name[condition][i]}", end=end_str)
    print("Unknown")


    print("Each counts contribution is", end=" ")
    counts_sum = 0
    for c in range(len(sample_counts[condition])):
        counts_sum += popt[condition][c] * sample_counts[condition][c]
    counts_sum = counts_sum.sum()
    for i in range(len(sample_counts[condition])):
        end_str = ", "
        if i == len(sample_counts[condition]):
            end_str = ""
        nominator = popt[condition][i] * sample_counts[condition][i].sum()
        denominator = counts_sum
        print(f"{np.round((nominator/denominator)*100, 4)} %", end=end_str)


    # Reset criteria after inspection 

    condition = np.where(popt >= cond2)

    select_prop = popt[condition]
    select_samples = sample_counts[condition]
    select_samples_uncer = sample_counts_uncer[condition]

    

    pred = np.zeros(len(unknown))
    pred_uncer = np.zeros(len(unknown))
    for i in range(len(select_samples)):
        pred = pred + select_prop[i] * select_samples[i]
        pred_uncer = np.sqrt(np.square(pred_uncer) + np.square(select_samples_uncer[i]))

    pred_uncer = pred_uncer / (pred+1)
    pred = np.log(pred+1)
    unknown_uncer = unknown_uncer / (unknown+1)
    unknown = np.log(unknown+1)

    pick_points = np.arange(int(len(pred)/15), int(len(pred)/1.7), int(len(pred)/30))
    
    xs = energy[pick_points]
    ys = pred[pick_points]
    ys_err = pred_uncer[pick_points]
    
    pick_points = np.arange(int(len(pred)/15), int(len(pred)/1.7), int(len(pred)/30))
    
    xu = energy[pick_points]
    yu = unknown[pick_points]
    yu_err = unknown_uncer[pick_points]

    if plotting == True:

        breakpoint = int(len(energy)*3/5)
        energy = energy[:breakpoint]
        unknown = unknown[:breakpoint]
        unknown_uncer = unknown_uncer[:breakpoint]
        pred = pred[:breakpoint]
        pred_uncer = pred_uncer[:breakpoint]

        fig, ax = plt.subplots(figsize=(12,5))
        ax.scatter(energy, unknown, label="Unknown", alpha=0.65, s=1)
        ax.scatter(energy, pred, label="Predition from least square", alpha=1, s=1)

        ax.errorbar(xu, 
                    yu, 
                    yerr=yu_err, 
                    ecolor="k", 
                    markerfacecolor="black", 
                    markeredgecolor="black", 
                    markersize=0.2,
                    fmt="o", 
                    capthick=1.5, 
                    capsize=2,
                    label="Unknown Error")
        ax.errorbar(xs, 
                    ys, 
                    yerr=ys_err, 
                    ecolor="r", 
                    markerfacecolor="black", 
                    markeredgecolor="black", 
                    markersize=0.2,
                    fmt="o", 
                    capthick=1.5, 
                    capsize=2,
                    label="Prediction Error")
        
        ax.set_title("Multi Digression Prediction Comparing with the Unknown Source", fontsize=13, fontweight="bold")
        ax.set_xlabel("Energy / keV", fontsize=11)
        ax.set_ylabel("log(Counts)", fontsize=11)
        ax.legend()
        plt.savefig("figures/Overview", format="png")
    
    return energy, pred, pred_uncer, unknown, unknown_uncer







