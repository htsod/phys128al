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
    fig, ax = plt.subplots(figsize=(12,8))
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


def least_square_weights(file_name, energy, countss, plotting=True):
    y = countss[0]
    x = np.array(countss[1:])

    popt, pcov = curve_fit(func, x, y, p0=[1, 0, 0, 0, 0, 0, 0, 0, 0], bounds=(0, 1))

    unknown_sample = countss[0]
    sample_counts = np.array(countss[1:])
    pred_name = np.array(file_name[1:])

    print("The predictions are", pred_name[np.where(popt >= 0.1)])
    print("With proportion of", popt[np.where(popt >= 0.1)])

    select_prop = popt[np.where(popt >= 0.1)]
    select_samples = sample_counts[np.where(popt >= 0.1)]
    pred_data = sample_counts[np.where(popt >= 0.1)]
    pred = np.zeros(len(unknown_sample))
    for i in range(len(select_samples)):
        pred = pred + select_prop[i] * select_samples[i]

    if plotting == True:
        plt.plot(energy, np.log(unknown_sample + 1), label="Unknown")
        plt.plot(energy, np.log(pred + 1), label="Predition from least square")
        plt.xlabel("Energy / keV")
        plt.ylabel("log(Counts)")
        plt.legend()




def eval_cost(y1, y2, y3, a):
    cost_array = abs((a * y1 + (1 - a) * y2) ** 2 - (y3) ** 2)
    return cost_array.sum()


def score(countss, file_name):
    ele_min_cost = []
    proportion = []
    ele_name = []
    cs_counts = countss[1]
    unknown_counts = countss[0]
    x_list = np.linspace(0, 1, 100)
    for ele in range(len(file_name)):
        if file_name[ele] != "Unknown" and file_name[ele] != "Cs137":
            ele_name.append(file_name[ele])
            counts = countss[ele]
            cost_list = []
            for a in range(len(x_list)):
                cost_sum = eval_cost(cs_counts, counts, unknown_counts, x_list[a])
                cost_list.append(cost_sum)
                
            ele_min_cost.append(np.array(cost_list).min())
            proportion.append(x_list[np.argmin(np.array(cost_list))])
        else:
            pass
    return  ele_name, ele_min_cost, proportion



def score_result(ele_name, ele_min_cost, 
                 proportion, plotting=True, 
                 file_name=None, countss=None,
                 energy=None):
    
    df_fit = pd.DataFrame(data=[ele_name, np.round(ele_min_cost), proportion]).T
    df_fit.columns = ["ele_name", "ele_min_cost", "proportion"]
    df_fit.sort_values(by="ele_min_cost", inplace=True, ignore_index=True)

    elename = df_fit["ele_name"]
    cost = df_fit["ele_min_cost"]
    a = df_fit["proportion"]

    df_fit.display()

    print("Best case Scenario: \n"
        f"The element present are Cs137 with proportion {round(a[0], 3)} \n"
        f"with element {elename[0]} with proportion {round(1-a[0], 3)} \n"
        f"With the cost of {cost[0]}")

    if plotting == True:
        Cs137 = np.array(countss[1])
        for i in range(len(elename)):
            Best_fit = np.array(countss[file_name.index(elename[i])])
            tot_pred_best = a[i] * Cs137 + (1 - a[i]) * Best_fit

            plt.figure(figsize=(18, 5))
            plt.plot(energy, tot_pred_best, label=f"{elename[i]} + Cs137")
            plt.plot(energy, countss[0], label="Unknown")
            plt.legend()
            plt.title(f"{elename[i]}")
            plt.ylabel("log(Counts)")
            plt.xlabel("Energy / keV")




