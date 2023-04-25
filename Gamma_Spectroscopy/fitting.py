import matplotlib.pyplot as plt
import math

import pandas
import pandas as pd
import csv
import numpy as np
from scipy.optimize import curve_fit

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

def read_file(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)
    #print(df) ### For checking
    # Extract the "Channel" and "counts" columns from the DataFrame
    channel = df["Channel"]
    counts = df["Counts"]
    eng = df['Energy']
    #print(type(eng))
    return channel, eng, counts

def plot_data(x_data,y_data):
    # Plot the data
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(x_data, y_data)
    plt.xticks(np.arange(min(x_data), max(x_data) + 1, 100))
    plt.grid(axis="x")
    # Add labels and a title to the plot
    ax.set_xlabel("Energy(keV)")
    ax.set_ylabel("Counts")
    plt.yscale('log', base=2)
    plt.xlim(0,1300)
    # Show the plot
    # plt.show()

def generate_err(y_data):
    err = []
    for i in y_data:
        err.append(i**(1/2))
    return err

def select_reg(xmin, xmax, x_data, y_data):
    # Get the indices of the x values within the range we want to fit
    selected_region = (xmin <= x_data) & (x_data <= xmax)
    # convert pandas.series to list
    # x_data = pandas.Series.tolist(x_data[selected_region])
    # y_data = pandas.Series.tolist(y_data[selected_region])
    x_data,y_data = x_data[selected_region],y_data[selected_region]
    return x_data,y_data

def find_max(xmin, xmax, x_data, y_data):
    x_data,y_data = pandas.Series.tolist(x_data),pandas.Series.tolist(y_data)
    index_min, index_max = x_data.index(xmin), x_data.index(xmax)
    y_max = max(y_data[index_min, index_max])
    return y_max, x_data[y_data.index[y_max]]

def gaussian_fit(xmin, xmax, x_data, y_data): # Define the range of x values to fit the Gaussian to
    # Plot the data
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(x_data, y_data, label="Data",color='red',s=20)
    err = generate_err(y_data)
    x_data,y_data = select_reg(xmin, xmax, x_data, y_data)
    # Use the curve_fit function to fit the Gaussian to the selected data
    y_max = max(y_data)
    print()
    popt, pcov = curve_fit(gaussian, x_data, y_data,
                           sigma = (y_data)**(1/2), absolute_sigma= True,
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




