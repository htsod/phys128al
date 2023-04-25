import matplotlib.pyplot as plt
import math
import fitting as ff
import pandas
import pandas as pd
import csv
import numpy as np
from scipy.optimize import curve_fit
def calibration():
    calibration_values, calibration_labels = [87,753,942],['122keV','1112keV','1408keV']

    ch,eng,counts = ff.read_file('/Users/eobardthawne/Desktop/PHYS 128AL/GammaRay/data/Eu152.csv')
    # Plot the data
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1.semilogy(ch, counts,label = 'Eu152 Spectrum')
    #ax1.plot(ch, counts)
    # plt.grid(axis="x")
    # Add labels and a title to the plot
    ax1.set_xlabel("Channels",fontsize = 17)
    ax1.set_ylabel("Counts",fontsize = 17)
    ax1.tick_params(labelsize=15)
    for value, label in zip(calibration_values, calibration_labels):
        ax1.axvline(x=value, color='grey', linestyle='--')
        ax1.text(value, 5.8*10**4, label, ha='left', va='top', color='black',fontsize = 16)

    ax1.legend(fontsize = 18)
    # Show the plot
    plt.show()

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def cal_analysis():
    cal_eng = [122,248.07,350.84,786.38,981.04,1112,1408]
    Channels = [87,175,246,540,668,753,942]
    calpoint_x, calpoint_y=[87,753,942],[122,1112,1408]
    x = range(110, 1500)
    y = [i for i in x]
    fig, ax1 = plt.subplots(figsize=(14, 14))
    params, _ = curve_fit(quadratic, Channels, cal_eng)
    # Generate a plot of the quadratic function and the data points
    x_plot = np.linspace(0,1500)
    y_plot = quadratic(x_plot, *params)
    print('Quadratic Fitting Parameters:')
    print("a =", params[0])
    print("b =", params[1])
    print("c =", params[2])
    #ax1.plot(x,y, color = 'grey',linestyle='dashed',label = '')
    ax1.plot(x_plot,y_plot, color = 'grey',linestyle='dashed',label = 'quadratic fitting line')
    ax1.scatter(Channels,cal_eng,s = 50, label = 'measured data')
    ax1.scatter(calpoint_x,calpoint_y,color = 'r',s = 55, label ='points for calibration')
    ax1.set_xlabel("Channels",fontsize = 17)
    ax1.set_ylabel("Calibrated Enengy(KeV)",fontsize = 17)
    ax1.tick_params(labelsize=15)
    plt.xlim(0,1050)
    plt.ylim(0,1700)
    plt.legend(fontsize = 17,loc = 'upper left')
    plt.show()


