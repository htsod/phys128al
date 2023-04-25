import matplotlib.pyplot as plt
import math
import fitting as ft
import pandas
import pandas as pd
import csv
import numpy as np

ch,x1,y1 = ft.read_file('/Users/eobardthawne/Desktop/PHYS 128AL/GammaRay/data/Cs137.csv')
ch,x2,y2 = ft.read_file('/Users/eobardthawne/Desktop/PHYS 128AL/GammaRay/data/Zn65.csv')
ch,x3,y3 = ft.read_file('/Users/eobardthawne/Desktop/PHYS 128AL/GammaRay/data/Unknown260.csv')

def plot_data(x1,y1,x2,y2,x3,y3):
    # Plot the data
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(x1, y1,label = 'Cs137')
    ax.scatter(x2, y2,label = 'Zn65')
    ax.scatter(x3, y3,label = 'Unknown')
    plt.grid(axis="x")
    # Add labels and a title to the plot
    ax.set_xlabel("Energy(keV)")
    ax.set_ylabel("Counts")
    plt.xlim(0,1300)
    plt.legend(fontsize=17)
    plt.yscale('log', base=2)
    # Show the plot
    plt.savefig('Compare.png')
    plt.show()

plot_data(x1,y1,x2,y2,x3,y3)