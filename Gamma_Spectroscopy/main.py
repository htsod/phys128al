import Calibration as cal
import fitting as ft
#### Calibration
ch,eng,counts = ft.read_file('cd109.csv')   # change this to the path to the csv file
ft.plot_data(eng,counts)
ft.plt.show()
#ft.gaussian_fit(70, 100, eng, counts)      # for a particular peak, enter the approximate lower bound and upper bound (replace 70, 100)



