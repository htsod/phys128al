# adjust_sample.py
# Max Liang
# created 04/26/2023
# Description:
#
#


import pandas as pd
import numpy as np
import csv
from datetime import date


folder_name = ["eu152_csv", "na22_csv"]
file_name = ["Unknown", "Cs137", "Na22", "Co57",
           "Co60", "Cd109", "Ba133", "Eu152",
            "Zn65", "Mn54"]

name_list = []
t_list = []

for i in range(len(file_name)):
    measurement_table = []
    folder_select = folder_name[1]
    file_select = file_name[i]

    with open(f"{folder_select}/{file_select}.csv") as csvfile:
        read_data = csv.reader(csvfile, delimiter=',')
        for row in read_data:
            try:
                check = row[0]
            except:
                check = None
            if check == None or check != "Channel Data:":
                measurement_table.append(row)
            else:
                break

    condition = True
    while condition:
        try:
            measurement_table.remove([])
        except ValueError:
            condition=False

    df1 = pd.DataFrame(measurement_table)
    df2 = df1.T.iloc[:, 5:19]
    df2.columns = df2.iloc[0, :]
    measure_stat_df = df2.iloc[1:, :]

    t_list.append(float(measure_stat_df.loc[1, "Elapsed Live Time:"]))

df_col = ["Unknown", "Cs137", "Na22", "Co57",
           "Co60", "Cd109", "Ba133", "Eu152",
            "Zn65", "Mn54"]
produce_date = [date(2019, 3, 1), date(2022, 11, 1), 
                date(2022, 11, 3), date(2022, 11, 3), 
                date(2023, 1, 18), date(2022, 12, 16),
                date(2023, 1, 18), date(2022, 9, 7),
                date(2022, 12, 1), date(2019, 3, 1)]
half_life = np.array(
            [1, 30.08*365, 2.6*365, 271.74, 5.27*365, 
             461.4, 10.6*365, 13.52*365, 243.93, 312])

half_life = half_life * 24 * 60 * 60

t_unknown = t_list[0]
t_exp = np.array(t_list)
elasped_time_correction = t_unknown / t_exp

t_elasped = []
unknown_second = (abs(produce_date[0] - date.today()).days) * 24 * 60 * 60
for d in range(len(produce_date)):
    t_elasped.append(abs(produce_date[d] - date.today()).days)
t_elasped = np.array(t_elasped) * 24 * 60 * 60
activity = np.log(2) / half_life
factor = elasped_time_correction * np.e**(activity * t_elasped) * np.e**(-activity * unknown_second)
factor[0] = 1

df_correction = pd.DataFrame(columns=df_col)
df_correction.loc[len(df_correction.index)] = half_life
df_correction.loc[len(df_correction.index)] = produce_date
df_correction.loc[len(df_correction.index)] = t_list
df_correction.loc[len(df_correction.index)] = factor
ser = pd.Series(["half_life / days", "produce_date", "t_exp / s", "corr_factor"])
df_correction.set_index(ser, drop=True, inplace=True)
