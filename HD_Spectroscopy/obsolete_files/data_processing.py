# calculate.py
# Max Liang
# created 04/11/2023
# Description:
#


import csv
import pandas as pd


def get_max(v):
    max = v[0]
    index = 0
    for i in range(1,len(v)):
        if v[i] > max:
            max = v[i]
            index = i
    return max, index


def change_scale(time):
    t_scale = get_max(time)[0] / len(time)
    return t_scale


def get_time(voltage, time, t_start, t_break):
    start = int(t_start / change_scale(time))
    end = int(t_break / change_scale(time))
    v = voltage[start:end]
    t = time[start:end]
    val, indic = get_max(v)
    return t[indic]


file_name = "n3"
df1 = pd.DataFrame(columns=["t1(s)", "t2(s)", "delta_t(s)", "wavelength(AngStrom)"])
for i in range(1, 6):
    time = []
    voltage = []
    with open(f"{file_name}/{file_name}_{i}.csv") as csvfile:
        read_data = csv.reader(csvfile, delimiter=',')
        for row in read_data:
            time.append(float(row[0]))
            voltage.append(float(row[1]))
    t1 = get_time(voltage, time, 15, 30)
    t2 = get_time(voltage, time, 30, 40)
    t_diff = t2 - t1
    w_diff = t_diff * 0.06961988
    df1.loc[len(df1.index)] = [round(t1, 3), round(t2, 3), round(t_diff, 5), round(w_diff, 5)]

print(df1)
print(df1.describe())


# time = []
# voltage = []
# with open("n6_1.csv") as csvfile:
#     read_data = csv.reader(csvfile, delimiter=',')
#     for row in read_data:
#         time.append(float(row[0]))
#         voltage.append(float(row[1]))
#
# t1 = get_time(voltage, time, 15, 30)
# t2 = get_time(voltage, time, 30, 40)
# t_diff = t2 - t1
# w_diff = t_diff * 0.0869
# print(f"t1 = {t1}, t2 = {t2} "
#       f"\nThe diff in time is {t_diff} "
#       f"\nThe different in wavelength = {w_diff} A")
