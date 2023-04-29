# read_csv.py
# Max Lian
# created 04/27/23
# Description:
# Scale all decay time to unknown so the counts are on the common ground.
# Such that counts from each element can be added and subtract



import numpy as np
import pandas as pd
import csv
from datetime import date
import matplotlib.pyplot as plt


def uncertainty_correction_factor(t_eu, t_eu_uncer, 
                                  t_es, t_es_uncer, 
                                  ts, ts_uncer, 
                                  tu, tu_uncer, 
                                  t_half, t_half_uncer):
    
    sum_1 = np.square(t_eu_uncer/t_eu) + np.square(t_es_uncer/t_es)
    sum_2 = np.square(ts_uncer) + np.square(tu_uncer)
    sum_3 = np.square(ts-tu)*np.square(np.log(2))*np.square(t_half_uncer/t_half)
    factor_23 = np.square(np.log(2)/t_half)*np.exp(np.log(2)*(ts-tu)/t_half)
    return np.sqrt(sum_1 + factor_23 * (sum_2 + sum_3))


def get_count_uncer(counts):
    
    return np.sqrt(counts)


def calculate_correction_factor():

    folder_name = ["eu152_csv", "na22_csv"]
    file_name = ["Unknown", "Cs137", "Na22", "Co57",
            "Co60", "Cd109", "Ba133", "Eu152",
                "Zn65", "Mn54"]
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
                461.4, 10.6*365, 13.52*365, 243.93, 312]) * 24 * 60 * 60


    t_unknown = t_list[0]
    t_exp = np.array(t_list)
    elasped_time_correction = t_unknown / t_exp

    t_elasped = []
    for d in range(len(produce_date)):
        t_elasped.append(abs(produce_date[d] - date.today()).days)
    t_elasped = np.array(t_elasped) * 24 * 60 * 60

    unknown_second = (abs(produce_date[0] - date.today()).days) * 24 * 60 * 60
    activity = np.log(2) / half_life
    factor = elasped_time_correction * np.e**(activity * t_elasped) * np.e**(-activity * unknown_second)

    # Calcuate the uncertainty for the correction factor
    t_eu = np.array([t_unknown]*len(t_list))
    t_eu_uncer = 0.05
    t_es = t_exp
    t_es_uncer = 0.05
    ts = t_elasped
    ts_uncer = np.array([30, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 24 * 60 * 60
    tu = np.array([unknown_second]*len(t_list))
    tu_uncer = np.array([30*24*60*60]*len(t_list))
    t_half = half_life
    t_half_uncer = np.array([1, 0.01*365, 
                                0.1*365, 0.01, 
                                0.01*365, 0.1, 
                                0.1*365, 0.01*365, 
                                0.01, 1]) * 24 * 60 * 60


    factor_uncer = uncertainty_correction_factor(t_eu, t_eu_uncer, 
                                  t_es, t_es_uncer, 
                                  ts, ts_uncer, 
                                  tu, tu_uncer, 
                                  t_half, t_half_uncer)
    
    factor[0] = 1
    factor_uncer[0] = 1

    # Collection results in a dataframe
    df_correction = pd.DataFrame(columns=df_col)
    df_correction.loc[len(df_correction.index)] = half_life
    df_correction.loc[len(df_correction.index)] = produce_date
    df_correction.loc[len(df_correction.index)] = t_list
    df_correction.loc[len(df_correction.index)] = factor
    df_correction.loc[len(df_correction.index)] = factor_uncer
    ser = pd.Series(["half_life / s", "produce_date", "t_exp / s", "corr_factor", "uncertainty"])
    df_correction.set_index(ser, drop=True, inplace=True)
     
    
    return factor, factor_uncer, df_correction


def input_bg():
    # specify file location
    folder_name = ["eu152_csv", "na22_csv"]
    file_name = ["Unknown", "bg"]
    folder_select = folder_name[1]

    # bg_noise to hold the counts, energy
    bg_noise = []
    t_list = []

    # Search for "Elapsed Live Time" for background noise and Unknown
    for i in range(len(file_name)):
        measurement_table = []
        with open(f"{folder_select}/{file_name[i]}.csv") as csvfile:
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


    # Store counts of background noise in bg_noise
    with open(f"{folder_select}/{file_name[1]}.csv") as csvfile:
        read_data = csv.reader(csvfile, delimiter=',')
        test = 0
        for row in read_data:
            if test == 1:
                bg_noise.append(int(row[2]))
            elif len(row) >=1 and row[0] == "Channel":
                test = 1
            else:
                pass

    bg_noise = np.array(bg_noise) * t_list[0] / t_list[1]
    bg_noise_uncer = get_count_uncer(bg_noise)
    return bg_noise, bg_noise_uncer


def read_data(bg_noise, bg_noise_uncer, factor, factor_uncer, plotting=[]):
    """
        Subtract the original data by bg_noise and then scale it by factor
        plotting takes list of element name as argument to plot the graph of that element
    """

    folder_name = ["eu152_csv", "na22_csv"]
    file_name = ["Unknown", "Cs137", 
                "Na22", "Co57",
                "Co60", "Cd109", 
                "Ba133", "Eu152",
                "Zn65", "Mn54"]
    
    countss = []
    countss_uncer = []

    for i in range(len(file_name)):
        channel = []
        energy = []
        counts = []
        
        folder_select = folder_name[1]
        file_select = file_name[i]

        with open(f"{folder_select}/{file_select}.csv") as csvfile:
            read_data = csv.reader(csvfile, delimiter=',')

            test = 0
            for row in read_data:
                if test == 1:
                    channel.append(int(row[0]))
                    energy.append(float(row[1]))
                    counts.append(int(row[2]))
                elif len(row) >=1 and row[0] == "Channel":
                    test = 1
                else:
                    pass

        energy = np.array(energy)        
        energy = energy[[i>=30 for i in energy]] # pick out energy > 30 keV

        counts = np.array(counts)
        counts_uncer = get_count_uncer(counts)

        counts = abs(np.array(counts) - bg_noise) # subtract background noise
        counts_bg_uncer = np.sqrt(np.square(counts_uncer) + np.square(bg_noise_uncer))

        if i != 0:
            tot_frac_uncer = np.sqrt(np.square(counts_bg_uncer/(counts+1)) + np.square(factor_uncer[i]/factor[i]))
        else:
            tot_frac_uncer = np.sqrt(np.square(counts_bg_uncer/(counts+1)))
        counts = factor[i] * counts[(len(counts) - len(energy)):]

        tot_uncer = tot_frac_uncer[(len(tot_frac_uncer) - len(counts)):] * counts

        countss.append(counts)
        countss_uncer.append(tot_uncer)

        for i in range(len(plotting)):
            if plotting[i] == file_select:
                fig, ax = plt.subplots(figsize=(18, 5))
                ax.plot(energy, np.log(counts+1))
                ax.set_title(f"Calibrated by {folder_select}, Measurement from {file_select}")
                ax.set_xlabel("Energy / keV")
                ax.set_ylabel("Counts")
            else:
                pass

    return file_name, energy, countss, countss_uncer






