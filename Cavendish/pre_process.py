# pre_process.py
# created 05/08/2023
# Max Liang
# Description:
#
#

from libtiff import TIFF
import csv 
import numpy as np
import matplotlib.pyplot as plt


def preview_images(file_location, image_number, plotting=True):
    tif = TIFF.open(file_location)
    img_num = 0

    for image in tif.iter_images():
        img = image
        img_num += 1
        if img_num >= image_number:
            break
    if plotting == True:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        plt.yticks(np.arange(0, len(image), 20))
        plt.xticks(np.arange(0, len(image[0]), 50))
        plt.show()

    return img


def get_laser_pos(image):
    y_len = np.shape(image)[0]
    x_len = np.shape(image)[1]
    x_sum, y_sum = [], []
    for i in range(x_len):
        for j in range(y_len):
            if (abs(image[j][i][0] - 255) <= 2):
                x_sum.append(i)
                y_sum.append(j)
            else:
                pass
        x_mean, y_mean = np.array(x_sum).mean(), np.array(y_sum).mean()
    return y_mean, x_mean


def cal_dist(pos1, pos2):
    diff_y = pos1[0] - pos2[0]
    diff_x = pos1[1] - pos2[1]
    return np.sqrt(diff_y**2 + diff_x**2)


def cal_conver_factor(x1, x2, pos1, pos2):
    return abs(x1-x2)/cal_dist(pos1, pos2)


def conver_factor_and_init_pos(y_bound):
    folder = ["./s1/", "./s2/"]

    conver_list = []
    init_pos_list = []
    for f in range(len(folder)):
        conver_fac = []
        with open(folder[f] + "conver.csv") as csvfile:
            read_data = csv.reader(csvfile, delimiter=',')
            for row in read_data:
                conver_fac.append(row)

        f_location = []
        for i in range(len(conver_fac)):
            f_location.append(folder[f] + f"pos{f+1}_{i+1}.tif")

        init_pos = []
        for i in range(len(conver_fac)):
            x = []
            pos = [[], []]
            for j in range(2):
                if j == 0:
                    init_pos.append(float(conver_fac[i][2*j+1]))
                image = preview_images(f_location[i], int(conver_fac[i][2*j]), plotting=False)
                image = image[y_bound[0]:y_bound[1]]
                pix_y, pix_x = get_laser_pos(image)
                pos[j].append(pix_y)
                pos[j].append(pix_x)
                x.append(float(conver_fac[i][2*j+1]))
            conv = cal_conver_factor(x[0], x[1], pos[0], pos[1])
            conver_list.append(conv)
        init_pos_list.append(init_pos)
    conver_mean = np.array(conver_list).mean()
    conver_std = np.array(conver_list).std()
    print(f"\n\n\nThe average of the conversion is {conver_mean} with std of {conver_std}")
    print("Conversion_list = ", conver_list, "\n", "initial_position_list = ", init_pos_list)
    print("\n\n\n")

    return conver_mean, conver_std, init_pos_list


def gen_pix_pos(file_location, y_bound):
    tif = TIFF.open(file_location)
    pix_pos = []
    img_num = 0
    for image in tif.iter_images():
        image = image[y_bound[0]:y_bound[1]][:][:]
        y_mean, x_mean = get_laser_pos(image)
        pix_pos.append(x_mean)
        img_num += 1
        
    return pix_pos, img_num


def get_x(pix_pos, x1, conver):
    init_pix = pix_pos[0]
    rel_dist = []
    for i in range(len(pix_pos)):
        rel_dist.append(init_pix - pix_pos[i])
    pos_cm = np.array(rel_dist) * conver + x1
    return pos_cm


def plot_dots(image_len, pos_cm, T):
    x = np.arange(T, T * image_len+1, T)
    y = pos_cm
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    plt.title("Laser Points displacement")
    plt.xlabel("Time / s")
    plt.ylabel("Displacement / m")
    return x, y
