# pre_process.py
# created 05/08/2023
# Max Liang
# Description:
#
#



from libtiff import TIFF
import numpy as np
import matplotlib.pyplot as plt


def mouse_event(event):
   print('x: {} and y: {}'.format(event.xdata, event.ydata))


def get_images(file_location, plotting=True):
    tif = TIFF.open(file_location)
    image = tif.read_image()
    image_len = 0
    for image in tif.iter_images():
        image_len += 1

    if plotting==True:
        fig = plt.figure()
        cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    
        plt.imshow(image)
        plt.show()


    return image_len, tif



def gen_pos_list(tif, x_bound, y_bound, 
                 init_pos_cm, pix_to_cm, rgb_laser):
    
    pos_list = [init_pos_cm]
    num_img = 0
    for image in tif.iter_images():
        pix_loc = []
        pix_color = []
        for j in range(y_bound[0], y_bound[1], 1):
            for i in range(x_bound[0], x_bound[1], 1):
                pix_loc.append([j, i])
                pix_color.append(image[j][i][:])

        pix_loc = np.array(pix_loc)
        pix_color = np.array(pix_color)

        loc_laser = 0
        pix_laser = 0
        compare = 2000
        for i in range(len(pix_color)):
            dist = np.sqrt(rgb_laser.dot(pix_color[i]))
            if dist < compare:
                compare = dist
                pix_laser = pix_color[i]
                loc_laser = pix_loc[i]

        num_img += 1
        if num_img == 1:
            init_pos_pix = loc_laser
        else:
            pix_diff = init_pos_pix - loc_laser
            pix_dist = np.sqrt(np.dot(pix_diff, pix_diff))
            pos_list.append(init_pos_pix + pix_dist * pix_to_cm)
            
        print(f"The laser pixel with rgb_val {pix_laser} at location {loc_laser}")
    return pos_list


def get_xy(image_len, pos_list, T):
    x = np.arange(T, T * image_len+1, T)
    y = pos_list
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    plt.title("Laser Points displacement")
    plt.xlabel("Time / s")
    plt.ylabel("Displacement / m")
    return x, y


