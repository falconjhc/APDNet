from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from utils.image_utils import image_show
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib
import cv2
import os
import scipy.misc as misc
import shutil

n1_list = [5,6,7,8,9,10,11,12,13]
r1 = 0.55 #


n2 = 5 # Number of possibly sharp edges
r2 = 1.5 #

data_group_num = 100
data_basic_num_in_each_group = 15
max_mask2_num = 7

background_mean_value = 0.25
mask1_mean_value = 0.55
mask2_mean_value =0.85

variance_paint = 0.0003



save_check_path = '/Users/harric/Downloads/ToyData/ToyDataCheck/'
save_data_path = '/Users/harric/Downloads/ToyData/ToyData/'

if os.path.exists(save_check_path):
    shutil.rmtree(save_check_path)
if os.path.exists(save_data_path):
    shutil.rmtree(save_data_path)

os.makedirs(save_check_path)


# matplotlib.use('Agg')  # environment for non-interactive environments
def add_gaussian_noise(X_img,var, trans):
    row, col = X_img.shape
    # Gaussian distribution parameters
    mean = 0
    sigma = var ** 0.5

    gaussian = np.squeeze(np.random.normal(mean,sigma,(row,col, 1))) +trans
    return gaussian



def draw_shape(n1_series,r1, n2, r2, mask2_translate_series):
    def _imp_with_series(n_series,r, rate=1, translate=0):

        pertubation  = np.random.normal(0,(np.max(n_series)-np.min(n_series))*0.1,len(n_series))
        n_series = n_series + pertubation

        angles = np.linspace(0, 2 * np.pi, len(n_series))
        codes = np.full(len(n_series), Path.CURVE4)
        codes[0] = Path.MOVETO

        verts = np.stack((np.cos(angles), np.sin(angles))).T * (2 * r * n_series+1-r)[:, None]
        verts[:,0] = verts[:,0] * rate + translate / 10 * np.random.choice([-1,1],1)[0]
        verts[:, 1] = verts[:, 1] * rate + translate / 10 * np.random.choice([-1,1],1)[0]
        verts[-1, :] = verts[0, :]  # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        return patch, verts

    def _imp_with_random_series(N,r, rate=1, translate=0):
        angles = np.linspace(0, 2 * np.pi, N)
        codes = np.full(N, Path.CURVE4)
        codes[0] = Path.MOVETO
        verts = np.stack((np.cos(angles), np.sin(angles))).T * (2 * r * np.random.random(N) + 1 - r)[:, None]
        verts[:,0] = verts[:,0] * rate + translate[0] + np.random.uniform(0.03,0.03)
        verts[:, 1] = verts[:, 1] * rate + translate[1] + np.random.uniform(0.03,0.03)
        verts[-1, :] = verts[0, :]  # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        return patch, verts


    #n1 = n1 + np.random.randint(0,max(int(n1/5),1)) * np.random.choice([-1,1],1)[0]
    r1 = r1 * np.random.uniform(0.95,1.05)
    #N1 = n1 * 3 + 1
    fig1 = plt.figure(figsize=[6.4, 6.4], dpi=10)
    ax1 = fig1.add_subplot(111)
    contour1, verts = _imp_with_series(n1_series,r1)
    ax1.add_patch(contour1)
    ax1.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax1.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax1.axis('off') # removes the axis to leave only the shape

    # fig.show()
    fig1.canvas.draw()
    data1 = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data1 = data1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    data1 = 255 - data1
    data_filled1 = np.float32(binary_fill_holes(data1[:,:,0]))
    plt.close()

    if len(mask2_translate_series[0])==0:
        data_filled2 = np.zeros_like(data_filled1)
    else:
        data_filled2_list = []
        for counter in range(len(mask2_translate_series[0])):

            valid = False
            while not valid:

                current_n2 = n2 + np.random.randint(0, max(int(n2 / 5),1)) * np.random.choice([-1, 1], 1)[0]
                current_r2 = r2 * np.random.uniform(0.75,1.25)
                N2 = current_n2 * 3 + 1
                fig2 = plt.figure(figsize=[6.4, 6.4], dpi=10)
                ax2 = fig2.add_subplot(111)
                scale = 0.5 * np.abs(np.random.rand(1,1))[0][0] / len(mask2_translate_series)
                translate0 = mask2_translate_series[0][counter]
                translate1 = mask2_translate_series[1][counter]

                contour2, _ = _imp_with_random_series(N2, current_r2, scale, [translate0,translate1])
                ax2.add_patch(contour2)
                ax2.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
                ax2.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
                ax2.axis('off')  # removes the axis to leave only the shape

                # fig.show()
                fig2.canvas.draw()
                data2 = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                data2 = data2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
                data2 = 255 - data2
                data_filled2 = np.float32(binary_fill_holes(data2[:, :, 0]))
                plt.close()

                guess = np.random.choice([0,1,2,3,4,5,6,7],1)
                if guess==0:
                    data_filled2 = np.zeros_like(data_filled2)
                    # print("Guess False")

                if np.sum(data_filled2 * data_filled1)!=0:
                    valid=True
                else:
                    valid=False
            data_filled2_list.append(np.expand_dims(data_filled2,axis=-1))
        data_filled2=np.sum(np.concatenate(data_filled2_list, axis=-1),axis=-1)
        data_filled2[np.where(data_filled2!=0)]=1

    data_filled1_corrected = data_filled1 + data_filled2
    data_filled1_corrected[np.where(data_filled1_corrected>1)]=1

    return data_filled1_corrected, data_filled2


for ii in range(data_group_num):
    current_data_path = os.path.join(save_data_path, 'Group%02d' % (ii + 1))

    current_background_value = background_mean_value + np.random.uniform(-0.1,0.1)
    current_mask1_vale = mask1_mean_value + np.random.uniform(-0.01, 0.01)
    current_mask2_vale = mask2_mean_value + np.random.uniform(-0.01, 0.01)


    if not os.path.exists(current_data_path):
        os.makedirs(current_data_path)
    n1 = np.random.choice(n1_list,1)[0]
    current_r1 = r1 * np.random.uniform(0.85,1.15)
    current_n2 = n2 + np.random.randint(0, int(n2 / 2)) * np.random.choice([-1, 1], 1)[0]
    current_r2 = r2 * np.random.rand()

    current_N1 = n1 * 3 + 1
    current_N1_series = np.random.random(current_N1)

    data_num_in_each_group = data_basic_num_in_each_group + np.random.choice([0,1,2,3],1)[0] * np.random.choice([-1,1],1)[0]

    mask2_num = np.random.randint(0, max_mask2_num)
    #mask2_num = 5
    mask2_translate_series1, mask2_translate_series2=[], []
    for jj in range(mask2_num):
        mask2_translate_series1.append(3 * np.random.randn(1, 1)[0][0] / 10 * np.random.choice([-1, 1], 1)[0])
        mask2_translate_series2.append(3 * np.random.randn(1, 1)[0][0] / 10 * np.random.choice([-1, 1], 1)[0])


    for jj in range(data_num_in_each_group):

        mask1, mask2 = draw_shape(current_N1_series, current_r1, current_n2, current_r2, [mask2_translate_series1,mask2_translate_series2])
        masks1_pixels = add_gaussian_noise(mask1, variance_paint, current_mask1_vale) * (mask1 - mask2)
        masks2_pixels = add_gaussian_noise(mask2, variance_paint, current_mask2_vale, ) * mask2
        background = add_gaussian_noise(mask2, variance_paint, current_background_value) * (np.ones_like(mask1) - mask1)
        final_img = masks1_pixels + masks2_pixels + background
        final_img = final_img - np.min(final_img)
        final_img = final_img / np.max(final_img)

        final_check = np.concatenate([np.tile(np.expand_dims(final_img, axis=-1), [1, 1, 3]),
                                      np.tile(np.expand_dims(mask1, axis=-1), [1, 1, 3]),
                                      np.tile(np.expand_dims(mask2, axis=-1), [1, 1, 3])], axis=1)

        misc.imsave(os.path.join(current_data_path, 'Img_%04d.tiff' % (jj+1)), final_img)
        misc.imsave(os.path.join(current_data_path, 'Mask1_%04d.tiff' % (jj+1)), mask1)
        misc.imsave(os.path.join(current_data_path, 'Mask2_%04d.tiff' % (jj+1)), mask2)
        misc.imsave(os.path.join(save_check_path, "CheckGroup%02dId%04d.tiff" % (ii+1,jj+1)), final_check)
    print("Group:%d/%d: Num:%d, n1:%d, r1:%f, n2:%d, r2:%f, mask2No:%d" % (ii + 1, data_group_num, data_num_in_each_group,
                                                                           n1, current_r1, current_n2, current_r2,mask2_num))
print("Complete All !")

