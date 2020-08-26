# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import os
import scipy
from PIL import Image, ImageDraw
from imageio import imwrite as imsave # harric modified
from scipy.ndimage.morphology import binary_fill_holes
import utils.data_utils
import matplotlib.pyplot as plt
import imageio.core.util
import matplotlib.path as pth



def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning
# harric added to disable ignoring warning messages

# harric added for efficient image drawing
def image_show(img):
    img =np.squeeze(img)
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=2)
        img = np.tile(img,[1,1,3])
    img = img - np.min(img)
    img = img / np.max(img)
    plt.imshow(img)
    plt.show()

def generate_mask_on_img(img, mask):
    img = np.squeeze(img)
    mask = np.squeeze(mask)
    img = np.expand_dims(img, axis=-1)
    img_cpy = np.copy(img)
    mask_pixel = np.copy(img)
    mask_pixel[np.where(mask == 1)] = 1
    output = np.concatenate([mask_pixel, img_cpy, img_cpy], axis=-1)
    output = np.concatenate([np.tile(img_cpy,[1,1,3]), output],axis=1)
    image_show(output)
    return  output



# harric added to incorporate with segmentation_option=4 case
def regression2segmentation(input_regression):
    indices_mask_background = np.where(np.logical_and(input_regression >= 0.0, input_regression < 1.0 / 3.0))
    indices_mask_myocardium = np.where(np.logical_and(input_regression >= 1.0 / 3.0, input_regression <= 2.0 / 3.0))
    indices_mask_infarction = np.where(np.logical_and(input_regression > 2.0 / 3.0, input_regression <= 1.0))
    mask_background = np.zeros(shape=input_regression.shape, dtype=input_regression.dtype)
    mask_myocardium = np.zeros(shape=input_regression.shape, dtype=input_regression.dtype)
    mask_infarction = np.zeros(shape=input_regression.shape, dtype=input_regression.dtype)
    mask_background[indices_mask_background] = 1.
    mask_myocardium[indices_mask_myocardium] = 1.
    mask_infarction[indices_mask_infarction] = 1.
    output_segmentation = np.concatenate([mask_myocardium, mask_infarction, mask_background], axis=3)
    return output_segmentation



def save_multiimage_segmentation(x, m, y, folder, epoch):
        rows = []
        for i in range(x.shape[0]):
            y_list = [y[0][i, :, :, chn] for chn in range(y[0].shape[-1])]
            m_list = [m[i, :, :, chn] for chn in range(m.shape[-1])]
            if m.shape[-1] < y[0].shape[-1]:
                m_list += [np.zeros(shape=(m.shape[1], m.shape[2]))] * (y[0].shape[-1] - m.shape[-1])
            assert len(y_list) == len(m_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_list))

            for j in range(x.shape[3]):
                if j==0:
                    x_combine = x[i,:,:,j]
                else:
                    x_combine = np.concatenate([x_combine, x[i,:,:,j]], axis=1)
            # rows += [np.concatenate([x[i, :, :, :]] + y_list + m_list, axis=1)]
            rows += [np.concatenate([x_combine] + y_list + m_list, axis=1)]

        im_plot = np.concatenate(rows, axis=0)
        imsave(folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)
        # harric modified
        return im_plot

def save_segmentation(s, images, masks):
    '''
    :param folder: folder to save the image
    :param model : segmentation model
    :param images: an image of shape [H,W,chn]
    :param masks : a mask of shape [H,W,chn]
    :return      : the predicted segmentation mask
    '''
    images = np.expand_dims(images, axis=0)
    masks  = np.expand_dims(masks, axis=0)
    s = np.expand_dims(s, axis=0)
    true_background = np.zeros(shape=[masks.shape[0],masks.shape[1],masks.shape[1],1])
    for ii in range(masks.shape[-1]):
        true_background = true_background - np.expand_dims(masks[:,:,:,ii],axis=-1)
    masks = np.concatenate([masks,true_background], axis=-1)

    # In this case the segmentor is multi-output, with each output corresponding to a mask.
    if len(s[0].shape) == 4:
        s = np.concatenate(s, axis=-1)
    s_valid = s


    mask_list_pred = [s_valid[:, :, :, j:j + 1] for j in range(s_valid.shape[-1])]
    mask_list_real = [masks[:, :, :, j:j + 1] for j in range(masks.shape[-1])]
    # mask_list_real_background = np.ones(shape=[masks.shape[0],masks.shape[1],masks.shape[2],1],dtype=masks.dtype)
    # mask_list_real.append(mask_list_real_background)
    # for ii in range(masks.shape[-1]):
    #     mask_list_real_background = mask_list_real_background - np.expand_dims(masks[:, :, :, ii], axis=-1)
    if masks.shape[-1] < s_valid.shape[-1]:
        mask_list_real += [np.zeros(shape=masks.shape[0:3] + (1,))] * (s.shape[-1] - masks.shape[-1])

    # if we use rotations, the sizes might differ
    m1, m2 = utils.data_utils.crop_same(mask_list_real, mask_list_pred)
    images_cropped, _ = utils.data_utils.crop_same([images], [images.copy()], size=(m1[0].shape[1], m1[0].shape[2]))
    mask_list_real = [s[0, :, :, 0] for s in m1]
    mask_list_pred = [s[0, :, :, 0] for s in m2]
    images_cropped = [s[0, :, :, :] for s in images_cropped]


    for ii in range(images_cropped[0].shape[2]):
        if ii ==0:
            source = images_cropped[0][:,:,ii]
        else:
            source = np.concatenate([source, images_cropped[0][:,:,ii]], axis=1)

    for ii in range(len(mask_list_pred)):
        if ii ==0:
            row1 = np.concatenate([source,mask_list_pred[ii]], axis=1)
        else:
            row1 = np.concatenate([row1, mask_list_pred[ii]], axis=1)
    for ii in range(len(mask_list_real)):
        if ii ==0:
            row2 = np.concatenate([source,mask_list_real[ii]], axis=1)
        else:
            row2 = np.concatenate([row2, mask_list_real[ii]], axis=1)

    # row1 = np.concatenate(images_cropped + mask_list_pred, axis=1)
    # row2 = np.concatenate(images_cropped + mask_list_real, axis=1)
    im = np.concatenate([row1, row2], axis=0)
    # imsave(os.path.join(folder, name_prefix + '.png'), im)
    return im


def convert_myo_to_lv(mask):
    '''
    Create a LV mask from a MYO mask. This assumes that the MYO is connected.
    :param mask: a 4-dim myo mask
    :return:     a 4-dim array with the lv mask.
    '''
    assert len(mask.shape) == 4, mask.shape

    # If there is no myocardium, then there's also no LV.
    if mask.sum() == 0:
        return np.zeros(mask)

    assert mask.max() == 1 and mask.min() == 0

    mask_lv = []
    for slc in range(mask.shape[0]):
        myo = mask[slc, :, :, 0]
        myo_lv = binary_fill_holes(myo).astype(int)
        lv = myo_lv - myo
        mask_lv.append(np.expand_dims(np.expand_dims(lv, axis=0), axis=-1))
    return np.concatenate(mask_lv, axis=0)


def makeTextHeaderImage(col_widths, headings, padding=(5, 5)):
    im_width = len(headings) * col_widths
    im_height = padding[1] * 2 + 11

    img = Image.new('RGB', (im_width, im_height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    for i, txt in enumerate(headings):

        while d.textsize(txt)[0] > col_widths - padding[0]:
            txt = txt[:-1]
        d.text((col_widths * i + padding[0], + padding[1]), txt, fill=(1, 0, 0))

    raw_img_data = np.asarray(img, dtype="int32")

    return raw_img_data[:, :, 0]


def get_roi_dims(mask_list, size_mult=16):
    # This assumes each element in the mask list has the same dimensions
    masks = np.concatenate(mask_list, axis=0)
    masks = np.squeeze(masks)
    assert len(masks.shape) == 3

    lx, hx, ly, hy = 0, 0, 0, 0
    for y in range(masks.shape[2] - 1, 0, -1):
        if masks[:, :, y].max() == 1:
            hy = y
            break
    for y in range(masks.shape[2]):
        if masks[:, :, y].max() == 1:
            ly = y
            break
    for x in range(masks.shape[1] - 1, 0, -1):
        if masks[:, x, :].max() == 1:
            hx = x
            break
    for x in range(masks.shape[1]):
        if masks[:, x, :].max() == 1:
            lx = x
            break

    l = np.max([np.min([lx, ly]) - 10, 0])
    r = np.min([np.max([hx, hy]) + 10, masks.shape[2]])

    l, r = greatest_common_divisor(l, r, size_mult)

    return l, r


def greatest_common_divisor(l, r, size_mult):
    if (r - l) % size_mult != 0:
        div = (r - l) / size_mult
        if div * size_mult < (div + 1) * size_mult:
            diff = (r - l) - div * size_mult
            l += diff / 2
            r -= diff - (diff / 2)
        else:
            diff = (div + 1) * size_mult - (r - l)
            l -= diff / 2
            r += diff - (diff / 2)
    return int(l), int(r)


def process_contour(input_img, endocardium, epicardium=None):
    '''
    in each pixel we sample these 8 points:
     _________________
    |    *        *   |
    |  *            * |
    |                 |
    |                 |
    |                 |
    |  *            * |
    |    *        *   |
     ------------------
    we say a pixel is in the contour if half or more of these 8 points fall within the contour line
    '''
    segm_mask = np.zeros(shape=input_img.shape,dtype=input_img.dtype)

    contour_endo = pth.Path(endocardium, closed=True)
    contour_epi = pth.Path(epicardium, closed=True) if epicardium is not None else None
    for x in range(segm_mask.shape[1]):
        for y in range(segm_mask.shape[0]):
            for (dx, dy) in [(-0.25, -0.375), (-0.375, -0.25), (-0.25, 0.375), (-0.375, 0.25), (0.25, 0.375),
                             (0.375, 0.25), (0.25, -0.375), (0.375, -0.25)]:

                point = (x + dx, y + dy)
                if contour_epi is None and contour_endo.contains_point(point):
                    segm_mask[y, x] += 1
                elif contour_epi is not None and \
                        contour_epi.contains_point(point) and not contour_endo.contains_point(point):
                    segm_mask[y, x] += 1

    segm_mask = (segm_mask >= 4) * 1.
    return segm_mask


