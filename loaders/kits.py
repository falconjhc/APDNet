
import os
import scipy.io as sio
import nibabel as nib
import numpy as np
from skimage import transform
from PIL import Image

import utils.data_utils
from loaders.base_loader import Loader

from loaders.data import Data
from parameters import conf
import logging
from utils.image_utils import image_show
import cv2

class KitsLoader(Loader):

    def __init__(self):
        super(KitsLoader, self).__init__()
        self.num_anato_masks = 1
        self.num_patho_masks = 1
        self.num_volumes = 75
        self.input_shape = (256, 256, 1)
        self.data_folder = conf['kits']
        self.log = logging.getLogger('kits')

    def sub_mask_generation(self, whole_mask, org_sub):

        mask_num = len(whole_mask)
        output_sub = []
        for ii in range(mask_num):
            current_whole = whole_mask[ii]
            current_sub = org_sub[ii]
            corrected_sub = current_whole - current_sub
            corrected_sub[np.where(corrected_sub==-1)]=0
            output_sub.append(corrected_sub)

        return output_sub

    def splits(self):
        """
        :return: an array of splits into validation, test and train indices
        """
        # valid_volume = [vol for vol in os.listdir(self.data_folder)
        #                 if (not vol[0]=='.'
        #                     and os.path.isdir(os.path.join(self.data_folder,
        #                                                    os.path.join(vol,'LGE'))))]
        # total_vol_num = len(valid_volume)
        # split_train_num_0 = 80
        # train_num_0 = np.float(split_train_num_0) / 100.0 * total_vol_num
        splits = [
            # {'validation': list(range(115,131)), # --> test on p11
            #  'test': list(range(101,115)),
            #  'training': list(range(1,101))
            #  },

            {'validation': [136,108,80,26,119],  # --> test on p11
             'test': [79,89,185,74,198],
             'training': [31,115,43,140,93,103,71,193,152,68,173,56,109,9,182,27,179,188,57,48]
             },


        ]

        return splits

    def load_labelled_data(self, split, split_type, modality='LGE',
                           normalise=True, value_crop=True, downsample=1, segmentation_option=-1):
        """
        Load labelled data, and return a Data object. In ACDC there are ES and ED annotations. Preprocessed data
        are saved in .npz files. If they don't exist, load the original images and preprocess.

        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :param downsample:      Downsample data to smaller size. Only used for testing.
        :return:                a Data object
        """
        # if segmentation_option == 0:
        #     input("Segmentation 0")

        if split < 0 or split > 4:
            raise ValueError('Invalid value for split: %d. Allowed values are 0, 1, 2.' % split)
        if split_type not in ['training', 'validation', 'test', 'all']:
            raise ValueError('Invalid value for split_type: %s. Allowed values are training, validation, test, all'
                             % split_type)

        npz_prefix = 'norm_' if normalise else 'unnorm_'

        def _only_get_pahtology_data():
            data_num = masks_tumour.shape[0]
            new_images, new_anato_masks, new_patho_masks,new_index, news_slice = [],[],[],[],[]
            for ii in range(data_num):
                if np.sum(patho_masks[ii,:,:,:])==0:
                    continue
                new_images.append(np.expand_dims(images[ii,:,:,:],axis=0))
                new_anato_masks.append(np.expand_dims(anato_masks[ii,:,:,:],axis=0))
                new_patho_masks.append(np.expand_dims(patho_masks[ii,:,:,:], axis=0))
                new_index.append(index[ii])
                news_slice.append(slice[ii])
            new_images = np.concatenate(new_images)
            new_anato_masks = np.concatenate(new_anato_masks)
            new_patho_masks = np.concatenate(new_patho_masks)
            new_index = np.concatenate(np.expand_dims(new_index,axis=0))
            news_slice = np.concatenate(np.expand_dims(news_slice,axis=0))
            return new_images, new_anato_masks, new_patho_masks,new_index,news_slice





        # If numpy arrays are not saved, load and process raw data
        if not os.path.exists(os.path.join(self.data_folder, npz_prefix + 'kits_image.npz')):
            if modality == 'LGE':
                value_crop = False
            images, masks_kidney, masks_tumour, patient_index,index, slice = \
                self.load_raw_labelled_data(normalise, value_crop)

            # save numpy arrays
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'kits_image'), images)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'kits_kidney_mask'), masks_kidney)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'kits_tumour_mask'), masks_tumour)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'kits_patienet_index'), patient_index)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'kits_index'),index)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'kits_slice'), slice)
        # Load data from saved numpy arrays
        else:
            images        = np.load(os.path.join(self.data_folder, npz_prefix + 'kits_image.npz'))['arr_0']
            masks_kidney      = np.load(os.path.join(self.data_folder, npz_prefix + 'kits_kidney_mask.npz'))['arr_0']
            masks_tumour = np.load(os.path.join(self.data_folder, npz_prefix + 'kits_tumour_mask.npz'))['arr_0']
            patient_index = np.load(os.path.join(self.data_folder, npz_prefix + 'kits_index.npz'))['arr_0']
            index = np.load(os.path.join(self.data_folder, npz_prefix + 'kits_index.npz'))['arr_0']
            slice = np.load(os.path.join(self.data_folder, npz_prefix + 'kits_slice.npz'))['arr_0']


        assert images is not None and masks_kidney is not None and masks_tumour is not None \
               and index is not None, 'Could not find saved data'

        assert images.max() == 1 and images.min() == -1, \
            'Images max=%.3f, min=%.3f' % (images.max(), images.min())

        self.log.debug('Loaded compressed kits data of shape: ' + str(images.shape) + ' ' + str(index.shape))

        anato_masks = masks_kidney
        patho_masks = masks_tumour
        anato_mask_names = ['kidney']
        patho_mask_names = ['tumour']

        images, anato_masks, patho_masks, index, slice = _only_get_pahtology_data()

        assert anato_masks.max() == 1 and anato_masks.min() == 0, 'Anatomy Masks max=%.3f, min=%.3f' \
                                                                  % (anato_masks.max(), anato_masks.min())
        assert patho_masks.max() == 1 and patho_masks.min() == 0, 'Pathology Masks max=%.3f, min=%.3f' \
                                                                  % (anato_masks.max(), anato_masks.min())

        scanner = np.array([modality] * index.shape[0])

        # Select images belonging to the volumes of the split_type (training, validation, test)
        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        anato_masks = np.concatenate([anato_masks[index == v] for v in volumes])
        patho_masks = np.concatenate([patho_masks[index == v] for v in volumes])

        assert images.shape[0] == anato_masks.shape[0] == patho_masks.shape[0], "Num of Images inconsistent"

        # create a volume index
        slice = np.concatenate([slice[index == v] for v in volumes])
        index = np.concatenate([index[index == v] for v in volumes])
        scanner = np.array([modality] * index.shape[0])
        assert images.shape[0] == index.shape[0]

        self.log.debug(split_type + ' set: ' + str(images.shape))
        return Data(images, [anato_masks, patho_masks], [anato_mask_names, patho_mask_names], index, slice, scanner, downsample)


    def load_unlabelled_data(self, split, split_type, modality='LGE', normalise=True, value_crop=True):
        """
        Load unlabelled data. In ACDC, this contains images from the cardiac phases between ES and ED.
        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :return:                a Data object
        """
        images, index, slice = self.load_unlabelled_images('kits', split, split_type, False, normalise, value_crop,modality=modality)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        scanner = np.array([modality] * index.shape[0])
        return Data(images, masks, '-1', index, slice, scanner)

    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True, segmentation_option='-1'):
        """
        Load all images, unlabelled and labelled, meaning all images from all cardiac phases.
        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :return:                a Data object
        """
        images, index, slice = self.load_unlabelled_images('kits', split, split_type, True, normalise, value_crop,modality=modality)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        scanner = np.array([modality] * index.shape[0])
        return Data(images, masks, '-1', index, slice, scanner)

    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        """
        Load labelled data iterating through the ACDC folder structure.
        :param normalise:   normalise data between -1, 1
        :param value_crop:  crop between 5 and 95 percentile
        :return:            a tuple of the image and mask arrays
        """
        self.log.debug('Loading kidney-ct data from original location')
        images, masks_kidney, masks_tumour, patient_index, index, slice = [], [], [], [], [], []
        existed_directories = [vol for vol in os.listdir(self.data_folder)
                               if (not vol.startswith('.')) and os.path.isdir(os.path.join(self.data_folder, vol))]
        existed_directories.sort()
        # assert len(existed_directories) == len(self.volumes), 'Incorrect Volume Num !'

        self.volumes = np.unique(self.volumes)
        self.volumes.sort()

        for patient_counter, patient_i in enumerate(self.volumes):

            patient_image, patient_kidney, patient_tumour = [], [], []
            # if not os.path.isdir(os.path.join(self.data_folder,existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]

            print('Extracting Labeled Patient: %s @ %d / %d' % (patient, patient_counter+1, len(self.volumes)))


            patient_folder = os.path.join(self.data_folder,patient)
            img_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('tr')==-1)]
            kidney_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('kidneymask') == -1)]
            tumour_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('tumourmask') == -1)]
            img_file_list.sort()
            kidney_file_list.sort()
            tumour_file_list.sort()
            slices_num = len(img_file_list)

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)

            volume_num = len(img_file_list)
            for v in range(volume_num):
                current_img_name = img_file_list[v]
                current_kidney_name = kidney_file_list[v]
                current_tumour_name = tumour_file_list[v]
                v_id_from_img = current_img_name.split('_')[1]
                v_id_from_kidney = current_kidney_name.split('_')[1]
                v_id_from_tumour = current_tumour_name.split('_')[1]
                assert v_id_from_img==v_id_from_kidney==v_id_from_tumour, 'Mis-Alignment !'
                slice.append(v_id_from_img[5:])


            # for original images
            for org_img_path in img_file_list:
                im = np.array(Image.open(os.path.join(patient_folder,org_img_path)))
                # im = im / np.max(im - np.min(im))
                # im = im[:,:,0]
                patient_image.append(np.expand_dims(im,axis=-1))
            patient_image = np.concatenate(patient_image, axis=-1)


            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_image.flatten(), 5)
                p95 = np.percentile(patient_image.flatten(), 95)
                patient_image = np.clip(patient_image, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_image = utils.data_utils.normalise(patient_image, -1, 1)
            images.append(np.expand_dims(patient_image,axis=-1))

            for kidney_seg_path in kidney_file_list:
                kidney = np.array(Image.open(os.path.join(patient_folder,kidney_seg_path)))
                if not (len(np.unique(kidney)) == 1 and np.unique(kidney)[0] == 0):
                    kidney = kidney / np.max(kidney)
                patient_kidney.append(np.expand_dims(kidney, axis=-1))
            patient_kidney = np.concatenate(patient_kidney,axis=-1)
            masks_kidney.append(np.expand_dims(patient_kidney,axis=-1))

            for tumour_seg_path in tumour_file_list:
                tumour = np.array(Image.open(os.path.join(patient_folder,tumour_seg_path)))
                if not (len(np.unique(tumour)) == 1 and np.unique(tumour)[0] == 0):
                    tumour = tumour / np.max(tumour)
                patient_tumour.append(np.expand_dims(tumour, axis=-1))
            patient_tumour = np.concatenate(patient_tumour,axis=-1)
            masks_tumour.append(np.expand_dims(patient_tumour, axis=-1))


        # move slice axis to the first position
        images = [np.moveaxis(im, 2, 0) for im in images]
        masks_kidney = [np.moveaxis(m, 2, 0) for m in masks_kidney]
        masks_tumour = [np.moveaxis(m, 2, 0) for m in masks_tumour]

        # crop images and masks to the same pixel dimensions and concatenate all data
        images_cropped, masks_kidney_cropped = utils.data_utils.crop_same(images, masks_kidney,
                                                                      (self.input_shape[0], self.input_shape[1]))
        _, masks_tumour_cropped = utils.data_utils.crop_same(images, masks_tumour,
                                                         (self.input_shape[0], self.input_shape[1]))


        images_cropped = np.concatenate(images_cropped, axis=0)
        masks_tumour_cropped = np.concatenate(masks_tumour_cropped, axis=0)
        masks_kidney_cropped = np.concatenate(masks_kidney_cropped, axis=0)
        patient_index = np.array(patient_index)
        index = np.array(index)
        slice = np.array(slice)


        return images_cropped, masks_kidney_cropped, masks_tumour_cropped, patient_index, index, slice

    def resample_raw_image(self, mask_fname, patient_folder, binary=True):
        """
        Load raw data (image/mask) and resample to fixed resolution.
        :param mask_fname:     filename of mask
        :param patient_folder: folder containing patient data
        :param binary:         boolean to define binary masks or not
        :return:               the resampled image
        """
        m_nii_fname = os.path.join(patient_folder, mask_fname)
        new_res = (1.37, 1.37)
        print('Resampling %s at resolution %s to file %s' % (m_nii_fname, str(new_res), new_res))
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()

        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1

        result = []
        for i in range(im_data.shape[-1]):
            im = im_data[..., i]
            rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
            result.append(np.expand_dims(rescaled, axis=-1))
        return np.concatenate(result, axis=-1)

    def process_raw_image(self, im_fname, patient_folder, value_crop, normalise):
        """
        Rescale between -1 and 1 and crop extreme values of an image
        :param im_fname:        filename of the image
        :param patient_folder:  folder of patient data
        :param value_crop:      True/False to crop values between 5/95 percentiles
        :param normalise:       True/False normalise images
        :return:                a processed image
        """
        im = self.resample_raw_image(im_fname, patient_folder, binary=False)

        # crop to 5-95 percentile
        if value_crop:
            p5 = np.percentile(im.flatten(), 5)
            p95 = np.percentile(im.flatten(), 95)
            im = np.clip(im, p5, p95)

        # normalise to -1, 1
        if normalise:
            im = utils.data_utils.normalise(im, -1, 1)

        return im

    def load_raw_unlabelled_data(self, include_labelled=True, normalise=True, value_crop=True, modality='LGE'):
        """
        Load unlabelled data iterating through the ACDC folder structure.
        :param include_labelled:    include images from ES, ED phases that are labelled. Can be True/False
        :param normalise:           normalise data between -1, 1
        :param value_crop:          crop between 5 and 95 percentile
        :return:                    an image array
        """
        self.log.debug('Loading unlabelled kits data from original location')
        images, patient_index, index, slice = [], [], [], []
        existed_directories = [vol for vol in os.listdir(self.data_folder)
                               if (not vol.startswith('.')) and os.path.isdir(os.path.join(self.data_folder,vol))]
        existed_directories.sort()
        # assert len(existed_directories) == len(self.volumes), 'Incorrect Volume Num !'

        self.volumes = np.unique(self.volumes)
        self.volumes.sort()

        for patient_counter, patient_i in enumerate(self.volumes):
            patient_images = []
            # if not os.path.isdir(os.path.join(self.data_folder,existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]
            print('Extracting UnLabeled Patient: %s @ %d / %d' % (patient, patient_counter+1, len(self.volumes)))

            patient_folder = os.path.join(self.data_folder, patient)
            img_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('tr') == -1)]
            img_file_list.sort()
            slices_num = len(img_file_list)
            for v in range(slices_num):
                current_img_name = img_file_list[v]
                v_id_from_img = current_img_name.split('_')[1]
                slice.append(v_id_from_img[5:])

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)

            # for original images
            for org_img_path in img_file_list:
                im = np.array(Image.open(os.path.join(patient_folder, org_img_path)))
                # im = im / np.max(im - np.min(im))
                # im = im[:, :, 0]
                patient_images.append(np.expand_dims(im, axis=-1))
            patient_images = np.concatenate(patient_images, axis=-1)

            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_images.flatten(), 5)
                p95 = np.percentile(patient_images.flatten(), 95)
                patient_images = np.clip(patient_images, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_images = utils.data_utils.normalise(patient_images, -1, 1)
            images.append(np.expand_dims(patient_images, axis=-1))


        images = [np.moveaxis(im, 2, 0) for im in images]
        zeros = [np.zeros(im.shape) for im in images]
        images_cropped, _ = utils.data_utils.crop_same(images, zeros,
                                                       (self.input_shape[0], self.input_shape[1]))
        images_cropped = np.concatenate(images_cropped, axis=0)[..., 0]
        index = np.array(index)
        slice = np.array(slice)

        return images_cropped, patient_index, index, slice

    def load_unlabelled_images(self, dataset, split, split_type, include_labelled, normalise, value_crop, modality):
        """
        Load only images.
        :param dataset:
        :param split:
        :param split_type:
        :param include_labelled:
        :param normalise:
        :param value_crop:
        :return:
        """
        npz_prefix_type = 'ul_' if not include_labelled else 'all_'
        npz_prefix = npz_prefix_type + 'norm_' if normalise else npz_prefix_type + 'unnorm_'

        # Load saved numpy array
        if os.path.exists(os.path.join(self.data_folder, npz_prefix + 'kits_image.npz')):
            images = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'kits_image.npz'))['arr_0']
            index  = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'kits_index.npz'))['arr_0']
            patient_index = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'kits_patient_index.npz'))['arr_0']
            slice = \
                np.load(os.path.join(self.data_folder,
                                     npz_prefix + 'kits_patient_slice.npz'))['arr_0']
            self.log.debug('Loaded compressed ' + dataset + ' unlabelled data of shape ' + str(images.shape))
        # Load from source
        else:
            if modality == 'LGE':
                value_crop = False
            images, patient_index, index, slice = \
                self.load_raw_unlabelled_data(include_labelled, normalise, value_crop, modality=modality)
            images = np.expand_dims(images, axis=3)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'kits_image'), images)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'kits_index'), index)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'kits_patient_index'), patient_index)
            np.savez_compressed(os.path.join(self.data_folder,
                                             npz_prefix + 'kits_patient_slice'), slice)
        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            return images, index

        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        slice = np.concatenate([slice[index == v] for v in volumes])
        index  = np.concatenate([index[index==v] for v in volumes])
        return images, index, slice