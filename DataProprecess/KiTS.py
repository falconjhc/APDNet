import nibabel as nib
import os
import numpy as np
import random
import scipy.misc as misc

from utils.image_utils import image_show
# org_path = '/Users/harric/Downloads/MedicalData/KiTS/'
org_path = '/Volumes/HarricEd/kits/kits19/data'
# image = nib.load('/Users/harric/Downloads/MedicalData/KiTS/case_00000/imaging.nii.gz').get_data()
# mask = nib.load('/Users/harric/Downloads/MedicalData/KiTS/case_00000/segmentation.nii.gz').get_data()
# mask_kidney = np.zeros_like(mask)
# mask_tumour = np.zeros_like(mask)
# mask_kidney[np.where(np.logical_or(mask==1,mask==2))]=1
# mask_tumour[np.where(mask==2)]=1



save_path_root = org_path.replace(org_path.split('/')[-1],'')
save_path_root = save_path_root.replace('KiTS', 'KiTS_Processed1')
save_path_processed = os.path.join(save_path_root,'Processed_OnlyKidney1')
check_path = os.path.join(save_path_root,'Check_OnlyKidney1')
reshape_size = 256

center_v = 80
range_v = 30

def linear_scale(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return np.int16(img)

left_list, right_list, top_list, bottom_list = [],[],[],[]

def process_raw_data(data_list):

    kidney_counter=tumour_counter=kidney_tumour_counter=0
    kidney_counter_list=tumour_counter_list=kidney_tumour_counter_list=[]
    kidney_tumour_rate_list=[]
    for filecounter  in range(len(data_list)):
        if filecounter>209:
            break
        patiend_id = data_list[filecounter].split('_')[1]
        kidney_counter_patient = tumour_counter_patient = kidney_tumour_counter_patient = 0
        current_image_file = os.path.join(os.path.join(org_path,data_list[filecounter]), 'imaging.nii.gz')
        current_mask_file =  os.path.join(os.path.join(org_path,data_list[filecounter]), 'segmentation.nii.gz')

        image_data = nib.load(current_image_file).get_data()
        image_data = linear_scale(image_data)

        mask_data = nib.load(current_mask_file).get_data()

        kidney_mask = np.zeros(shape=mask_data.shape,dtype=mask_data.dtype)
        tumour_mask = np.zeros(shape=mask_data.shape, dtype=mask_data.dtype)
        kidney_mask[np.where(mask_data != 0)] = 1
        tumour_mask[np.where(mask_data == 2)] = 1
        kidney_mask = linear_scale(kidney_mask)
        tumour_mask = linear_scale(tumour_mask)
        assert mask_data.shape[0] == image_data.shape[0], "Inconsistent Slice number!"
        print("%d/%d: Shape: %d %d %d %d, SliceNum:%d" % (filecounter+1,
                                                          len(data_list),
                                                          mask_data.shape[1],mask_data.shape[2], image_data.shape[1],image_data.shape[2],
                                                          mask_data.shape[0]))
        current_patient_directory = 'Patient%s' % (patiend_id)
        current_patient_directory = os.path.join(save_path_processed, current_patient_directory)
        if not os.path.exists(current_patient_directory):
            os.makedirs(current_patient_directory)
        if not os.path.exists(check_path):
            os.makedirs(check_path)

        slice_number = mask_data.shape[0]
        for slice_traveller in range(slice_number):

            if (not slice_traveller==0) and slice_traveller%100==0:
                print("Slices: %d / %d; Kidney:%d, Tumour:%d" % (slice_traveller, slice_number, kidney_counter, tumour_counter))

            current_image = image_data[slice_traveller,:,:]
            current_kidneymask = kidney_mask[slice_traveller,:,:]
            current_tumourmask = tumour_mask[slice_traveller,:,:]

            if np.sum(current_kidneymask)==0:
                continue

            current_image = misc.imresize(current_image, (reshape_size,reshape_size), 'bicubic')
            current_kidneymask = misc.imresize(current_kidneymask, (reshape_size, reshape_size), 'nearest')
            current_tumourmask = misc.imresize(current_tumourmask, (reshape_size, reshape_size), 'nearest')

            if np.sum(current_kidneymask)>0:
                kidney_counter+=1
                kidney_counter_patient+=1
            if np.sum(current_tumourmask)>0:
                tumour_counter+=1
                tumour_counter_patient+=1
            if np.sum(current_kidneymask)>0 and np.sum(current_tumourmask)>0:
                kidney_tumour_counter+=1
                kidney_tumour_counter_patient+=1
                

            file_name_prefix = 'Patient%03d_Slice%03d' % (filecounter+1, slice_traveller+1)

            misc.imsave(os.path.join(current_patient_directory, file_name_prefix+'_tr.tiff'), current_image)
            misc.imsave(os.path.join(current_patient_directory, file_name_prefix + '_kidneymask.tiff'), current_kidneymask)
            misc.imsave(os.path.join(current_patient_directory, file_name_prefix + '_tumourmask.tiff'), current_tumourmask)

            current_image = np.expand_dims(current_image,axis=-1)
            current_image_for_check = np.copy(current_image)
            #current_image_for_check[np.where(current_image_for_check < center_v - range_v)] = center_v - range_v
            #current_image_for_check[np.where(current_image_for_check > center_v + range_v)] = center_v + range_v
            pixel_kidney = np.copy(current_image_for_check)
            pixel_tumour = np.copy(current_image_for_check)
            pixel_kidney[np.where(current_kidneymask==255)]=255
            pixel_tumour[np.where(current_tumourmask==255)]=255
            pixel_combined = np.concatenate([pixel_tumour, current_image_for_check, pixel_kidney], axis=-1)
            pixel_combined = np.concatenate([np.tile(current_image,[1,1,3]),
                                             pixel_combined], axis=1)



            misc.imsave(os.path.join(check_path,file_name_prefix+'.tiff'), pixel_combined)


        kidney_counter_list.append(kidney_counter_patient)
        tumour_counter_list.append(tumour_counter_patient)
        kidney_tumour_counter_list.append(kidney_tumour_counter_patient)
        kidney_tumour_rate_list.append(kidney_tumour_counter_patient/kidney_counter_patient * 100)
        print("Patient:%03d, SliceNum:%d, Kidney: %d, Tumours: %d, LiverTumour: %d (%.3f); "
              % (filecounter+1, mask_data.shape[2],
                 kidney_counter_patient, tumour_counter_patient, kidney_tumour_counter_patient,
                 kidney_tumour_counter_patient/kidney_counter_patient * 100))
    print("Total Kidney: %d, Total Tumour: %d, Total Kidneyr&&Tumour: %d;"
          % (kidney_counter, tumour_counter, kidney_tumour_counter))
    ranking = [index for index,value in sorted(list(enumerate(kidney_tumour_rate_list)),key=lambda x:x[1])]
    ranking = [index+1 for index in ranking]
    ranking.reverse()
    print(ranking)
    kidney_tumour_rate_list.sort()
    kidney_tumour_rate_list.reverse()
    print(kidney_tumour_rate_list)



data_list = [ii for ii in os.listdir(org_path) if not ii.startswith('.')]
process_raw_data(data_list)


print("Complete All !")