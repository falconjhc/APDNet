import nibabel as nib
import os
import numpy as np
import random
import scipy.misc as misc

from utils.image_utils import image_show


org_path = '/remote/rds/users/hjiang2/Data/Liver-CT/OrgData'
save_path_root = org_path.replace(org_path.split('/')[-1],'')
save_path_processed = os.path.join(save_path_root,'Processed_OnlyLivers')
check_path = os.path.join(save_path_root,'Check_OnlyLivers')
reshape_size = 256

center_v = 80
range_v = 30
sample_slices = 15

def linear_scale(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return np.int16(img)

left_list, right_list, top_list, bottom_list = [],[],[],[]
mean_v_liver_list, std_v_liver_list = [],[]
mean_v_liver_with_tumour, std_v_liver_with_tumour = [],[]
mean_v_liver_without_tumour, std_v_liver_without_tumour = [],[]
def process_raw_data(trlist,masklist):
    # ts_patient_id_list = []
    # for filecounter in range(len(tslist)):
    #     current_ts_file = tslist[filecounter]
    #     patient_id = current_ts_file.split('_')[1].split('.')[0]
    #     ts_patient_id_list.append(patient_id)
    liver_counter=tumour_counter=liver_tumour_counter=0
    liver_counter_list=tumour_counter_list=liver_tumour_counter_list=[]
    liver_tumour_rate_list=[]
    # mean_v_liver_list_cp, std_v_liver_list_cp = [], []
    # mean_v_liver_with_tumour_cp, std_v_liver_with_tumour_cp = [], []
    # mean_v_liver_without_tumour_cp, std_v_liver_without_tumour_cp = [], []
    for filecounter  in range(len(trlist)):
    #for filecounter in range(3):
        liver_counter_patient = tumour_counter_patient = liver_tumour_counter_patient = 0
        current_tr_file = trlist[filecounter]
        current_mask_file = masklist[filecounter]
        patient_id_from_tr = current_tr_file.split('_')[1].split('.')[0]
        patient_id_from_mask = current_tr_file.split('_')[1].split('.')[0]
        assert patient_id_from_tr==patient_id_from_mask, 'not the same patient!'

        current_tr_file_path = os.path.join(os.path.join(org_path,'imagesTr'), current_tr_file)
        tr_data = nib.load(current_tr_file_path).get_data()
        tr_data = linear_scale(tr_data)

        current_mask_file_path = os.path.join(os.path.join(org_path, 'labelsTr'), current_mask_file)
        mask_data = nib.load(current_mask_file_path).get_data()

        liver_mask = np.zeros(shape=mask_data.shape,dtype=mask_data.dtype)
        tumour_mask = np.zeros(shape=mask_data.shape, dtype=mask_data.dtype)
        liver_mask[np.where(mask_data!=0)] = 1
        tumour_mask[np.where(mask_data == 2)] = 1
        liver_mask = linear_scale(liver_mask)
        tumour_mask = linear_scale(tumour_mask)
        # print("%d/%d: Shape: %d %d %d %d, SliceNum:%d" % (filecounter,
        #                                                   len(trlist),
        #                                                   mask_data.shape[0],mask_data.shape[1], tr_data.shape[0],tr_data.shape[1],
        #                                                   mask_data.shape[2]))
        assert mask_data.shape[0]==512 and tr_data.shape[0]==512, 'incorrect shape'

        current_patient_directory = 'Patient%03d' % (filecounter+1)
        current_patient_directory = os.path.join(save_path_processed, current_patient_directory)
        if not os.path.exists(current_patient_directory):
            os.makedirs(current_patient_directory)
        if not os.path.exists(check_path):
            os.makedirs(check_path)

        assert mask_data.shape[2] == tr_data.shape[2], 'inconsistent slice number'
        slice_number = mask_data.shape[2]
        for slice_traveller in range(slice_number):

            # if (not slice_traveller==0) and slice_traveller%100==0:
            #     print("Slices: %d / %d; Liver%d, Tumour:%d" % (slice_traveller, slice_number, liver_counter, tumour_counter))

            current_tr = tr_data[:,:,slice_traveller]
            current_livermask = liver_mask[:,:,slice_traveller]
            current_tumourmask = tumour_mask[:,:,slice_traveller]

            if np.sum(current_livermask)==0:
                continue

            #current_tr = misc.imresize(current_tr, (reshape_size,reshape_size), 'bicubic')
            #current_livermask = misc.imresize(current_livermask, (reshape_size, reshape_size), 'nearest')
            #current_tumourmask = misc.imresize(current_tumourmask, (reshape_size, reshape_size), 'nearest')

            if np.sum(current_livermask)>0:
                liver_counter+=1
                liver_counter_patient+=1
            if np.sum(current_tumourmask)>0:
                tumour_counter+=1
                tumour_counter_patient+=1
            if np.sum(current_livermask)>0 and np.sum(current_tumourmask)>0:
                liver_tumour_counter+=1
                liver_tumour_counter_patient+=1
                

            file_name_prefix = 'Patient%03d_Slice%03d' % (filecounter+1, slice_traveller+1)

            #misc.imsave(os.path.join(current_patient_directory, file_name_prefix+'_tr.tiff'), current_tr)
            #misc.imsave(os.path.join(current_patient_directory, file_name_prefix + '_livermask.tiff'), current_livermask)
            #misc.imsave(os.path.join(current_patient_directory, file_name_prefix + '_tumourmask.tiff'), current_tumourmask)

            current_tr = np.expand_dims(current_tr,axis=-1)
            current_tr_for_check = np.copy(current_tr)
            current_tr_for_check[np.where(current_tr_for_check < center_v - range_v)] = center_v - range_v
            current_tr_for_check[np.where(current_tr_for_check > center_v + range_v)] = center_v + range_v
            pixel_liver = np.copy(current_tr_for_check)
            pixel_tumour = np.copy(current_tr_for_check)
            pixel_liver[np.where(current_livermask==255)]=255
            pixel_tumour[np.where(current_tumourmask==255)]=255
            pixel_combined = np.concatenate([pixel_tumour, current_tr_for_check, pixel_liver], axis=-1)
            pixel_combined = np.concatenate([np.tile(current_tr,[1,1,3]),
                                             np.tile(current_tr_for_check,[1,1,3]),
                                             pixel_combined], axis=1)

            if np.sum(current_livermask)>0:
                mean_v = np.mean(current_tr[np.where(current_livermask==255)])
                std_v = np.std(current_tr[np.where(current_livermask==255)])
                mean_v_liver_list.append(mean_v)
                std_v_liver_list.append(std_v)
                if np.sum(current_tumourmask)>0:
                    mean_v_liver_with_tumour.append(mean_v)
                    std_v_liver_with_tumour.append(std_v)
                else:
                    mean_v_liver_without_tumour.append(mean_v)
                    std_v_liver_without_tumour.append(std_v)
        # mean_v_liver_list.append(np.mean(mean_v_liver_list_cp))
        # std_v_liver_list.append(np.mean(std_v_liver_list_cp))
        # mean_v_liver_with_tumour.append(np.mean(mean_v_liver_with_tumour_cp))
        # std_v_liver_with_tumour.append(np.mean(std_v_liver_with_tumour_cp))
        # mean_v_liver_without_tumour.append(np.mean(mean_v_liver_without_tumour_cp))
        # std_v_liver_without_tumour.append(np.mean(std_v_liver_without_tumour_cp))


            misc.imsave(os.path.join(check_path,file_name_prefix+'.tiff'), pixel_combined)
        liver_counter_list.append(liver_counter_patient)
        tumour_counter_list.append(tumour_counter_patient)
        liver_tumour_counter_list.append(liver_tumour_counter_patient)
        liver_tumour_rate_list.append(liver_tumour_counter_patient/liver_counter_patient * 100)
        print("Patient:%03d, SliceNum:%d, Liver: %d, Tumours: %d, LiverTumour: %d (%.3f); Mean:%.3f/%.3f/%.3f, Std:%.3f/%.3f/%.3f, Length:%d/%d/%d/%d/%d/%d"
              % (filecounter+1, mask_data.shape[2],
                 liver_counter_patient, tumour_counter_patient, liver_tumour_counter_patient,
                 liver_tumour_counter_patient/liver_counter_patient * 100,
                 np.mean(mean_v_liver_list), np.mean(mean_v_liver_with_tumour), np.mean(mean_v_liver_without_tumour),
                 np.mean(std_v_liver_list), np.mean(std_v_liver_with_tumour), np.mean(std_v_liver_without_tumour),
                 len(mean_v_liver_list), len(mean_v_liver_with_tumour), len(mean_v_liver_without_tumour),
                 len(std_v_liver_list), len(std_v_liver_with_tumour), len(std_v_liver_without_tumour)))
    print("Total Liver: %d, Total Tumour: %d, Total Liver&&Tumour: %d;"
          % (liver_counter, tumour_counter, liver_tumour_counter))
    ranking = [index for index,value in sorted(list(enumerate(liver_tumour_rate_list)),key=lambda x:x[1])]
    ranking = [index+1 for incurrent_mask_filedex in ranking]
    ranking.reverse()
    print(ranking)
    liver_tumour_rate_list.sort()
    liver_tumour_rate_list.reverse()
    print(liver_tumour_rate_list)()
    print(np.max(mean_v_liver_list))
    print(np.min(mean_v_liver_list))
    print(np.max(mean_v_liver_with_tumour))
    print(np.min(mean_v_liver_with_tumour))
    print(np.max(mean_v_liver_without_tumour))
    print(np.min(mean_v_liver_without_tumour))


tr_list = os.listdir(os.path.join(org_path,'imagesTr'))
ts_list = os.listdir(os.path.join(org_path,'imagesTs'))
mask_list = os.listdir(os.path.join(org_path,'labelsTr'))
tr_list = [file for file in tr_list if not file.startswith('.')]
ts_list = [file for file in ts_list if not file.startswith('.')]
mask_list = [file for file in mask_list if not file.startswith('.')]

tr_list.sort()
ts_list.sort()
mask_list.sort()

choice_indices = random.sample(list(range(len(tr_list))), sample_slices)
tr_list_new, mask_list_new = [],[]
for ii in choice_indices:
    tr_list_new.append(tr_list[ii])
    mask_list_new.append(mask_list[ii])
process_raw_data(tr_list_new, mask_list_new)


print("Complete All !")