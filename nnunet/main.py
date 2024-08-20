#libraries
import shutil
from collections import OrderedDict
import json
import numpy as np

#visualization of the dataset
import matplotlib.pyplot as plt
import nibabel as nib
import os

#%% base fiel paths
base_dir = "/Users/maxxyouu/Desktop/bowang" # parent directory of nnUnet
repository_dir = os.path.join(base_dir,'nnUNet')
os.system('pip install -e .')
os.chdir(base_dir)

if os.getcwd()==base_dir:
    print('We are in the correct directory')
else:
    print("Run set base directory step again, then check to verify.")

#%% make sure dataset folder structure
task_name = 'Task004_Hippocampus' #change here for different task name
nnunet_dir = "nnUNet/nnunet/nnUNet_raw_data_base/nnUNet_raw_data"
task_folder_name = os.path.join(nnunet_dir,task_name)
train_image_dir = os.path.join(task_folder_name,'imagesTr')
train_label_dir = os.path.join(task_folder_name,'labelsTr')
test_dir = os.path.join(task_folder_name,'imagesTs')
main_dir = os.path.join(base_dir,'nnUNet/nnunet')

#%% enviroment variables
os.environ['nnUNet_raw_data_base'] = os.path.join(main_dir,'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(main_dir,'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(main_dir,'nnUNet_trained_models')

#%% verification of the dataset
train_files = os.listdir(train_image_dir)
label_files = os.listdir(train_label_dir)
print("train image files:",len(train_files))
print("train label files:",len(label_files))
print("Matches:",len(set(train_files).intersection(set(label_files))))

def check_modality(filename):
    """
    check for the existence of modality
    return False if modality is not found else True
    """
    end = filename.find('.nii.gz')
    modality = filename[end-4:end]
    for mod in modality: 
        if not(ord(mod)>=48 and ord(mod)<=57): #if not in 0 to 9 digits
            return False
    return True

def rename_for_single_modality(directory):
    
    for file in os.listdir(directory):
        
        if check_modality(file)==False:
            new_name = file[:file.find('.nii.gz')]+"_0000.nii.gz"
            os.rename(os.path.join(directory,file),os.path.join(directory,new_name))
            print(f"Renamed to {new_name}")
        else:
            print(f"Modality present: {file}")

## note that both folder need to follow the same file format set by nnunet_v1
rename_for_single_modality(train_image_dir)
rename_for_single_modality(test_dir)

#%%running it from the experiment_planning folder to verify the path settings
# this is the data preprocessing step with default ExperimentPlanner3D_v21
os.chdir(main_dir)
os.system('python experiment_planning/nnUNet_plan_and_preprocess.py -t 04 --verify_dataset_integrity')
os.chdir(base_dir)


#%% dataset visualization
#visualizing some of the training images and labels
# (re-run to see random pick-ups)
# only maximum of first 5 slices are plotted
# train_img_name = os.listdir(train_image_dir)[1]
# train_img = np.array(nib.load(os.path.join(train_image_dir,train_img_name)).dataobj)[:,:,:5]
# train_label_name = train_img_name[:train_img_name.find('_0000.nii.gz')]+'.nii.gz'
# train_label = np.array(nib.load(os.path.join(train_label_dir,train_label_name)).dataobj)[:,:,:5]

# print(train_img.shape,train_label.shape)

# max_rows = 2
# max_cols = train_img.shape[2]

# fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,8))
# for idx in range(max_cols):
#     axes[0, idx].axis("off") 
#     axes[0, idx].set_title('Train Image'+str(idx+1))
#     axes[0 ,idx].imshow(train_img[:,:,idx], cmap="gray")
# for idx in range(max_cols):    
#     axes[1, idx].axis("off")
#     axes[1, idx].set_title('Train Label'+str(idx+1))
#     axes[1, idx].imshow(train_label[:,:,idx])
# # note that since we only take out the first 5 slices, might not have any labels
# plt.subplots_adjust(wspace=.1, hspace=.1)
# plt.show()
# %% train basic model
os.chdir(main_dir)
os.system('nnUNet_train 3d_fullres nnUNetTrainerV2 004 0')
os.chdir(base_dir)
